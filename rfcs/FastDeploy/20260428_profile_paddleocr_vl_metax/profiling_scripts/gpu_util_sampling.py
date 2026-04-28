"""GPU utilization sampling during inference with mx-smi polling.

Modes:
  Default (S1 only):     python gpu_util_sampling.py
  Separate phases (S1+S4, Prefill/Decode split):
                         python gpu_util_sampling.py --separate-phases

Output:
  Default:    output/gpu_utilization_detailed.json
  Phases:     output/gpu_utilization_phases.json
"""
import argparse
import os
import time
import json
import subprocess
import threading

# Env vars must be set before importing paddle — run: source setup_env.sh

import paddle
from fastdeploy import LLM, SamplingParams

paddle.device.set_device('metax_gpu:0')

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_INTERVAL = 0.2  # 200ms
MX_SMI_CMD = ['mx-smi', '--show-usage', '--show-memory',
              '--show-hbm-bandwidth', '--show-core-usage']


def parse_mx_smi_output(output):
    """Parse mx-smi output into GPU metrics dict. Returns None on parse failure."""
    gpu_util = None
    mem_used_kb = 0
    mem_total_kb = 0
    hbm_bw = 0
    ccx_vals = []

    for line in output.split('\n'):
        if 'GPU' in line and ':' in line and '%' in line and 'VPUE' not in line and 'VPUD' not in line:
            parts = line.split(':')
            if len(parts) >= 2:
                val = parts[-1].strip().replace('%', '').strip()
                try:
                    gpu_util = int(val)
                except ValueError:
                    pass
        elif 'vis_vram used' in line:
            parts = line.split(':')
            if len(parts) >= 2:
                try:
                    mem_used_kb = int(parts[-1].strip().split()[0])
                except ValueError:
                    pass
        elif 'vis_vram total' in line:
            parts = line.split(':')
            if len(parts) >= 2:
                try:
                    mem_total_kb = int(parts[-1].strip().split()[0])
                except ValueError:
                    pass
        elif 'throughput' in line:
            parts = line.split(':')
            if len(parts) >= 2:
                val = parts[-1].strip().split()[0]
                try:
                    hbm_bw = int(val)
                except ValueError:
                    pass
        elif 'CCX' in line:
            tokens = line.replace('CCX', '').split()
            for i in range(0, len(tokens) - 1, 2):
                try:
                    ccx_vals.append(int(tokens[i]))
                except ValueError:
                    pass

    # gpu_util stays None if we never found a GPU line — treat as parse failure
    if gpu_util is None:
        return None

    ccx_avg = int(sum(ccx_vals) / len(ccx_vals)) if ccx_vals else 0

    return {
        'timestamp': time.time(),
        'gpu_util_pct': gpu_util,
        'mem_used_mb': mem_used_kb // 1024,
        'mem_total_mb': mem_total_kb // 1024,
        'hbm_bw_mbytes': hbm_bw,
        'ccx_avg_pct': ccx_avg,
    }


def sample_gpu(samples, active_flag, start_barrier):
    """Background thread to sample GPU metrics via mx-smi.

    Args:
        samples: list to append results to (owned by caller)
        active_flag: list[bool], active_flag[0] is the loop control
        start_barrier: threading.Barrier(2) to sync with inference start
    """
    start_barrier.wait()
    while active_flag[0]:
        try:
            r = subprocess.run(MX_SMI_CMD, capture_output=True, text=True, timeout=3)
            if r.returncode != 0:
                samples.append({'timestamp': time.time(), 'error': f'mx-smi rc={r.returncode}'})
            else:
                parsed = parse_mx_smi_output(r.stdout)
                if parsed is not None:
                    samples.append(parsed)
                else:
                    samples.append({'timestamp': time.time(), 'error': 'mx-smi parse failed: no GPU line'})
        except subprocess.TimeoutExpired:
            samples.append({'timestamp': time.time(), 'error': 'mx-smi timeout'})
        except Exception as e:
            samples.append({'timestamp': time.time(), 'error': str(e)})

        time.sleep(SAMPLE_INTERVAL)


def run_inference_with_sampling(llm, prompt, sampling_params):
    """Run a single inference with GPU sampling, return (output, e2e, ttft, samples)."""
    samples = []
    active_flag = [True]
    # Barrier ensures sampler thread starts before inference begins
    start_barrier = threading.Barrier(2, timeout=10)

    sampler_thread = threading.Thread(
        target=sample_gpu, args=(samples, active_flag, start_barrier), daemon=True,
    )
    sampler_thread.start()

    # Wait for sampler to be ready before starting inference
    start_barrier.wait()

    paddle.device.synchronize("metax_gpu:0")
    start = time.perf_counter()
    output = llm.chat(prompt, sampling_params=sampling_params)
    paddle.device.synchronize("metax_gpu:0")
    end = time.perf_counter()

    active_flag[0] = False
    sampler_thread.join(timeout=5)

    e2e = end - start
    n_tokens = len(output[0].outputs.token_ids)
    ttft = None
    if output[0].metrics is not None and hasattr(output[0].metrics, 'first_token_time'):
        ttft = output[0].metrics.first_token_time

    return output, e2e, ttft, samples


def summarize_samples(samples, e2e, ttft, n_tokens, separate_phases=False):
    """Build result dict from GPU samples."""
    valid = [s for s in samples if 'error' not in s]
    if not valid:
        errors = [s for s in samples if 'error' in s]
        error_detail = errors[0]['error'] if errors else 'unknown'
        return {'e2e_s': round(e2e, 3), 'error': f'no valid GPU samples ({len(samples)} total, first: {error_detail})'}

    utils = [s['gpu_util_pct'] for s in valid]
    bws = [s['hbm_bw_mbytes'] for s in valid]
    ccxs = [s['ccx_avg_pct'] for s in valid]

    result = {
        'e2e_s': round(e2e, 3),
        'ttft_s': round(ttft, 3) if ttft else None,
        'tokens': n_tokens,
        'throughput_tok_s': round(n_tokens / e2e, 2),
        'gpu_overall': {
            'util_avg': round(sum(utils) / len(utils), 1),
            'util_max': max(utils),
            'hbm_bw_avg_gb': round(sum(bws) / len(bws) / 1024, 2),
            'hbm_bw_max_gb': round(max(bws) / 1024, 2),
            'ccx_avg': round(sum(ccxs) / len(ccxs), 1),
        },
        'num_samples': len(valid),
    }

    if separate_phases:
        t0 = valid[0]['timestamp']
        split_time = ttft if ttft else 0.25
        prefill = [s for s in valid if s['timestamp'] - t0 < split_time]
        decode = [s for s in valid if s['timestamp'] - t0 >= split_time]
        result['gpu_prefill'] = {
            'samples': len(prefill),
            'util_avg': round(sum(s['gpu_util_pct'] for s in prefill) / len(prefill), 1) if prefill else 0,
            'hbm_bw_avg_gb': round(sum(s['hbm_bw_mbytes'] for s in prefill) / len(prefill) / 1024, 2) if prefill else 0,
        }
        result['gpu_decode'] = {
            'samples': len(decode),
            'util_avg': round(sum(s['gpu_util_pct'] for s in decode) / len(decode), 1) if decode else 0,
            'hbm_bw_avg_gb': round(sum(s['hbm_bw_mbytes'] for s in decode) / len(decode) / 1024, 2) if decode else 0,
        }

    result['raw_samples'] = valid
    return result


def main():
    parser = argparse.ArgumentParser(description='GPU utilization sampling during inference')
    parser.add_argument('--separate-phases', action='store_true',
                        help='Enable Prefill/Decode phase separation and test S1+S4 scenarios')
    args = parser.parse_args()

    llm = LLM(
        model='/mnt/moark-models/PaddleOCR-VL-1.5',
        graph_optimization_config={"use_cudagraph": False},
    )
    sp = SamplingParams(max_tokens=256, temperature=0.0, repetition_penalty=1.05)

    s1_prompt = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "file:///data/images/test_doc.png"}},
            {"type": "text", "text": "Recognize this table"},
        ]},
    ]

    if args.separate_phases:
        # S1 + S4 with phase separation
        s4_prompt = [
            {"role": "user", "content": [{"type": "text", "text": "List the top 5 most populated countries in the world."}]},
        ]
        scenarios = [("S1_single_image", s1_prompt), ("S4_text_only", s4_prompt)]
        all_results = {}

        for name, prompt in scenarios:
            # Warmup
            for _ in range(2):
                _ = llm.chat(prompt, sampling_params=sp)
                paddle.device.synchronize("metax_gpu:0")

            output, e2e, ttft, samples = run_inference_with_sampling(llm, prompt, sp)
            n_tokens = len(output[0].outputs.token_ids)
            result = summarize_samples(samples, e2e, ttft, n_tokens, separate_phases=True)
            all_results[name] = result

            print(f"[{name}] E2E={e2e:.3f}s, TTFT={ttft:.3f}s, Tokens={n_tokens}, "
                  f"Throughput={n_tokens/e2e:.2f} tok/s", flush=True)

        out_path = f'{OUTPUT_DIR}/gpu_utilization_phases.json'
        with open(out_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"[Done] Saved to {out_path}", flush=True)

    else:
        # S1 only, overall metrics
        print("[Warmup] 3 iterations...", flush=True)
        for i in range(3):
            _ = llm.chat(s1_prompt, sampling_params=sp)
            paddle.device.synchronize("metax_gpu:0")
            print(f"[Warmup] Iter {i+1}/3 done", flush=True)

        output, e2e, ttft, samples = run_inference_with_sampling(llm, s1_prompt, sp)
        n_tokens = len(output[0].outputs.token_ids)
        throughput = n_tokens / e2e if e2e > 0 else 0
        result = summarize_samples(samples, e2e, ttft, n_tokens, separate_phases=False)

        print(f"[Result] E2E={e2e:.3f}s, TTFT={ttft:.3f}s, Tokens={n_tokens}, "
              f"Throughput={throughput:.2f} tok/s", flush=True)

        out_path = f'{OUTPUT_DIR}/gpu_utilization_detailed.json'
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"[Done] GPU data saved to {out_path}", flush=True)


if __name__ == '__main__':
    main()
