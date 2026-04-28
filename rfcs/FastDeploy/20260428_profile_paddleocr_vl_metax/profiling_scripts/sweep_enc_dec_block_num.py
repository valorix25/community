"""Sweep FD_ENC_DEC_BLOCK_NUM across multiple values.

This parameter controls KV cache block pre-allocation for the encoder->decoder
transition. It is read at LLM init time, so each value requires a separate
process. This script runs as a wrapper that spawns child processes.

Usage:
    # Full sweep (spawns child process for each value)
    python sweep_enc_dec_block_num.py

    # Single child run (called by wrapper internally)
    python sweep_enc_dec_block_num.py --child --block-num 2

Output: output/enc_dec_block_num_sweep.json
"""
import os
import sys
import time
import json
import argparse
import subprocess

SWEEP_VALUES = [1, 2, 4, 8]
WARMUP = 2
REPEATS = 3
MODEL_PATH = '/mnt/moark-models/PaddleOCR-VL-1.5'
OUTPUT_DIR = 'output'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'enc_dec_block_num_sweep.json')

# Env vars for metax_gpu — keep in sync with setup_env.sh
# (not sourced via shell because this script spawns child processes with varying FD_ENC_DEC_BLOCK_NUM)
BASE_ENV = {
    'MACA_VISIBLE_DEVICES': '0',
    'FD_SAMPLING_CLASS': 'rejection',
    'ENABLE_V1_KVCACHE_SCHEDULER': '1',
    'FD_METAX_KVCACHE_MEM': '8',
    'FLAGS_weight_only_linear_arch': '80',
    'PADDLE_XCCL_BACKEND': 'metax_gpu',
    'FD_MOE_BACKEND': 'cutlass',
}


def run_child(block_num):
    """Run a single benchmark as a child process with the given block_num."""
    env = os.environ.copy()
    env.update(BASE_ENV)
    env['FD_ENC_DEC_BLOCK_NUM'] = str(block_num)

    cmd = [sys.executable, __file__, '--child', '--block-num', str(block_num)]
    print(f"[Sweep] Spawning child: FD_ENC_DEC_BLOCK_NUM={block_num}", flush=True)

    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"[Sweep] Child FAILED (block_num={block_num})", flush=True)
        print(f"  stderr: {result.stderr[-500:]}", flush=True)
        return None

    # Parse JSON from last line of stdout (child may print progress lines too)
    lines = result.stdout.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line.startswith('{'):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    print(f"[Sweep] No JSON found in child output (block_num={block_num})", flush=True)
    return None


def child_main(block_num):
    """Child process: init LLM, run benchmark, print JSON to stdout."""
    # These must be set BEFORE importing paddle/fastdeploy
    os.environ['MACA_VISIBLE_DEVICES'] = '0'
    os.environ['FD_ENC_DEC_BLOCK_NUM'] = str(block_num)
    for k, v in BASE_ENV.items():
        os.environ[k] = v

    import paddle
    from fastdeploy import LLM, SamplingParams

    paddle.device.set_device('metax_gpu:0')

    llm = LLM(
        model=MODEL_PATH,
        graph_optimization_config={"use_cudagraph": False},
    )

    prompt = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "file:///data/images/test_doc.png"}},
            {"type": "text", "text": "Recognize this table"},
        ]},
    ]
    sampling_params = SamplingParams(max_tokens=256, temperature=0.0, repetition_penalty=1.05)

    # Warmup
    for i in range(WARMUP):
        _ = llm.chat(prompt, sampling_params=sampling_params)
        paddle.device.synchronize('metax_gpu:0')
        print(f"[Child block_num={block_num}] Warmup {i+1}/{WARMUP} done", flush=True)

    # Measured runs
    latencies = []
    ttfts = []
    tokens_list = []

    for r in range(REPEATS):
        paddle.device.synchronize('metax_gpu:0')
        start = time.perf_counter()
        output = llm.chat(prompt, sampling_params=sampling_params)
        paddle.device.synchronize('metax_gpu:0')
        end = time.perf_counter()

        e2e = end - start
        n_tokens = len(output[0].outputs.token_ids)
        ttft = None
        if output[0].metrics is not None and hasattr(output[0].metrics, 'first_token_time'):
            ttft = output[0].metrics.first_token_time

        latencies.append(e2e)
        ttfts.append(ttft)
        tokens_list.append(n_tokens)
        print(f"[Child block_num={block_num}] Run {r+1}/{REPEATS}: E2E={e2e:.3f}s", flush=True)

    avg_e2e = sum(latencies) / len(latencies)
    avg_ttft = sum(t for t in ttfts if t) / len([t for t in ttfts if t]) if any(ttfts) else None
    avg_tokens = sum(tokens_list) / len(tokens_list)
    avg_throughput = avg_tokens / avg_e2e if avg_e2e > 0 else 0

    result = {
        'e2e_latency_s': round(avg_e2e, 3),
        'ttft_s': round(avg_ttft, 3) if avg_ttft else None,
        'tokens': round(avg_tokens),
        'throughput_tok_s': round(avg_throughput, 2),
        'latencies_s': [round(l, 3) for l in latencies],
        'ttfts_s': [round(t, 3) if t else None for t in ttfts],
    }

    # Print JSON as the LAST line — wrapper parses this
    print(json.dumps(result), flush=True)


def wrapper_main():
    """Wrapper: run child for each sweep value, collect and save results."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = {}
    for val in SWEEP_VALUES:
        result = run_child(val)
        if result is not None:
            all_results[str(val)] = result
        else:
            all_results[str(val)] = {'error': 'child process failed'}

    output = {
        'sweep_config': {
            'values': SWEEP_VALUES,
            'model': MODEL_PATH,
            'warmup': WARMUP,
            'repeat': REPEATS,
            'prompt': 'single image + Recognize this table',
            'max_tokens': 256,
        },
        'results': all_results,
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)

    # Print comparison table
    print(f"\n{'='*70}", flush=True)
    print(f"  FD_ENC_DEC_BLOCK_NUM Sweep Results", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'block_num':>10} {'E2E(s)':>10} {'TTFT(s)':>10} {'tok/s':>10} {'tokens':>10}", flush=True)
    print(f"{'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}", flush=True)
    for val in SWEEP_VALUES:
        r = all_results.get(str(val), {})
        if 'error' in r:
            print(f"{val:>10} {'FAILED':>10}", flush=True)
        else:
            print(f"{val:>10} {r['e2e_latency_s']:>10.3f} {r['ttft_s'] or 'N/A':>10} {r['throughput_tok_s']:>10.2f} {r['tokens']:>10.0f}", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Results saved to {OUTPUT_FILE}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sweep FD_ENC_DEC_BLOCK_NUM')
    parser.add_argument('--child', action='store_true', help='Run as child process')
    parser.add_argument('--block-num', type=int, default=2, help='enc_dec_block_num value for child')
    args = parser.parse_args()

    if args.child:
        child_main(args.block_num)
    else:
        wrapper_main()
