"""Method A: Paddle Profiler for PaddleOCR-VL-1.5 on Metax GPU.

Captures CPU-side operator stats and timing metrics via paddle.profiler.
GPU kernel data is NOT available because FastDeploy runs inference in a
separate worker process — Paddle Profiler only hooks the current process.
Use mcTracer (method_mctracer_inference.py) for GPU kernel-level profiling.
"""
import os
import time
import json

import paddle
from fastdeploy import LLM, SamplingParams

paddle.device.set_device('metax_gpu:0')

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

llm = LLM(
    model='/mnt/moark-models/PaddleOCR-VL-1.5',
    graph_optimization_config={"use_cudagraph": False},
)

prompt = [
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"file://{os.path.dirname(os.path.abspath(__file__))}/../images/test_doc.png"}},
            {"type": "text", "text": "Recognize this table"},
        ],
    }
]
sampling_params = SamplingParams(max_tokens=256, temperature=0.0, repetition_penalty=1.05)

# Warmup
print("[Warmup] 3 iterations...", flush=True)
for i in range(3):
    _ = llm.chat(prompt, sampling_params=sampling_params)
    paddle.device.synchronize("metax_gpu:0")
    print(f"[Warmup] Iter {i+1}/3 done", flush=True)

# Profiled run — CPU targets only (CUSTOM_DEVICE is useless in multi-process mode)
from paddle.profiler import Profiler, ProfilerTarget, SortedKeys

profiler = Profiler(
    targets=[ProfilerTarget.CPU],
    record_shapes=True,
    profile_memory=True,
    on_trace_ready=lambda p: None,  # suppress default ./profiler_log/ output; we export manually below
)

profiler.start()
print("[Profile] Starting profiled inference...", flush=True)

paddle.device.synchronize("metax_gpu:0")
start = time.perf_counter()
output = llm.chat(prompt, sampling_params=sampling_params)
paddle.device.synchronize("metax_gpu:0")
end = time.perf_counter()

profiler.stop()

e2e = end - start
n_tokens = len(output[0].outputs.token_ids)
throughput = n_tokens / e2e if e2e > 0 else 0

ttft = None
if output[0].metrics is not None and hasattr(output[0].metrics, 'first_token_time'):
    ttft = output[0].metrics.first_token_time

print(f"[Result] E2E={e2e:.3f}s, TTFT={ttft:.3f}s, Tokens={n_tokens}, Throughput={throughput:.2f} tok/s", flush=True)

# Print profiler summary (CPU operators + memory)
print("\n[Profiler Summary]", flush=True)
profiler.summary(sorted_by=SortedKeys.CPUTotal)

# Export trace and parse CPU operator stats
trace_path = f'{OUTPUT_DIR}/hook_profiler_trace.json'
profiler.export(path=trace_path, format='json')

op_stats = {}
if os.path.exists(trace_path):
    print(f"[Done] Trace exported to {trace_path}", flush=True)
    try:
        with open(trace_path, 'r') as f:
            trace_data = json.load(f)

        for ev in trace_data.get('traceEvents', []):
            if ev.get('ph') != 'X' or ev.get('cat') != 'Operator':
                continue
            name = ev.get('name', '')
            dur_us = ev.get('dur', 0)
            clean_name = name.split('[')[0].strip() if '[' in name else name
            if clean_name not in op_stats:
                op_stats[clean_name] = {'cpu_total_ms': 0, 'calls': 0}
            op_stats[clean_name]['calls'] += 1
            op_stats[clean_name]['cpu_total_ms'] += dur_us / 1000

    except Exception as e:
        print(f"[Warning] Trace parsing failed: {e}", flush=True)
else:
    print("[Warning] Trace export failed", flush=True)

# Sort, save, and print results
op_stats_sorted = dict(sorted(op_stats.items(), key=lambda x: x[1]['cpu_total_ms'], reverse=True))

result_data = {
    'operators': op_stats_sorted,
    'meta': {
        'e2e_s': e2e,
        'ttft_s': ttft,
        'tokens': n_tokens,
        'throughput_tok_s': throughput,
    },
    'note': 'GPU kernel data not available — FastDeploy runs inference in a separate worker process. Use mcTracer for kernel-level profiling.',
}

with open(f'{OUTPUT_DIR}/profiler_op_stats.json', 'w') as f:
    json.dump(result_data, f, indent=2)
print(f"\n[Done] {len(op_stats_sorted)} operators saved to {OUTPUT_DIR}/profiler_op_stats.json", flush=True)

if op_stats_sorted:
    print("\n[Top-10 Operators by CPU time]", flush=True)
    for i, (name, stats) in enumerate(list(op_stats_sorted.items())[:10]):
        print(f"  {i+1}. {name}: cpu={stats['cpu_total_ms']:.1f}ms, calls={stats['calls']}", flush=True)
