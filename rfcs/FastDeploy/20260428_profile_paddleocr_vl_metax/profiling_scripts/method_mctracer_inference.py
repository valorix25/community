"""Method B: mcTracer GPU kernel trace for PaddleOCR-VL-1.5 on Metax GPU.

Run under mcTracer:
  /opt/maca/bin/mcTracer --mctx --name paddleocr_vl_profile python profiling_scripts/method_b_mctracer_inference.py

mcTracer intercepts all GPU kernel launches and produces .mctx trace files
with real GPU execution times. This script only needs to run inference —
mcTracer handles the tracing automatically.

Scenarios: S1 (single image) + S4 (text-only) per plan 2.3.2.
"""
import os
import time
import json

# Env vars must be set before importing paddle — run: source setup_env.sh

import paddle
from fastdeploy import LLM, SamplingParams

paddle.device.set_device('metax_gpu:0')

OUTPUT_DIR = '/data/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

llm = LLM(
    model='/mnt/moark-models/PaddleOCR-VL-1.5',
    graph_optimization_config={"use_cudagraph": False},
)

sampling_params = SamplingParams(max_tokens=256, temperature=0.0, repetition_penalty=1.05)

# === S1: Single image request ===
s1_prompt = [
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "file:///data/images/test_doc.png"}},
            {"type": "text", "text": "Recognize this table"},
        ],
    }
]

print("[S1 Warmup] Single image...", flush=True)
for _ in range(2):
    _ = llm.chat(s1_prompt, sampling_params=sampling_params)
    paddle.device.synchronize("metax_gpu:0")

print("[S1 Profiled] Single image under mcTracer...", flush=True)
paddle.device.synchronize("metax_gpu:0")
s1_start = time.perf_counter()
s1_output = llm.chat(s1_prompt, sampling_params=sampling_params)
paddle.device.synchronize("metax_gpu:0")
s1_end = time.perf_counter()

s1_e2e = s1_end - s1_start
s1_tokens = len(s1_output[0].outputs.token_ids)
s1_ttft = None
if s1_output[0].metrics is not None and hasattr(s1_output[0].metrics, 'first_token_time'):
    s1_ttft = s1_output[0].metrics.first_token_time
print(f"[S1] E2E={s1_e2e:.3f}s, TTFT={s1_ttft:.3f}s, Tokens={s1_tokens}, Throughput={s1_tokens/s1_e2e:.2f} tok/s", flush=True)

# === S4: Text-only request ===
s4_prompt = [
    {"role": "user", "content": [{"type": "text", "text": "List the top 5 most populated countries in the world."}]},
]

print("[S4 Warmup] Text-only...", flush=True)
for _ in range(2):
    _ = llm.chat(s4_prompt, sampling_params=sampling_params)
    paddle.device.synchronize("metax_gpu:0")

print("[S4 Profiled] Text-only under mcTracer...", flush=True)
paddle.device.synchronize("metax_gpu:0")
s4_start = time.perf_counter()
s4_output = llm.chat(s4_prompt, sampling_params=sampling_params)
paddle.device.synchronize("metax_gpu:0")
s4_end = time.perf_counter()

s4_e2e = s4_end - s4_start
s4_tokens = len(s4_output[0].outputs.token_ids)
s4_ttft = None
if s4_output[0].metrics is not None and hasattr(s4_output[0].metrics, 'first_token_time'):
    s4_ttft = s4_output[0].metrics.first_token_time
print(f"[S4] E2E={s4_e2e:.3f}s, TTFT={s4_ttft:.3f}s, Tokens={s4_tokens}, Throughput={s4_tokens/s4_e2e:.2f} tok/s", flush=True)

# Save timing results
results = {
    "S1_single_image": {
        "e2e_latency_s": round(s1_e2e, 3),
        "ttft_s": round(s1_ttft, 3) if s1_ttft else None,
        "tokens": s1_tokens,
        "throughput_tok_s": round(s1_tokens / s1_e2e, 2),
    },
    "S4_text_only": {
        "e2e_latency_s": round(s4_e2e, 3),
        "ttft_s": round(s4_ttft, 3) if s4_ttft else None,
        "tokens": s4_tokens,
        "throughput_tok_s": round(s4_tokens / s4_e2e, 2),
    },
}

with open(f'{OUTPUT_DIR}/mctracer_timing.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"[Done] Timing saved to {OUTPUT_DIR}/mctracer_timing.json", flush=True)
print("[Done] mcTracer .mctx trace files will be in the working directory after process exits", flush=True)
