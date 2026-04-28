import os
import atexit

# Env vars must be set before importing paddle — run: source setup_env.sh
import time
import paddle
from fastdeploy import LLM, SamplingParams

paddle.device.set_device('metax_gpu:0')

llm = LLM(
    model='/mnt/moark-models/PaddleOCR-VL-1.5',
    graph_optimization_config={"use_cudagraph": False},
)

# Construct multimodal input — use OpenAI-style messages for chat template
prompt = [
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "file:///data/images/test_doc.png"}},
            {"type": "text", "text": "Recognize this table"},
        ],
    }
]

sampling_params = SamplingParams(max_tokens=256, temperature=0.0, repetition_penalty=1.05)

# ---- Warmup ----
WARMUP_ITERS = 3  # 1st: JIT compile + malloc, 2nd: cache hit, 3rd: steady state
print(f"[Warmup] Running {WARMUP_ITERS} warmup iterations...", flush=True)
for i in range(WARMUP_ITERS):
    warmup_start = time.perf_counter()
    _ = llm.chat(prompt, sampling_params=sampling_params)
    paddle.device.synchronize("metax_gpu:0")
    print(f"[Warmup] Iter {i+1}/{WARMUP_ITERS} done in {time.perf_counter() - warmup_start:.1f}s", flush=True)

# ---- Measured run ----
paddle.device.synchronize("metax_gpu:0")
start = time.perf_counter()
output = llm.chat(prompt, sampling_params=sampling_params)
paddle.device.synchronize("metax_gpu:0")
end = time.perf_counter()

# Extract metrics
e2e_latency = end - start
generated_text = output[0].outputs.text
num_tokens = len(output[0].outputs.token_ids)
tokens_per_sec = num_tokens / e2e_latency if e2e_latency > 0 else 0

# TTFT: check output metrics if available
ttft = None
if output[0].metrics is not None:
    m = output[0].metrics
    if hasattr(m, 'first_token_time'):
        ttft = m.first_token_time

# ---- Report ----
print("=" * 50, flush=True)
print("PERFORMANCE METRICS", flush=True)
print("=" * 50, flush=True)
print(f"End-to-end latency:  {e2e_latency:.3f} s", flush=True)
if ttft is not None:
    print(f"TTFT (first token):  {ttft:.3f} s", flush=True)
else:
    print(f"TTFT (first token):  N/A (not available in output metrics)", flush=True)
print(f"Throughput:          {tokens_per_sec:.2f} tokens/s", flush=True)
print(f"Generated tokens:    {num_tokens}", flush=True)
print("=" * 50, flush=True)
print(f"Output preview: {generated_text[:200]}", flush=True)
