"""Scenario benchmarks: S1-S4, S8 per the plan.

Uses FastDeploy LLM.chat() for single-request scenarios.
Saves results to output/scenario_benchmarks.json
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

results = {}

def run_benchmark(name, prompt, sampling_params, warmup=5, repeats=3):
    """Run a benchmark with warmup and multiple repeats."""
    # Warmup
    for _ in range(warmup):
        _ = llm.chat(prompt, sampling_params=sampling_params)
        paddle.device.synchronize("metax_gpu:0")

    latencies = []
    ttfts = []
    tokens_list = []

    for r in range(repeats):
        paddle.device.synchronize("metax_gpu:0")
        start = time.perf_counter()
        output = llm.chat(prompt, sampling_params=sampling_params)
        paddle.device.synchronize("metax_gpu:0")
        end = time.perf_counter()

        e2e = end - start
        n_tokens = len(output[0].outputs.token_ids)
        ttft = None
        if output[0].metrics is not None and hasattr(output[0].metrics, 'first_token_time'):
            ttft = output[0].metrics.first_token_time

        latencies.append(e2e)
        ttfts.append(ttft)
        tokens_list.append(n_tokens)

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
    print(f"[{name}] E2E={avg_e2e:.3f}s, TTFT={avg_ttft:.3f}s, Tokens={avg_tokens:.0f}, Throughput={avg_throughput:.2f} tok/s", flush=True)
    return result

# S1: Single image, single request (baseline)
print("[S1] Single image, single request...", flush=True)
s1_prompt = [
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "file:///data/images/test_doc.png"}},
        {"type": "text", "text": "Recognize this table"},
    ]},
]
results['S1_single_image'] = run_benchmark('S1', s1_prompt, SamplingParams(max_tokens=256, temperature=0.0, repetition_penalty=1.05))

# S2: Single image, multi-turn conversation (tests KV Cache accumulation and stability)
# Builds a growing conversation history across turns so the model can reuse
# previously computed KV Cache for the shared prefix (when prefix_caching is
# enabled).  Even with prefix_caching=False (current default for multimodal),
# this still measures how latency scales as conversation length grows and how
# KV Cache is allocated/freed across turns.
print("[S2] Multi-turn conversation, 5 rounds...", flush=True)
s2_conversation = [
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "file:///data/images/test_doc.png"}},
        {"type": "text", "text": "Recognize this table"},
    ]},
]
s2_followups = [
    "How many rows does the table have?",
    "What is the value in the second column of the first row?",
    "Are there any merged cells in the table?",
    "Summarize the table in one sentence.",
]
s2_turn_metrics = []
for turn in range(5):
    paddle.device.synchronize("metax_gpu:0")
    start = time.perf_counter()
    output = llm.chat(s2_conversation, sampling_params=SamplingParams(max_tokens=256, temperature=0.0, repetition_penalty=1.05))
    paddle.device.synchronize("metax_gpu:0")
    e2e = time.perf_counter() - start

    n_tokens = len(output[0].outputs.token_ids)
    ttft = None
    if output[0].metrics is not None and hasattr(output[0].metrics, 'first_token_time'):
        ttft = output[0].metrics.first_token_time

    s2_turn_metrics.append({
        'e2e_s': round(e2e, 3),
        'ttft_s': round(ttft, 3) if ttft else None,
        'tokens': n_tokens,
        'throughput_tok_s': round(n_tokens / e2e, 2) if e2e > 0 else 0,
    })
    print(f"  [S2 Turn {turn+1}] E2E={e2e:.3f}s, TTFT={ttft:.3f}s, Tokens={n_tokens}", flush=True)

    # Append assistant response + next user question to grow the conversation
    assistant_text = output[0].outputs.text if output[0].outputs.text else ""
    s2_conversation.append({"role": "assistant", "content": [{"type": "text", "text": assistant_text}]})
    if turn < len(s2_followups):
        s2_conversation.append({"role": "user", "content": [{"type": "text", "text": s2_followups[turn]}]})

s2_e2es = [m['e2e_s'] for m in s2_turn_metrics]
results['S2_multi_turn'] = {
    'turn_metrics': s2_turn_metrics,
    'avg_e2e_s': round(sum(s2_e2es) / len(s2_e2es), 3),
    'std_e2e_s': round((sum((l - sum(s2_e2es)/len(s2_e2es))**2 for l in s2_e2es)/len(s2_e2es))**0.5, 3),
    'note': 'True multi-turn: conversation history grows each turn. Tests KV Cache accumulation and prefix reuse.',
}
print(f"[S2] Avg={results['S2_multi_turn']['avg_e2e_s']:.3f}s, Std={results['S2_multi_turn']['std_e2e_s']:.3f}s", flush=True)

# S3: Different images
print("[S3] Different images...", flush=True)
for img in ['test_doc.png', 'test_receipt.png', 'test_table.png']:
    img_path = f'/data/images/{img}'
    if not os.path.exists(img_path):
        continue
    prompt = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"file://{img_path}"}},
            {"type": "text", "text": "Extract all text from this image"},
        ]},
    ]
    results[f'S3_{img}'] = run_benchmark(f'S3_{img}', prompt, SamplingParams(max_tokens=256, temperature=0.0, repetition_penalty=1.05))

# S4: Text-only
print("[S4] Text-only...", flush=True)
s4_prompt = [
    {"role": "user", "content": [{"type": "text", "text": "List the top 5 most populated countries in the world with their populations."}]},
]
results['S4_text_only'] = run_benchmark('S4', s4_prompt, SamplingParams(max_tokens=256, temperature=0.0, repetition_penalty=1.05))

# S5: Concurrent requests (2-way)
# S6: Concurrent requests (4-way)
# S7: Concurrent requests (8-way)
# Per plan 2.1.2: test scheduler continuous batching behavior via serving + concurrent HTTP.
# Since LLM.chat() is single-process and cannot test true concurrency,
# these scenarios require a separate serving-based script (concurrent_serving_benchmarks.py).
results['S5_S6_S7_note'] = 'Concurrent scenarios require FastDeploy serving mode. See concurrent_serving_benchmarks.py.'

# S8: Different output lengths
print("[S8] Different output lengths...", flush=True)
for max_tok in [64, 128, 256, 512]:
    results[f'S8_max_tokens_{max_tok}'] = run_benchmark(
        f'S8_{max_tok}',
        s1_prompt,
        SamplingParams(max_tokens=max_tok, temperature=0.0, repetition_penalty=1.05),
    )

# # FD_ENC_DEC_BLOCK_NUM sweep
# print("[BLOCK_NUM] FD_ENC_DEC_BLOCK_NUM sweep...", flush=True)
# # Note: This requires restarting the engine, which we can't do in-process.
# # We'll record the current value and note this needs separate runs.
# results['enc_dec_block_num_note'] = 'Current FD_ENC_DEC_BLOCK_NUM=2. Sweep requires separate process runs with different env vars.'

# Save
with open(f'{OUTPUT_DIR}/scenario_benchmarks.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n[Done] All scenarios saved to {OUTPUT_DIR}/scenario_benchmarks.json", flush=True)
