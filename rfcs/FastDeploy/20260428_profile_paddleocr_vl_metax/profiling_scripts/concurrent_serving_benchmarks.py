"""Concurrent serving benchmarks: S5 (2-way), S6 (4-way), S7 (8-way).

Per plan 2.1.2, these scenarios test scheduler continuous batching behavior
by sending concurrent HTTP requests to FastDeploy OpenAI serving server.

Usage:
  1. Start serving in a separate terminal:
     python -m fastdeploy.entrypoints.openai.api_server \
       --model /mnt/moark-models/PaddleOCR-VL-1.5 \
       --graph-optimization-config '{"use_cudagraph": false}' \
       --max-model-len 16384 \
       --gpu-memory-utilization 0.7 \
       --port 8000

  2. Run this script:
     python profiling_scripts/concurrent_serving_benchmarks.py
"""
import os
import time
import json
import asyncio
import aiohttp

OUTPUT_DIR = '/data/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_URL = os.environ.get('FD_SERVING_URL', 'http://127.0.0.1:8000')
IMAGES_DIR = '/data/images'

# Pick distinct images for concurrent requests
CONCURRENT_IMAGES = [
    'test_doc.png',
    'test_receipt.png',
    'test_table.png',
    'test_en_doc.jpg',
    'test_form.png',
    'test_invoice.png',
    'test_pubtab.jpg',
    'test_handwritten.png',
]

PROMPT_TEXT = "Extract all text from this image"


def build_request(image_filename: str) -> dict:
    """Build an OpenAI-compatible chat completion request with one image."""
    image_url = f"file://{IMAGES_DIR}/{image_filename}"
    return {
        "model": "PaddleOCR-VL-1.5",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": PROMPT_TEXT},
                ],
            }
        ],
        "max_tokens": 256,
        "temperature": 0.0,
    }


async def send_request(
    session: aiohttp.ClientSession,
    request_body: dict,
    request_id: int,
) -> dict:
    """Send a single request and return timing + response info."""
    url = f"{BASE_URL}/v1/chat/completions"
    start = time.perf_counter()
    try:
        async with session.post(url, json=request_body) as resp:
            body = await resp.json()
            end = time.perf_counter()
        if resp.status != 200:
            return {
                "request_id": request_id,
                "status": "error",
                "http_status": resp.status,
                "error": body.get("error", {}).get("message", str(body))[:200],
                "e2e_latency_s": round(end - start, 3),
            }
        usage = body.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        return {
            "request_id": request_id,
            "status": "ok",
            "http_status": resp.status,
            "e2e_latency_s": round(end - start, 3),
            "completion_tokens": completion_tokens,
            "throughput_tok_s": round(completion_tokens / (end - start), 2) if (end - start) > 0 else 0,
        }
    except Exception as e:
        end = time.perf_counter()
        return {
            "request_id": request_id,
            "status": "error",
            "error": str(e)[:200],
            "e2e_latency_s": round(end - start, 3),
        }


async def run_concurrency_test(concurrency: int, warmup: int = 1, repeats: int = 3) -> dict:
    """Run concurrent request test at the given concurrency level."""
    images = CONCURRENT_IMAGES[:concurrency]
    all_repeat_results = []

    for r in range(repeats):
        # Warmup on first repeat only
        if r == 0 and warmup > 0:
            print(f"  [Warmup] concurrency={concurrency}, sending {warmup * concurrency} warmup requests...", flush=True)
            async with aiohttp.ClientSession() as session:
                warmup_tasks = []
                for w in range(warmup):
                    for i, img in enumerate(images):
                        warmup_tasks.append(send_request(session, build_request(img), i))
                await asyncio.gather(*warmup_tasks)

        # Actual test
        print(f"  [Repeat {r+1}/{repeats}] concurrency={concurrency}, sending {concurrency} concurrent requests...", flush=True)
        async with aiohttp.ClientSession() as session:
            tasks = [
                send_request(session, build_request(images[i]), i)
                for i in range(concurrency)
            ]
            wall_start = time.perf_counter()
            results = await asyncio.gather(*tasks)
            wall_end = time.perf_counter()

        all_repeat_results.append({
            "repeat": r + 1,
            "wall_time_s": round(wall_end - wall_start, 3),
            "requests": results,
        })

    # Aggregate
    ok_results = [r for rep in all_repeat_results for r in rep["requests"] if r["status"] == "ok"]
    err_results = [r for rep in all_repeat_results for r in rep["requests"] if r["status"] == "error"]

    if ok_results:
        avg_e2e = sum(r["e2e_latency_s"] for r in ok_results) / len(ok_results)
        avg_tokens = sum(r.get("completion_tokens", 0) for r in ok_results) / len(ok_results)
        avg_throughput = sum(r.get("throughput_tok_s", 0) for r in ok_results) / len(ok_results)
        total_wall = sum(rep["wall_time_s"] for rep in all_repeat_results)
        total_ok_tokens = sum(r.get("completion_tokens", 0) for r in ok_results)
        aggregate_throughput = round(total_ok_tokens / total_wall, 2) if total_wall > 0 else 0
    else:
        avg_e2e = avg_tokens = avg_throughput = aggregate_throughput = 0

    summary = {
        "concurrency": concurrency,
        "repeats": repeats,
        "total_requests": concurrency * repeats,
        "ok_count": len(ok_results),
        "error_count": len(err_results),
        "avg_e2e_latency_s": round(avg_e2e, 3),
        "avg_completion_tokens": round(avg_tokens, 1),
        "avg_per_request_throughput_tok_s": round(avg_throughput, 2),
        "aggregate_throughput_tok_s": aggregate_throughput,
        "details": all_repeat_results,
    }

    if err_results:
        summary["errors"] = err_results

    print(f"  [Result] concurrency={concurrency}: avg_e2e={avg_e2e:.3f}s, "
          f"avg_tokens={avg_tokens:.0f}, aggregate_throughput={aggregate_throughput:.2f} tok/s, "
          f"ok={len(ok_results)}, err={len(err_results)}", flush=True)

    return summary


async def check_server_ready() -> bool:
    """Check if the serving server is up."""
    url = f"{BASE_URL}/v1/models"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                return resp.status == 200
    except Exception:
        return False


async def main():
    print("[Check] Verifying serving server is up...", flush=True)
    if not await check_server_ready():
        print(f"[ERROR] Serving server not reachable at {BASE_URL}", flush=True)
        print("Start it with:", flush=True)
        print("  python -m fastdeploy.entrypoints.openai.api_server "
              "--model /mnt/moark-models/PaddleOCR-VL-1.5 "
              "--graph-optimization-config '{\"use_cudagraph\": false}' "
              "--max-model-len 16384 --gpu-memory-utilization 0.7 --port 8000", flush=True)
        return

    print("[OK] Serving server is up.\n", flush=True)

    all_results = {}

    # S5: 2-way concurrent (必做)
    print("[S5] 2-way concurrent requests...", flush=True)
    all_results['S5_concurrent_2'] = await run_concurrency_test(concurrency=2, warmup=1, repeats=3)

    # S6: 4-way concurrent (推荐)
    print("\n[S6] 4-way concurrent requests...", flush=True)
    all_results['S6_concurrent_4'] = await run_concurrency_test(concurrency=4, warmup=1, repeats=3)

    # S7: 8-way concurrent (可选)
    print("\n[S7] 8-way concurrent requests...", flush=True)
    all_results['S7_concurrent_8'] = await run_concurrency_test(concurrency=8, warmup=1, repeats=3)

    # Save
    output_path = f'{OUTPUT_DIR}/concurrent_serving_benchmarks.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Done] Results saved to {output_path}", flush=True)

    # Print summary table
    print("\n=== Concurrent Benchmark Summary ===", flush=True)
    print(f"{'Scenario':<20} {'Conc':>4} {'Avg E2E(s)':>10} {'Agg Tput(tok/s)':>16} {'OK':>4} {'Err':>4}", flush=True)
    print("-" * 60, flush=True)
    for key in ['S5_concurrent_2', 'S6_concurrent_4', 'S7_concurrent_8']:
        r = all_results[key]
        print(f"{key:<20} {r['concurrency']:>4} {r['avg_e2e_latency_s']:>10.3f} "
              f"{r['aggregate_throughput_tok_s']:>16.2f} {r['ok_count']:>4} {r['error_count']:>4}", flush=True)


if __name__ == '__main__':
    asyncio.run(main())
