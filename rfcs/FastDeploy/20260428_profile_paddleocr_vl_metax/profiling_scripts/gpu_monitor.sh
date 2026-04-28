#!/bin/bash
# GPU utilization long-running background monitor (no Python/Paddle dependency).
#
# Purpose: Low-overhead, dependency-free GPU monitoring for long-running sessions
# (e.g. serving benchmarks, extended profiling). Outputs CSV at 1s intervals.
# Complements the Python gpu_util_sampling.py which provides higher-frequency
# sampling with JSON output and Prefill/Decode phase separation.
#
# Usage:
#   bash profiling_scripts/gpu_monitor.sh
#   # Runs in foreground; Ctrl+C or 'kill <PID>' to stop
#
# Output: /data/output/gpu_utilization.log (CSV format)

OUTPUT=/data/output/gpu_utilization.log
mkdir -p /data/output

echo "timestamp,gpu_util_pct,mem_used_mb,mem_total_mb,hbm_bw_mbytes,ccx_avg_pct" > "$OUTPUT"
echo "[Monitor] GPU monitor started. Logging to $OUTPUT"
echo "[Monitor] PID: $$"
echo "[Monitor] Run 'kill $$' to stop monitoring."

while true; do
    # GPU utilization: "    GPU : 0 %"
    GPU_UTIL=$(mx-smi --show-usage 2>/dev/null | grep -E "^\s+GPU\s+:" | head -1 | awk -F: '{print $2}' | awk '{print $1}')
    GPU_UTIL=${GPU_UTIL:-0}

    # Memory: "    vis_vram used : 846264 KB"
    MEM_USED_KB=$(mx-smi --show-memory 2>/dev/null | grep "vis_vram used" | awk -F: '{print $2}' | awk '{print $1}')
    MEM_TOTAL_KB=$(mx-smi --show-memory 2>/dev/null | grep "vis_vram total" | awk -F: '{print $2}' | awk '{print $1}')
    MEM_USED_MB=$(( ${MEM_USED_KB:-0} / 1024 ))
    MEM_TOTAL_MB=$(( ${MEM_TOTAL_KB:-0} / 1024 ))

    # HBM bandwidth: "    throughput : 1 MBytes/s"
    HBM_BW=$(mx-smi --show-hbm-bandwidth 2>/dev/null | grep "throughput" | awk -F: '{print $2}' | awk '{print $1}')
    HBM_BW=${HBM_BW:-0}

    # CCX core utilization: "    CCX  95 %  95 %  96 %" → average
    CCX_AVG=$(mx-smi --show-core-usage 2>/dev/null | grep "CCX" | sed 's/CCX//' | tr -s ' ' | awk '{s=0;c=0; for(i=1;i<=NF;i+=2){s+=$i;c++}} END{if(c>0) printf "%d",s/c; else print 0}')
    CCX_AVG=${CCX_AVG:-0}

    TIMESTAMP=$(date +%s)
    echo "${TIMESTAMP},${GPU_UTIL},${MEM_USED_MB},${MEM_TOTAL_MB},${HBM_BW},${CCX_AVG}" >> "$OUTPUT"
    sleep 1
done
