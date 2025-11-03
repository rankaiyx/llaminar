#!/bin/bash
# Check status of full validation benchmark

completed=$(grep -c "^\[       OK \]" /tmp/full_validation.log 2>/dev/null || echo 0)
current=$(grep "^\[ RUN      \]" /tmp/full_validation.log | tail -1 | awk '{print $4}' | sed 's/CudaGemmHeuristicValidation\.//')
remaining=$((27 - completed))
est_minutes=$((remaining * 2))

echo "=========================================="
echo "  Full Validation Status"
echo "=========================================="
printf "  Tests: %2d / 27 completed\n" "$completed"
echo "  Current: $current"
printf "  ETA: ~%2d minutes\n" "$est_minutes"
echo "=========================================="
echo ""

if ps aux | grep -v grep | grep -q "v2_perf_cuda_heuristic_validation"; then
    echo "Status: [RUNNING] Benchmark in progress"
else
    echo "Status: [COMPLETE] Benchmark finished, checking validation..."
    if [ -f validation_full_results.json ]; then
        echo "Status: [SUCCESS] Validation complete!"
        echo ""
        echo "Top-N Hit Rates:"
        cat validation_full_results.json | python3 -m json.tool 2>/dev/null | grep -A 15 "ranking_metrics" | grep -E "top_|mean" | head -10
    else
        echo "Status: [WAITING] Validation analysis pending..."
    fi
fi
