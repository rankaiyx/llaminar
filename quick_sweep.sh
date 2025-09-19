#!/bin/bash

# Quick MPI Binding Performance Sweep
# Tests key process counts and binding scenarios for optimal GFLOPS

echo "=============================================="
echo "  QUICK MPI BINDING PERFORMANCE SWEEP"
echo "=============================================="
echo "Testing key configurations with socket binding"
echo "=============================================="

# Results storage
RESULTS_FILE="sweep_results/quick_sweep_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p sweep_results

# Function to run test and extract key metrics
run_test() {
    local np=$1
    local bind_to=$2
    local map_by=$3
    local description="$4"
    
    echo ""
    echo "Testing: $description ($np processes, $bind_to binding)"
    echo "=============================================="
    
    # Log to file
    {
        echo "=============================================="
        echo "Test: $description"
        echo "Processes: $np"
        echo "Binding: $bind_to"
        echo "Map: $map_by"
        echo "Date: $(date)"
        echo "=============================================="
    } >> "$RESULTS_FILE"
    
    # Run the test with timeout
    echo "Running: mpirun -np $np --bind-to $bind_to --map-by $map_by build/Llaminar --test-mode benchmark"
    
    timeout 300 mpirun -np "$np" --bind-to "$bind_to" --map-by "$map_by" \
                       build/Llaminar --test-mode benchmark 2>&1 | \
    while IFS= read -r line; do
        echo "$line"
        echo "$line" >> "$RESULTS_FILE"
        
        # Extract GFLOPS from performance lines
        if [[ "$line" =~ Performance:.*([0-9]+\.[0-9]+).*GFLOPS ]]; then
            gflops="${BASH_REMATCH[1]}"
            echo "  >>> GFLOPS: $gflops" | tee -a "$RESULTS_FILE"
        fi
    done
    
    echo "" | tee -a "$RESULTS_FILE"
}

echo "Starting benchmark sweep..."

# Test 1: 2-process socket binding (should be optimal)
run_test 2 "socket" "ppr:1:socket" "2-Process Socket Binding (Optimal)"

# Test 2: 2-process core binding (problematic default)  
run_test 2 "core" "ppr:1:socket" "2-Process Core Binding (Default Problem)"

# Test 3: 4-process socket binding
run_test 4 "socket" "ppr:2:socket" "4-Process Socket Binding"

# Test 4: 8-process socket binding
run_test 8 "socket" "ppr:4:socket" "8-Process Socket Binding"

# Test 5: 14-process socket binding (half cores per socket)
run_test 14 "socket" "ppr:7:socket" "14-Process Socket Binding"

# Test 6: 28-process socket binding (one per physical core)
run_test 28 "socket" "ppr:14:socket" "28-Process Socket Binding"

echo ""
echo "=============================================="
echo "SWEEP COMPLETE!"
echo "=============================================="
echo "Results saved to: $RESULTS_FILE"

# Quick analysis
echo ""
echo "Analyzing peak GFLOPS per configuration..."
{
    echo ""
    echo "=============================================="
    echo "PERFORMANCE SUMMARY"
    echo "=============================================="
    
    # Extract best GFLOPS for each test
    echo "Peak GFLOPS by configuration:"
    
    grep -B 1 ">>> GFLOPS:" "$RESULTS_FILE" | grep -E "(Test:|>>> GFLOPS:)" | \
    while read -r line1 && read -r line2; do
        if [[ "$line1" =~ Test:\ (.+) ]]; then
            test_name="${BASH_REMATCH[1]}"
        fi
        if [[ "$line2" =~ GFLOPS:\ ([0-9]+\.[0-9]+) ]]; then
            gflops="${BASH_REMATCH[1]}"
            printf "%-40s | %10s GFLOPS\n" "$test_name" "$gflops"
        fi
    done | sort -k3 -nr
    
} >> "$RESULTS_FILE"

echo "Analysis complete!"
echo ""
echo "Key findings will be in: $RESULTS_FILE"