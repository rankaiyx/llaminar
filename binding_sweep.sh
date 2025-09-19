#!/bin/bash

# Comprehensive MPI Binding and Process Count Sweep
# Tests various process counts and binding scenarios for optimal GFLOPS

echo "=============================================="
echo "  COMPREHENSIVE MPI BINDING SWEEP"
echo "=============================================="
echo "Testing process counts and binding scenarios"
echo "Matrix sizes: up to 4096^3 only"
echo "=============================================="

# Create results directory
mkdir -p sweep_results
RESULTS_FILE="sweep_results/binding_sweep_$(date +%Y%m%d_%H%M%S).txt"

echo "Results will be saved to: $RESULTS_FILE"
echo ""

# Function to run test and capture results
run_test() {
    local np=$1
    local bind_to=$2
    local map_by=$3
    local ht_flag=$4
    local description="$5"
    
    echo "=============================================="
    echo "Testing: $description"
    echo "Process count: $np"
    echo "Binding: --bind-to $bind_to --map-by $map_by"
    echo "Hyperthreading: $ht_flag"
    echo "=============================================="
    
    # Log to file
    {
        echo "=============================================="
        echo "Test: $description"
        echo "Command: mpirun -np $np --bind-to $bind_to --map-by $map_by --report-bindings $ht_flag"
        echo "Date: $(date)"
        echo "=============================================="
    } >> "$RESULTS_FILE"
    
    # Run the actual test
    if [[ -n "$ht_flag" ]]; then
        timeout 300 mpirun -np "$np" --bind-to "$bind_to" --map-by "$map_by" --report-bindings \
                           ./build/Llaminar --test-mode sweep "$ht_flag" 2>&1 | tee -a "$RESULTS_FILE"
    else
        timeout 300 mpirun -np "$np" --bind-to "$bind_to" --map-by "$map_by" --report-bindings \
                           ./build/Llaminar --test-mode sweep 2>&1 | tee -a "$RESULTS_FILE"
    fi
    
    echo "" | tee -a "$RESULTS_FILE"
    echo "" | tee -a "$RESULTS_FILE"
}

# Test 1: Optimal 2-process socket binding (baseline)
run_test 2 "socket" "ppr:1:socket" "" "2 Processes - Socket Binding (Optimal Baseline)"

# Test 2: 2-process core binding (problematic default)
run_test 2 "core" "ppr:1:socket" "" "2 Processes - Core Binding (Problematic Default)"

# Test 3: 4-process socket binding
run_test 4 "socket" "ppr:2:socket" "" "4 Processes - Socket Binding"

# Test 4: 4-process core binding
run_test 4 "core" "ppr:2:socket" "" "4 Processes - Core Binding"

# Test 5: 8-process socket binding
run_test 8 "socket" "ppr:4:socket" "" "8 Processes - Socket Binding"

# Test 6: 14-process socket binding (half cores per socket)
run_test 14 "socket" "ppr:7:socket" "" "14 Processes - Socket Binding (Half Cores)"

# Test 7: 28-process socket binding (one per physical core)
run_test 28 "socket" "ppr:14:socket" "" "28 Processes - Socket Binding (One Per Physical Core)"

# Test 8: 56-process socket binding (one per core including HT)
run_test 56 "socket" "ppr:28:socket" "" "56 Processes - Socket Binding (All Physical Cores)"

echo "=============================================="
echo "HYPERTHREADING TESTS"
echo "=============================================="

# Test 9: 2-process with hyperthreading
run_test 2 "socket" "ppr:1:socket" "--enable-hyperthreading" "2 Processes - Socket Binding + Hyperthreading"

# Test 10: 4-process with hyperthreading
run_test 4 "socket" "ppr:2:socket" "--enable-hyperthreading" "4 Processes - Socket Binding + Hyperthreading"

# Test 11: 56-process with hyperthreading (one per physical core)
run_test 56 "socket" "ppr:28:socket" "--enable-hyperthreading" "56 Processes - Socket Binding + Hyperthreading"

# Test 12: 112-process with hyperthreading (one per logical core)
run_test 112 "socket" "ppr:56:socket" "--enable-hyperthreading" "112 Processes - Socket Binding + All Hyperthreads"

echo "=============================================="
echo "SWEEP COMPLETE"
echo "=============================================="
echo "Results saved to: $RESULTS_FILE"
echo ""

# Extract and summarize best performing configurations
echo "Analyzing results..."
{
    echo "=============================================="
    echo "PERFORMANCE SUMMARY"
    echo "=============================================="
    
    # Extract GFLOPS results from the log
    echo "Best GFLOPS results by configuration:"
    grep -A 10 "QUICK SWEEP SUMMARY" "$RESULTS_FILE" | grep -E "(Test:|Best GFLOPS:|Process count:)" | 
    while read -r line1 && read -r line2 && read -r line3; do
        test_name=$(echo "$line1" | sed 's/Test: //')
        process_count=$(echo "$line2" | sed 's/Process count: //')
        gflops=$(echo "$line3" | sed 's/Best GFLOPS: //')
        printf "%-50s | %3s processes | %10s GFLOPS\n" "$test_name" "$process_count" "$gflops"
    done | sort -k5 -nr
    
} >> "$RESULTS_FILE"

echo "Analysis complete! Check $RESULTS_FILE for detailed results and summary."