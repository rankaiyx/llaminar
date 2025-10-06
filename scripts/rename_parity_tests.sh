#!/bin/bash
# Automated test file renaming script
# Renames misnamed "parity" tests to "correctness" tests
# Preserves true ParityFramework tests

set -e

echo "============================================"
echo "Test Suite Renaming: parity → correctness"
echo "============================================"
echo ""

# Files to rename (excludes test_parity_framework.cpp and test_abstract_pipeline_parity.cpp)
FILES=(
    "test_attention_shard_parity"
    "test_attention_tp_parity"
    "test_attention_tp_sim_parity"
    "test_end2end_shard_parity"
    "test_incremental_decode_parity"
    "test_kv_cache_growth_parity"
    "test_linear_orientation_parity"
    "test_mlp_shard_parity"
    "test_mlp_tp_parity"
    "test_mpi_softmax_parity"
    "test_quant_dequant_parity"
    "test_rmsnorm_core_parity"
    "test_rope_recurrence_parity"
    "test_softmax_core_parity"
    "test_tp_gemm_parity"
    "test_weight_slice_parity"
)

echo "Step 1: Renaming test files..."
echo "-------------------------------"
for file in "${FILES[@]}"; do
    old="tests/${file}.cpp"
    new="tests/${file/parity/correctness}.cpp"
    
    if [ -f "$old" ]; then
        echo "✓ $old → $new"
        git mv "$old" "$new"
    else
        echo "⚠ Warning: $old not found, skipping"
    fi
done

echo ""
echo "Step 2: Updating test suite names in files..."
echo "----------------------------------------------"
for file in "${FILES[@]}"; do
    new_file="tests/${file/parity/correctness}.cpp"
    
    if [ -f "$new_file" ]; then
        # Extract the base name (e.g., AttentionShardParity)
        # Convert test_attention_shard_parity → AttentionShardParity
        base_old=$(echo "$file" | sed 's/test_//' | sed 's/_parity$//' | sed -r 's/(^|_)([a-z])/\U\2/g')
        base_new="${base_old/Parity/Correctness}"
        
        # Update TEST() and TEST_F() declarations
        sed -i "s/${base_old}Parity/${base_new}/g" "$new_file"
        sed -i "s/TEST(${base_old}/TEST(${base_new}/g" "$new_file"
        sed -i "s/TEST_F(${base_old}/TEST_F(${base_new}/g" "$new_file"
        
        echo "✓ Updated test names in $new_file"
    fi
done

echo ""
echo "Step 3: Summary"
echo "---------------"
echo "✓ Renamed ${#FILES[@]} test files"
echo "✓ Updated test suite names"
echo ""
echo "⚠ Manual action required:"
echo "  1. Update CMakeLists.txt to use new filenames"
echo "  2. Review changes with: git status"
echo "  3. Commit changes: git commit -m 'Rename parity tests to correctness tests'"
echo ""
echo "Files preserved (true ParityFramework tests):"
echo "  - test_parity_framework.cpp"
echo "  - test_abstract_pipeline_parity.cpp"
echo ""
