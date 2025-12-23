#!/bin/bash
# Show gcov coverage metrics for Llaminar V2 project
# Reports line and branch coverage for all source files, sorted by coverage %

cd "$(dirname "$0")/.."

echo ""
echo "=============================================================================="
echo "                    V2 Test Coverage Summary (gcov)                           "
echo "=============================================================================="
echo ""

GCDA_COUNT=$(find build_v2/CMakeFiles/llaminar2_core.dir -name '*.gcda' 2>/dev/null | wc -l)

if [ "$GCDA_COUNT" -eq 0 ]; then
    echo "No coverage data found."
    echo ""
    echo "Make sure you:"
    echo "  1. Built with coverage enabled (Debug build with --coverage)"
    echo "  2. Ran tests to generate .gcda files"
    echo ""
    echo "Example:"
    echo "  ctest --test-dir build_v2 -R '^V2_Unit_'"
    exit 1
fi

# Find all .gcno files for llaminar2_core (in CMakeFiles directory)
GCNO_COUNT=$(find build_v2/CMakeFiles/llaminar2_core.dir -name "*.gcno" 2>/dev/null | wc -l)

if [ "$GCNO_COUNT" -eq 0 ]; then
    echo "No .gcno files found. Rebuild with coverage enabled."
    echo ""
    echo "Rebuild with:"
    echo "  cmake -B build_v2 -S src/v2 -DCMAKE_BUILD_TYPE=Debug \\"
    echo "    -DCMAKE_CXX_FLAGS_DEBUG='-g -O0 --coverage -fprofile-abs-path' \\"
    echo "    -DCMAKE_EXE_LINKER_FLAGS_DEBUG='--coverage'"
    echo "  cmake --build build_v2 --parallel"
    exit 1
fi

echo "Analyzing $GCNO_COUNT instrumented source files..."
echo ""

# Create temp file for results
RESULTS_FILE=$(mktemp)
trap "rm -f $RESULTS_FILE" EXIT

TOTAL_LINES=0
COVERED_LINES=0
TOTAL_BRANCHES=0
COVERED_BRANCHES=0
FILE_COUNT=0

# Process all .gcda files (coverage data)
for gcda_file in $(find build_v2/CMakeFiles/llaminar2_core.dir -name "*.gcda" 2>/dev/null); do
    # Get corresponding source file name
    gcno_file="${gcda_file%.gcda}.gcno"
    
    if [ -f "$gcno_file" ]; then
        # Run gcov with branch coverage (-b) on the gcda file
        # Use -n for no output files, redirect to capture
        GCOV_OUTPUT=$(gcov -b -n "$gcda_file" 2>&1)
        
        # Extract the source file name from gcov output
        SOURCE_FILE=$(echo "$GCOV_OUTPUT" | grep "File '" | head -1 | sed "s/File '\\(.*\\)'/\\1/")
        
        # Skip if no source file found or it's a system header
        if [ -z "$SOURCE_FILE" ] || [[ "$SOURCE_FILE" == /usr/* ]] || [[ "$SOURCE_FILE" == *"_deps"* ]]; then
            continue
        fi
        
        # Extract line coverage: "Lines executed:XX.XX% of Y"
        LINE_INFO=$(echo "$GCOV_OUTPUT" | grep "Lines executed:" | head -1)
        LINE_PERCENT=$(echo "$LINE_INFO" | grep -oE '[0-9]+\.[0-9]+%' | head -1)
        LINE_TOTAL=$(echo "$LINE_INFO" | grep -oE 'of [0-9]+' | head -1 | sed 's/of //')
        
        # Extract branch coverage: "Branches executed:XX.XX% of Y"
        BRANCH_INFO=$(echo "$GCOV_OUTPUT" | grep "Branches executed:" | head -1)
        BRANCH_PERCENT=$(echo "$BRANCH_INFO" | grep -oE '[0-9]+\.[0-9]+%' | head -1)
        BRANCH_TOTAL=$(echo "$BRANCH_INFO" | grep -oE 'of [0-9]+' | head -1 | sed 's/of //')
        
        if [ -n "$LINE_PERCENT" ] && [ -n "$LINE_TOTAL" ]; then
            # Get just the filename
            BASENAME=$(basename "$SOURCE_FILE")
            
            # Calculate covered lines from percentage
            LINE_PCT_NUM=${LINE_PERCENT%\%}
            LINE_COVERED=$(echo "$LINE_PCT_NUM $LINE_TOTAL" | awk '{printf "%.0f", $1 * $2 / 100}')
            
            # Accumulate totals
            TOTAL_LINES=$((TOTAL_LINES + LINE_TOTAL))
            COVERED_LINES=$((COVERED_LINES + LINE_COVERED))
            
            if [ -n "$BRANCH_PERCENT" ] && [ -n "$BRANCH_TOTAL" ]; then
                BRANCH_PCT_NUM=${BRANCH_PERCENT%\%}
                BRANCH_COVERED=$(echo "$BRANCH_PCT_NUM $BRANCH_TOTAL" | awk '{printf "%.0f", $1 * $2 / 100}')
                TOTAL_BRANCHES=$((TOTAL_BRANCHES + BRANCH_TOTAL))
                COVERED_BRANCHES=$((COVERED_BRANCHES + BRANCH_COVERED))
                BRANCH_STR="$BRANCH_PERCENT"
            else
                BRANCH_STR="  n/a  "
            fi
            
            # Store for sorting: sort key (numeric), then the display line
            SORT_KEY=$(echo "$LINE_PERCENT" | sed 's/%//')
            
            echo "$SORT_KEY|$BASENAME|$LINE_PERCENT|$LINE_COVERED/$LINE_TOTAL|$BRANCH_STR" >> "$RESULTS_FILE"
            FILE_COUNT=$((FILE_COUNT + 1))
        fi
    fi
done

# Sort the results file for display
SORTED_FILE=$(mktemp)
sort -t'|' -k1 -rn "$RESULTS_FILE" > "$SORTED_FILE"

# Print header
printf "%-45s %10s %12s %10s\n" "Source File" "Lines" "Covered" "Branches"
echo "--------------------------------------------------------------------------"

# Display with colors (using process substitution to avoid subshell)
while IFS='|' read -r sort_key basename line_pct line_counts branch_pct; do
    # Color code based on percentage
    PERCENT_INT=${sort_key%.*}
    if [ "$PERCENT_INT" -ge 80 ] 2>/dev/null; then
        COLOR="\033[0;32m" # Green
    elif [ "$PERCENT_INT" -ge 50 ] 2>/dev/null; then
        COLOR="\033[0;33m" # Yellow
    else
        COLOR="\033[0;31m" # Red
    fi
    NC="\033[0m"
    
    printf "${COLOR}%-45s %10s %12s %10s${NC}\n" "$basename" "$line_pct" "$line_counts" "$branch_pct"
done < "$SORTED_FILE"

rm -f "$SORTED_FILE"

echo "--------------------------------------------------------------------------"

# Calculate and display totals
echo ""
if [ "$TOTAL_LINES" -gt 0 ]; then
    OVERALL_LINE_PCT=$((COVERED_LINES * 100 / TOTAL_LINES))
    echo "Line Coverage:   $OVERALL_LINE_PCT% ($COVERED_LINES of $TOTAL_LINES lines)"
fi

if [ "$TOTAL_BRANCHES" -gt 0 ]; then
    OVERALL_BRANCH_PCT=$((COVERED_BRANCHES * 100 / TOTAL_BRANCHES))
    echo "Branch Coverage: $OVERALL_BRANCH_PCT% ($COVERED_BRANCHES of $TOTAL_BRANCHES branches)"
fi

echo ""
echo "Files analyzed: $FILE_COUNT"
echo ""
echo "Legend: Green >= 80% | Yellow >= 50% | Red < 50%"
echo ""
