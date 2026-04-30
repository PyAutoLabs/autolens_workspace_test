#!/bin/bash

WORKSPACE=/mnt/c/Users/Jammy/Code/PyAutoLabs/autolens_workspace_test
FAILED_DIR="$WORKSPACE/failed"
mkdir -p "$FAILED_DIR"

cd "$WORKSPACE"

TOTAL=0
PASSED=0
FAILED=0

find scripts -name "*.py" | sort | while read script; do
    TOTAL=$((TOTAL + 1))
    echo "[$TOTAL] Running: $script"

    output=$(PYAUTO_TEST_MODE=1 python "$script" 2>&1)
    exit_code=$?

    if [ $exit_code -ne 0 ]; then
        FAILED=$((FAILED + 1))
        safe_name=$(echo "$script" | tr '/' '__')
        fail_file="$FAILED_DIR/${safe_name}.txt"
        echo "=== FAILED: $script ===" > "$fail_file"
        echo "Exit code: $exit_code" >> "$fail_file"
        echo "" >> "$fail_file"
        echo "$output" >> "$fail_file"
        echo "  FAILED -> $fail_file"
    else
        PASSED=$((PASSED + 1))
        echo "  OK"
    fi
done

echo ""
echo "Done."
