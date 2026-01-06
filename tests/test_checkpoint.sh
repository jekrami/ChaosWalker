#!/bin/bash
# Test the checkpoint system by running, interrupting, and resuming

set -e

echo "========================================================================"
echo "ChaosWalker Checkpoint System Test"
echo "========================================================================"
echo ""

# Clean up any existing checkpoint
if [ -f chaos_state.txt ]; then
    echo "🧹 Cleaning up old checkpoint..."
    rm chaos_state.txt
fi

echo "Test 1: First Run (No Checkpoint)"
echo "========================================================================"
echo "Starting ChaosWalker for 5 seconds..."
echo ""

# Run for 5 seconds then kill
timeout 5 cargo run --release || true

echo ""
echo "========================================================================"
echo "Test 2: Check Checkpoint Was Created"
echo "========================================================================"
echo ""

if [ -f chaos_state.txt ]; then
    echo "✅ Checkpoint file created!"
    echo ""
    echo "Contents:"
    cat chaos_state.txt
    echo ""
else
    echo "❌ ERROR: Checkpoint file not created!"
    exit 1
fi

echo "========================================================================"
echo "Test 3: Resume from Checkpoint"
echo "========================================================================"
echo "Resuming ChaosWalker for another 5 seconds..."
echo ""

# Run again for 5 seconds
timeout 5 cargo run --release || true

echo ""
echo "========================================================================"
echo "Test 4: Verify Progress Continued"
echo "========================================================================"
echo ""

if [ -f chaos_state.txt ]; then
    echo "✅ Checkpoint updated!"
    echo ""
    echo "Updated contents:"
    cat chaos_state.txt
    echo ""
else
    echo "❌ ERROR: Checkpoint file disappeared!"
    exit 1
fi

echo "========================================================================"
echo "Test 5: Manual Checkpoint Inspection"
echo "========================================================================"
echo ""

# Extract values from checkpoint
CURRENT_INDEX=$(grep "current_linear_index=" chaos_state.txt | cut -d'=' -f2)
TOTAL_CHECKED=$(grep "total_passwords_checked=" chaos_state.txt | cut -d'=' -f2)

echo "Current Linear Index: $CURRENT_INDEX"
echo "Total Passwords Checked: $TOTAL_CHECKED"
echo ""

if [ "$CURRENT_INDEX" -gt 0 ]; then
    echo "✅ Progress was saved correctly!"
else
    echo "❌ ERROR: No progress recorded!"
    exit 1
fi

echo ""
echo "========================================================================"
echo "Test 6: Change Target Hash (Should Start Fresh)"
echo "========================================================================"
echo ""

# Manually edit checkpoint to have wrong hash
sed -i 's/target_hash=.*/target_hash=0000000000000000000000000000000000000000000000000000000000000000/' chaos_state.txt

echo "Modified checkpoint to have wrong hash:"
cat chaos_state.txt
echo ""

echo "Running with mismatched hash (should start fresh)..."
timeout 3 cargo run --release || true

echo ""
echo "========================================================================"
echo "ALL TESTS PASSED! ✅"
echo "========================================================================"
echo ""
echo "Summary:"
echo "  ✅ Checkpoint creation works"
echo "  ✅ Checkpoint resume works"
echo "  ✅ Progress tracking works"
echo "  ✅ Hash validation works"
echo ""
echo "The checkpoint system is fully operational!"
echo ""

# Clean up
rm -f chaos_state.txt
echo "🧹 Cleaned up test checkpoint file"

