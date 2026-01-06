#!/bin/bash
# Demonstrate the checkpoint system with a longer-running search

echo "========================================================================"
echo "ChaosWalker Checkpoint System Demo"
echo "========================================================================"
echo ""

# Clean up
rm -f chaos_state.txt

echo "Step 1: Starting search (will run for 35 seconds to trigger checkpoint)"
echo "------------------------------------------------------------------------"
echo ""

# Run for 35 seconds to ensure at least one checkpoint is saved
timeout 35 cargo run --release 2>&1 | tee /tmp/chaoswalker_run1.log || true

echo ""
echo ""
echo "========================================================================"
echo "Step 2: Checking if checkpoint was created"
echo "========================================================================"
echo ""

if [ -f chaos_state.txt ]; then
    echo "✅ SUCCESS! Checkpoint file created:"
    echo ""
    cat chaos_state.txt
    echo ""
    
    # Extract the current index
    SAVED_INDEX=$(grep "current_linear_index=" chaos_state.txt | cut -d'=' -f2)
    echo "📊 Progress saved at index: $SAVED_INDEX"
    echo ""
else
    echo "⚠️  No checkpoint file found (search may have completed too quickly)"
    echo ""
    exit 0
fi

echo "========================================================================"
echo "Step 3: Resuming from checkpoint"
echo "========================================================================"
echo ""
echo "Press Ctrl+C after a few seconds to see it save again..."
echo ""

sleep 2

# Run again to show resume
timeout 35 cargo run --release 2>&1 | tee /tmp/chaoswalker_run2.log || true

echo ""
echo ""
echo "========================================================================"
echo "Step 4: Verify progress continued"
echo "========================================================================"
echo ""

if [ -f chaos_state.txt ]; then
    echo "✅ Checkpoint updated:"
    echo ""
    cat chaos_state.txt
    echo ""
    
    NEW_INDEX=$(grep "current_linear_index=" chaos_state.txt | cut -d'=' -f2)
    echo "📊 New progress: $NEW_INDEX"
    echo "📈 Advanced by: $((NEW_INDEX - SAVED_INDEX)) passwords"
    echo ""
else
    echo "✅ Search completed! Checkpoint was deleted."
    echo ""
fi

echo "========================================================================"
echo "Demo Complete!"
echo "========================================================================"
echo ""
echo "Key observations:"
echo "  • Checkpoint saves every 30 seconds automatically"
echo "  • Resume picks up exactly where you left off"
echo "  • No progress is lost on interruption"
echo "  • Checkpoint deleted automatically on success"
echo ""

