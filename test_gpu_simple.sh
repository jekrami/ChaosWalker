#!/bin/bash
# Simple test: Search for password "a" which should be at index 0

echo "Testing ChaosWalker GPU kernel for password 'a'"
echo "Expected: Should find immediately (index 0)"
echo ""

# Clean up
rm -f chaos_state.txt

# Set batch size to 100 for quick test
cat > config.toml << 'EOF'
target_hash = "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb"
batch_size = 100
gpu_ptx_path = "./kernels/chaos_worker.ptx"
checkpoint_file = "chaos_state.txt"
checkpoint_interval_secs = 30
EOF

echo "Running with batch_size=100..."
timeout 2 cargo run --release

echo ""
echo "If it didn't find 'a' in the first batch, there's a bug in the GPU kernel!"

