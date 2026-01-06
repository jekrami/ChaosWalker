# ChaosWalker Quick Start Guide

## Prerequisites

1. **NVIDIA GPU** with CUDA support (tested on RTX 3090)
2. **CUDA Toolkit** installed (11.4+ or 12.x)
3. **Rust** toolchain (1.70+)
4. **nvcc** compiler in PATH

## Installation

```bash
# Clone the repository
cd /home/ekrami/ChaosWalker

# Build the project (this will compile the CUDA kernel automatically)
cargo build --release
```

The `build.rs` script automatically compiles `kernels/chaos_worker.cu` to `kernels/chaos_worker.ptx`.

## Configuration

Edit `src/main.rs` to configure:

```rust
const BATCH_SIZE: usize = 1_000_000;  // Passwords per GPU launch
const GPU_PTX_PATH: &str = "./kernels/chaos_worker.ptx";
```

Set your target hash:
```rust
let target_hex = "17912ee268297e742817c187b5a1b3240247657954930379462509d37575209f";
```

## Running

```bash
cargo run --release
```

### Expected Output

```
--- Project ChaosWalker: Initiating High-Performance Mode ---
GPU Connected: RTX 3090 active.
Target loaded. Engine started.
Batch Size: 1000000 keys/cycle
Checked: 50.0 M | Speed: 1234.56 M/sec | Offset: 50000000
```

### On Success

```
!!! SUCCESS !!!
Target Found at Random Index: 123456789
(Use a Python script to convert this index back to the password string)
```

## Understanding the Output

- **Checked**: Total number of passwords tested
- **Speed**: Millions of passwords per second
- **Offset**: Current `start_index` value
- **Random Index**: The Feistel-encrypted index that produced the match

## Converting Random Index to Password

Create a Python script to reverse the process:

```python
def feistel_decrypt(ciphertext, rounds=4):
    """Reverse the Feistel encryption"""
    # Implementation matches the CUDA kernel
    pass

def base95_decode(number):
    """Convert number to Base-95 string"""
    if number == 0:
        return chr(32)
    
    result = []
    while number > 0:
        result.append(chr((number % 95) + 32))
        number //= 95
    
    return ''.join(result)

# Usage
random_index = 123456789  # From GPU output
linear_index = feistel_decrypt(random_index)
password = base95_decode(random_index)
print(f"Password: {password}")
```

## Performance Tuning

### Batch Size
- **Larger batches**: Better GPU utilization, less frequent CPU checks
- **Smaller batches**: More responsive, easier to interrupt
- **Recommended**: 1,000,000 for RTX 3090

### GPU Selection
If you have multiple GPUs, change the device ID:
```rust
let ctx = CudaContext::new(0)?;  // Change 0 to 1, 2, etc.
```

### Kernel Optimization
Edit `kernels/chaos_worker.cu`:
- Adjust Feistel rounds (currently 4)
- Modify the key seed (currently `0xDEADBEEF`)
- Change character set range (currently ASCII 32-126)

## Troubleshooting

### "Kernel not found"
- Ensure `build.rs` successfully compiled the CUDA kernel
- Check that `kernels/chaos_worker.ptx` exists
- Verify nvcc is in your PATH

### "CUDA error"
- Check GPU is available: `nvidia-smi`
- Verify CUDA toolkit version matches cudarc features
- Try reducing `BATCH_SIZE`

### Low Performance
- Use `--release` build mode
- Check GPU utilization: `nvidia-smi dmon`
- Ensure no other processes are using the GPU
- Verify PCIe link speed: `nvidia-smi -q | grep PCIe`

## Architecture Overview

```
CPU (Host)          GPU (Device)
    |                    |
    |-- start_index ---->|
    |    (8 bytes)       |
    |                    |--- Thread 0: Feistel → Base95 → SHA256 → Check
    |                    |--- Thread 1: Feistel → Base95 → SHA256 → Check
    |                    |--- Thread N: Feistel → Base95 → SHA256 → Check
    |                    |
    |<-- found_flag -----|
    |    (8 bytes)       |
```

**Key Insight**: Only 16 bytes transferred per 1,000,000 password checks!

## Next Steps

1. **Test with known password**: Generate SHA-256 of a test password and verify recovery
2. **Benchmark**: Measure your GPU's throughput
3. **Customize**: Modify Feistel parameters for your use case
4. **Scale**: Run on multiple GPUs or machines

## Advanced Usage

### Multi-GPU Setup
```rust
// Launch on GPU 0
let ctx0 = CudaContext::new(0)?;
// Launch on GPU 1
let ctx1 = CudaContext::new(1)?;
// Partition search space between them
```

### Resume from Checkpoint
```rust
let mut current_linear_index: u64 = 50_000_000_000; // Resume from 50B
```

### Custom Character Set
Edit `chaos_worker.cu`:
```cuda
// Change from ASCII 32-126 to custom range
temp_pass[len++] = (temp % 62) + 48; // 0-9, A-Z, a-z only
temp /= 62;
```

## Support

For issues or questions:
1. Check `ARCHITECTURE.md` for design details
2. Review CUDA kernel code in `kernels/chaos_worker.cu`
3. Verify Rust host code in `src/main.rs`

## License

See LICENSE file for details.

