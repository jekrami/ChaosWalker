# ChaosWalker Architecture: GPU-Accelerated Password Cracking

## Overview
ChaosWalker implements a **zero-CPU-computation** architecture where all password generation, hashing, and checking happens entirely on the GPU. The CPU's only job is to trigger batches and check results.

## Design Philosophy

### The Problem with Traditional Approaches
Traditional GPU password crackers suffer from:
- **PCIe Bottleneck**: Transferring millions of password candidates from CPU to GPU
- **CPU Overhead**: Generating passwords on CPU wastes time
- **Memory Waste**: Storing large arrays of candidates in GPU memory

### The ChaosWalker Solution
**Move everything to the GPU.** The CPU sends only 8 bytes per batch of 1 million hashes.

## Architecture Components

### 1. CPU (Host) - "The Trigger"
**Role**: Minimal coordinator
**Responsibilities**:
- Initialize GPU context and load kernel
- Send a single `start_index` (8 bytes) per batch
- Launch kernel
- Read back `found_flag` (8 bytes)
- Increment counter and repeat

**Data Transfer Per Batch**:
- Upload: 8 bytes (start_index)
- Download: 8 bytes (found_flag)
- **Total: 16 bytes for 1,000,000 password checks**

### 2. GPU (Device) - "The Factory"
**Role**: Complete password lifecycle
**Responsibilities**: Each GPU thread independently:

#### Step A: Identity Calculation
```cuda
my_linear_id = start_index + threadIdx.x + blockIdx.x * blockDim.x
```
Example: If `start_index = 0` and thread ID = 42, then `my_linear_id = 42`

#### Step B: Feistel Scrambler
```cuda
random_id = feistel_encrypt(my_linear_id)
```
- Transforms sequential IDs into pseudo-random unique IDs
- Uses 4-round Feistel network with MurmurHash-style mixing
- Example: `0 → 59201`, `1 → 4102`, `2 → 891234`
- **Guarantees**: Bijective (one-to-one), no collisions

#### Step C: Base-95 Mapper
```cuda
password_string = base95_encode(random_id)
```
- Converts number to ASCII string using charset [32-126]
- Example: `59201 → "abc"`, `4102 → "xyz"`
- Supports passwords up to 16 characters

#### Step D: SHA-256 Hasher
```cuda
hash = sha256(password_string)
```
- Full SHA-256 implementation in GPU registers
- No memory access during hashing (ultra-fast)

#### Step E: Checker
```cuda
if (hash == target_hash) {
    *found_flag = random_id;
}
```
- Atomic write to global memory only on match
- Returns the **random_id**, not the linear_id

## Performance Characteristics

### Bandwidth Efficiency
| Metric | Traditional | ChaosWalker |
|--------|-------------|-------------|
| Data per 1M passwords | ~8 MB | 16 bytes |
| PCIe utilization | High | Near zero |
| CPU load | 50-100% | <1% |

### Memory Footprint
- **GPU Memory Required**:
  - Target hash: 32 bytes
  - Found flag: 8 bytes
  - **Total: 40 bytes** (plus kernel code)
- **No candidate storage needed**

### Scalability
- Batch size: 1,000,000 passwords per kernel launch
- RTX 3090: ~10,496 CUDA cores
- Each core processes ~95 passwords per batch
- Theoretical throughput: Limited only by GPU compute, not bandwidth

## Code Structure

### Rust Host (`src/main.rs`)
```rust
loop {
    // Send 8 bytes
    kernel.launch(start_index, &found_flag, &target_hash, batch_size);
    
    // Read 8 bytes
    let result = read_found_flag();
    
    if result != 0 {
        println!("Found at random_id: {}", result);
        break;
    }
    
    start_index += batch_size;
}
```

### CUDA Kernel (`kernels/chaos_worker.cu`)
```cuda
__global__ void crack_kernel(uint64_t start_index, ...) {
    // A: Identity
    uint64_t my_linear_id = start_index + threadIdx.x + blockIdx.x * blockDim.x;
    
    // B: Scrambler
    uint64_t random_id = feistel_encrypt(my_linear_id);
    
    // C: Mapper
    char password[16];
    base95_encode(random_id, password);
    
    // D: Hasher
    uint32_t hash[8];
    sha256(password, hash);
    
    // E: Checker
    if (memcmp(hash, target_hash, 32) == 0) {
        *found_flag = random_id;
    }
}
```

## Key Innovations

### 1. Feistel Network on GPU
- **Purpose**: Transform sequential space into random walk
- **Benefit**: Exhaustive coverage without storing visited states
- **Implementation**: Lightweight MurmurHash-style rounds (not full SHA-256)

### 2. In-Register Computation
- All operations happen in GPU registers
- Zero global memory access except final write
- Maximum throughput

### 3. Minimal Host-Device Communication
- Only control signals, no data
- PCIe bus is effectively idle
- Can scale to multiple GPUs trivially

## Usage

### Build
```bash
cargo build --release
```

### Run
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

## Future Optimizations

1. **Multi-GPU Support**: Partition search space across GPUs
2. **Async Kernel Launches**: Pipeline multiple batches
3. **Custom Feistel Keys**: User-configurable randomization
4. **Resume Capability**: Save/restore `start_index`
5. **Distributed Computing**: Network-based coordination

## Conclusion

ChaosWalker achieves **maximum GPU utilization** by eliminating all unnecessary data movement. The CPU becomes a simple trigger, while the GPU does 100% of the computational work. This architecture is the theoretical optimum for GPU-accelerated password cracking.

