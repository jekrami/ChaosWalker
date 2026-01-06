# Length-Aware Optimization Strategy

## The Problem

If you know the password length, you're wasting time checking all shorter passwords:

```
Known length = 8 characters
Wasted checks:
  - Length 1-7: ~770 billion passwords
  - Time wasted: ~10 minutes @ 1.3 GH/s
```

## The Solution: Start Offset

### Formula for Starting Index

```python
def get_start_index(target_length, base=95):
    """Calculate first index of passwords with target_length"""
    if target_length == 1:
        return 0
    
    # Sum of all passwords shorter than target_length
    start = 0
    for length in range(1, target_length):
        start += base ** length
    
    return start

# Examples:
# Length 1: start = 0
# Length 2: start = 95
# Length 3: start = 95 + 9,025 = 9,120
# Length 8: start = 770,688,256,950 (~770 billion)
```

### Implementation

Add to `config.toml`:
```toml
# Password length optimization (0 = check all lengths)
known_password_length = 8  # Skip lengths 1-7
```

Add to `src/main.rs`:
```rust
fn calculate_start_offset(known_length: usize) -> u64 {
    if known_length <= 1 {
        return 0;
    }
    
    let mut offset = 0u64;
    for len in 1..known_length {
        offset += 95u64.pow(len as u32);
    }
    offset
}
```

## Strategy by Mode

### LINEAR MODE + Known Length

**Best approach: Offset start index**

```rust
// In worker_thread()
let length_offset = calculate_start_offset(config.known_password_length);
let start_linear_index = start_linear_index + length_offset;
```

**Benefits:**
- Skip all shorter passwords
- Direct jump to target length range
- Simple, no kernel changes needed

**Example:** For length 8:
```
Normal start:  0 → "a"
Optimized:     770,688,256,950 → "aaaaaaaa" (first 8-char password)
```

### RANDOM MODE + Known Length

**Problem:** Random indices don't correlate with password length!

```
Random index 0 → Could be any length password
Random index 1000 → Could be any length password
```

**Solutions:**

#### Option 1: Filter in Kernel (Recommended)
Check length after generation, skip if wrong:

```cuda
// In chaos_worker_random.cu
if (len != EXPECTED_LENGTH) {
    return;  // Skip this password
}
```

**Pros:** Simple, works with Feistel
**Cons:** ~99% of checks wasted for length 8

#### Option 2: Pre-compute Valid Indices (Advanced)
Calculate which random indices map to length 8, only check those.

**Pros:** Zero wasted checks
**Cons:** Complex, requires pre-computation

#### Option 3: Hybrid Feistel (Best for 8+)
Apply Feistel ONLY within the length-8 range:

```cuda
// Start at first length-8 password
uint64_t base_offset = 770688256950;
uint64_t length_8_space = 6634204312890625;

// Apply Feistel within this range
uint64_t my_linear_id = start_index + idx;
uint64_t scrambled = feistel_encrypt_bounded(my_linear_id, length_8_space);
uint64_t password_index = base_offset + scrambled;
```

**Pros:** Random search within length-8 space only
**Cons:** Requires bounded Feistel implementation

## Recommendations by Use Case

### 1. Known Length 8, LINEAR MODE (Fastest)
```toml
gpu_ptx_path = "./kernels/chaos_worker_linear.ptx"
known_password_length = 8
start_offset = 770688256950
```

**Performance:**
- Skip 770B passwords instantly
- Start checking "aaaaaaaa" immediately
- Smart Mapper puts common patterns first

**Best for:** 
- Corporate passwords (minimum 8 chars policy)
- Known password policies

### 2. Known Length 8, RANDOM MODE (Most thorough)
```toml
gpu_ptx_path = "./kernels/chaos_worker_random.ptx"
known_password_length = 8
start_offset = 770688256950
```

**Implementation:** Use Option 3 (Hybrid Feistel)

**Performance:**
- Random search within 6.6 quadrillion length-8 passwords
- No wasted checks on shorter passwords
- Unpredictable coverage

**Best for:**
- Research/compliance needing random sampling
- Long-term exhaustive search
- No assumptions about password patterns

### 3. Unknown Length, LINEAR MODE (Default)
```toml
gpu_ptx_path = "./kernels/chaos_worker_linear.ptx"
known_password_length = 0  # Check all
```

**Performance:**
- Checks 1, 2, 3... 8+ character passwords
- Finds short passwords fast
- Long passwords eventually

**Best for:**
- No knowledge of password policy
- Wide net approach

### 4. Unknown Length, RANDOM MODE (Deep mining)
```toml
gpu_ptx_path = "./kernels/chaos_worker_random.ptx"
known_password_length = 0  # Check all
```

**Performance:**
- Random across entire keyspace
- All lengths randomly distributed

**Best for:**
- Maximum unpredictability
- Long-term research

## Implementation Priority

### Phase 1: Easy Win (5 minutes)
Add `start_offset` parameter to skip lengths 1-7:

```rust
// src/main.rs
let global_index = Arc::new(AtomicU64::new(
    start_linear_index + 770_688_256_950  // Skip to length 8
));
```

**Benefit:** Saves 10 minutes on every run!

### Phase 2: Config-driven (30 minutes)
Add to config.toml and calculate offset dynamically:

```toml
known_password_length = 8
```

```rust
let offset = if config.known_password_length > 0 {
    calculate_start_offset(config.known_password_length)
} else {
    0
};
```

### Phase 3: Bounded Feistel (2 hours)
Implement Feistel that works within specific length range.

## Real-World Example

### Scenario: Corporate password (8-12 chars, known)

**Without optimization:**
```
Check: a, b, c, ..., (770B passwords) ..., aaaaaaaa
Time wasted: 10 minutes
```

**With optimization:**
```
Start: aaaaaaaa (first 8-char)
Time saved: 10 minutes PER RUN
```

**For 12-char minimum:**
```
Start offset: 5,604,398,009,217,682,950 (5.6 quintillion)
Time saved: ~50 days @ 1.3 GH/s!
```

## Code Template

```rust
// Add to Config struct
#[derive(Deserialize, Clone)]
struct Config {
    target_hash: String,
    batch_size: usize,
    gpu_ptx_path: String,
    checkpoint_file: String,
    checkpoint_interval_secs: u64,
    known_password_length: usize,  // NEW
}

// Helper function
fn calculate_start_offset(known_length: usize) -> u64 {
    if known_length <= 1 {
        return 0;
    }
    
    let mut offset = 0u64;
    for len in 1..known_length {
        // Be careful of overflow for large lengths!
        offset = offset.saturating_add(95u64.pow(len as u32));
    }
    offset
}

// In main()
let length_offset = calculate_start_offset(config.known_password_length);
let start_linear_index = checkpoint_index + length_offset;

println!("Known password length: {}", config.known_password_length);
println!("Starting at offset: {} (skipping shorter passwords)", length_offset);
```

## Performance Impact

| Password Length | Passwords Skipped | Time Saved @ 1.3 GH/s |
|-----------------|-------------------|------------------------|
| 1 | 0 | 0 |
| 2 | 95 | < 1 ms |
| 3 | 9,120 | < 1 ms |
| 4 | 866,495 | < 1 ms |
| 5 | 81,316,120 | 0.06 seconds |
| 6 | 7,819,125,495 | 6 seconds |
| 7 | 743,010,016,120 | 9.5 minutes |
| 8 | 70,576,739,625,495 | 15 hours |
| 9 | 6,704,745,649,140,120 | 60 days |
| 10 | 637,151,036,168,316,245 | 15.5 years |

**Bottom line:** For 8+ character passwords, the optimization is ESSENTIAL!
