#!/usr/bin/env python3
"""
Find a good test password that appears early in the linear search
"""
import hashlib

def base95_encode(number):
    """Convert number to Base-95 string (matches CUDA kernel)"""
    if number == 0:
        return chr(32)
    
    result = []
    temp = number
    while temp > 0:
        result.append(chr((temp % 95) + 32))
        temp //= 95
    
    return ''.join(result)

def murmur_mix(val, round_num, key_seed):
    """Matches the feistel_round function in CUDA kernel"""
    mask32 = 0xFFFFFFFF
    
    k = (val * 0xcc9e2d51) & mask32
    k = ((k << 15) | (k >> 17)) & mask32
    k = (k * 0x1b873593) & mask32
    
    h = (key_seed ^ k) & mask32
    h = ((h << 13) | (h >> 19)) & mask32
    h = (h * 5 + 0xe6546b64 + round_num) & mask32
    
    h ^= h >> 16
    h = (h * 0x85ebca6b) & mask32
    h ^= h >> 13
    h = (h * 0xc2b2ae35) & mask32
    h ^= h >> 16
    
    return h & mask32

def feistel_encrypt(plaintext, rounds=4, key_seed=0xDEADBEEF):
    """Encrypt using Feistel network (matches CUDA kernel)"""
    mask32 = 0xFFFFFFFF
    
    left = (plaintext >> 32) & mask32
    right = plaintext & mask32
    
    for r in range(rounds):
        temp = right
        round_output = murmur_mix(right, r, key_seed)
        right = (left ^ round_output) & mask32
        left = temp
    
    return ((left << 32) | right) & 0xFFFFFFFFFFFFFFFF

def sha256_hash(text):
    """Calculate SHA-256 hash"""
    return hashlib.sha256(text.encode()).hexdigest()

print("=" * 70)
print("FINDING GOOD TEST PASSWORDS")
print("=" * 70)
print()
print("Searching for passwords that appear in the first 100 million linear indices...")
print()

candidates = []

for linear_idx in range(100_000_000):
    random_idx = feistel_encrypt(linear_idx)
    password = base95_encode(random_idx)
    
    # Filter for readable passwords
    if (len(password) >= 3 and len(password) <= 8 and 
        password.isalnum() and password.isascii()):
        
        hash_val = sha256_hash(password)
        candidates.append((linear_idx, password, hash_val))
        
        if len(candidates) >= 20:
            break
    
    if linear_idx % 10_000_000 == 0 and linear_idx > 0:
        print(f"  Searched {linear_idx:,} indices... found {len(candidates)} candidates")

print()
print("=" * 70)
print("TOP 20 TEST PASSWORD CANDIDATES")
print("=" * 70)
print()
print(f"{'Linear Index':<15} {'Password':<15} {'SHA-256 Hash'}")
print("-" * 70)

for linear_idx, password, hash_val in candidates[:20]:
    batch_num = linear_idx // 10_000_000
    print(f"{linear_idx:<15,} {password:<15} {hash_val}")
    print(f"                ^ Batch #{batch_num}, will be found in ~{linear_idx/1_000_000:.1f}M checks")
    print()

print("=" * 70)
print("RECOMMENDATION")
print("=" * 70)
print()

if candidates:
    best = candidates[0]
    print(f"Use this password for testing: '{best[1]}'")
    print()
    print("Update your src/main.rs:")
    print(f'  let target_hex = "{best[2]}";')
    print()
    print(f"Expected result:")
    print(f"  - Will be found at linear_index: {best[0]:,}")
    print(f"  - Batch number: {best[0] // 10_000_000}")
    print(f"  - Time to find: ~{best[0]/1_000_000:.1f} million checks")
    print()
    print("This should complete in seconds on an RTX 3090!")
else:
    print("No suitable candidates found in first 100M. Try increasing search range.")

