#!/usr/bin/env python3
"""
Test script to understand the Feistel mapping and find where "jafar" appears
"""

def base95_encode(number):
    """Convert number to Base-95 string (matches CUDA kernel)"""
    if number == 0:
        return chr(32)  # Space character
    
    result = []
    temp = number
    while temp > 0:
        result.append(chr((temp % 95) + 32))
        temp //= 95
    
    return ''.join(result)

def base95_decode(string):
    """Convert Base-95 string to number"""
    result = 0
    for i, char in enumerate(string):
        result += (ord(char) - 32) * (95 ** i)
    return result

def murmur_mix(val, round_num, key_seed):
    """Matches the feistel_round function in CUDA kernel"""
    # Ensure 32-bit arithmetic
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

def feistel_decrypt(ciphertext, rounds=4, key_seed=0xDEADBEEF):
    """Decrypt using Feistel network"""
    mask32 = 0xFFFFFFFF
    
    left = (ciphertext >> 32) & mask32
    right = ciphertext & mask32
    
    # Reverse the rounds
    for r in range(rounds - 1, -1, -1):
        temp = left
        round_output = murmur_mix(left, r, key_seed)
        left = (right ^ round_output) & mask32
        right = temp
    
    return ((left << 32) | right) & 0xFFFFFFFFFFFFFFFF

# Test: Find where "jafar" appears
target_password = "jafar"
random_index = base95_decode(target_password)
linear_index = feistel_decrypt(random_index)

print("=" * 70)
print("ANALYSIS: Where does 'jafar' appear in the search space?")
print("=" * 70)
print()
print(f"Target Password: '{target_password}'")
print(f"Base-95 Encoding: {random_index:,}")
print(f"After Feistel Decrypt: {linear_index:,}")
print()
print("This means:")
print(f"  - The GPU will find 'jafar' when it reaches linear_index = {linear_index:,}")
print(f"  - With BATCH_SIZE = 10,000,000, this is batch #{linear_index // 10_000_000:,}")
print(f"  - Total passwords to check before finding it: {linear_index:,}")
print()

# Calculate search space size
print("=" * 70)
print("SEARCH SPACE ANALYSIS")
print("=" * 70)
print()

# Calculate how many passwords exist for each length
for length in range(1, 11):
    count = 95 ** length
    print(f"Length {length:2d}: {count:>20,} possible passwords")
    if length == len(target_password):
        print(f"           ^ 'jafar' is here (length {length})")

print()
print("Total search space (up to 10 chars):", f"{sum(95**i for i in range(1, 11)):,}")
print()

# Verify the mapping works
print("=" * 70)
print("VERIFICATION")
print("=" * 70)
print()
print(f"Linear Index {linear_index:,} ->")
print(f"  Feistel Encrypt -> {random_index:,}")
print(f"  Base-95 Encode  -> '{base95_encode(random_index)}'")
print()

# Show some nearby passwords
print("=" * 70)
print("NEARBY PASSWORDS (first 20 linear indices)")
print("=" * 70)
print()
for i in range(20):
    random_i = feistel_encrypt(i)
    password = base95_encode(random_i)
    print(f"Linear {i:3d} -> Random {random_i:12,} -> '{password}'")

print()
print("=" * 70)
print("CONCLUSION")
print("=" * 70)
print()
if linear_index > 1_000_000_000:
    print("⚠️  WARNING: 'jafar' appears VERY late in the search space!")
    print(f"   It would take {linear_index / 1_000_000_000:.1f} billion iterations to reach it.")
    print()
    print("   This is why the Feistel network is important:")
    print("   - It randomizes the search order")
    print("   - But it doesn't change the TOTAL search space")
    print("   - You still need to check the same number of passwords")
    print()
    print("   For testing, use a password that appears earlier, like:")
    for i in range(100):
        random_i = feistel_encrypt(i)
        password = base95_encode(random_i)
        if len(password) <= 5 and password.isalnum():
            print(f"     Linear {i} -> '{password}'")
            if i > 10:
                break
else:
    print(f"✅ 'jafar' should be found relatively quickly!")
    print(f"   Expected batches: {linear_index // 10_000_000:,}")

