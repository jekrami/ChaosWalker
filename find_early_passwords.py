#!/usr/bin/env python3
"""
Find passwords that appear EARLY in the linear search (v1.0 Smart Mapper)
"""
import hashlib
from smart_mapper import smart_encode, feistel_encrypt

def sha256_hash(text):
    """Calculate SHA-256 hash"""
    return hashlib.sha256(text.encode()).hexdigest()

print("=" * 70)
print("Finding Early Passwords for ChaosWalker v1.0 (Smart Mapper)")
print("=" * 70)
print()
print("Searching first 100 million linear indices...")
print()

candidates = []

for linear_idx in range(100_000_000):
    # Encrypt to get random index
    random_idx = feistel_encrypt(linear_idx)
    
    # Convert to password using Smart Mapper
    password = smart_encode(random_idx)
    
    # Filter for readable passwords
    if (len(password) >= 3 and len(password) <= 8 and 
        password.isalnum() and password.isascii()):
        
        hash_val = sha256_hash(password)
        candidates.append((linear_idx, password, hash_val, random_idx))
        
        if len(candidates) >= 20:
            break
    
    if linear_idx % 10_000_000 == 0 and linear_idx > 0:
        print(f"  Searched {linear_idx:,} indices... found {len(candidates)} candidates")

print()
print("=" * 70)
print("TOP 20 TEST PASSWORD CANDIDATES (Smart Mapper v1.0)")
print("=" * 70)
print()
print(f"{'Linear Index':<15} {'Password':<15} {'Random Index':<20} {'SHA-256 Hash'}")
print("-" * 70)

for linear_idx, password, hash_val, random_idx in candidates[:20]:
    batch_num = linear_idx // 10_000_000
    print(f"{linear_idx:<15,} {password:<15} {random_idx:<20,} {hash_val}")
    print(f"                ^ Batch #{batch_num}, ~{linear_idx/1_000_000:.1f}M checks")
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
    print(f"  - Random index: {best[3]:,}")
    print(f"  - Batch number: {best[0] // 10_000_000}")
    print(f"  - Time to find: ~{best[0]/1_000_000:.1f} million checks")
    print()
    print("This should complete in seconds on an RTX 3090!")
    print()
    print("To decode the result:")
    print(f"  python3 decode_result.py {best[3]}")
else:
    print("No suitable candidates found in first 100M. Try increasing search range.")

