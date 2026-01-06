#!/usr/bin/env python3
"""
Debug script to simulate what the GPU kernel should be doing
"""
import hashlib
from smart_mapper import SMART_CHARSET

def gpu_base95_encode(number):
    """Simulate GPU's base95 encoding"""
    if number == 0:
        return SMART_CHARSET[0]  # 'a'
    
    result = []
    temp = number
    while temp > 0:
        result.append(SMART_CHARSET[temp % 95])
        temp //= 95
    
    return ''.join(result)

def test_first_passwords():
    """Test first 10 passwords"""
    target_hash = "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb"
    
    print("Testing first 10 passwords (GPU simulation):")
    print("=" * 70)
    
    for i in range(10):
        pwd = gpu_base95_encode(i)
        hash_val = hashlib.sha256(pwd.encode()).hexdigest()
        match = "✅ MATCH!" if hash_val == target_hash else ""
        print(f"Index {i}: '{pwd}' -> {hash_val[:16]}... {match}")
    
    print()
    print("Expected: Index 0 should match (password 'a')")

if __name__ == "__main__":
    test_first_passwords()

