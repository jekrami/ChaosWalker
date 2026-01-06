#!/usr/bin/env python3
"""
ChaosWalker v1.0 - Decode Result
Decode the random index returned by ChaosWalker to get the original password
"""
import hashlib
import sys

# Import Smart Mapper
from smart_mapper import smart_encode, SMART_CHARSET

def base95_encode(number):
    """
    Convert number to password string using Smart Mapper v1.0

    Args:
        number: Integer index returned by GPU

    Returns:
        Password string using optimized character ordering
    """
    return smart_encode(number)

def sha256_hash(text):
    """Calculate SHA-256 hash of text"""
    return hashlib.sha256(text.encode()).hexdigest()

def decode_random_index(random_index):
    """
    Decode the random index to get the password
    
    Args:
        random_index: The value returned by the GPU in found_flag
    
    Returns:
        The password string
    """
    password = base95_encode(random_index)
    return password

# Main execution
if __name__ == "__main__":
    print("=" * 70)
    print("ChaosWalker Result Decoder")
    print("=" * 70)
    print()
    
    # Get the random index from command line or use the one from your output
    if len(sys.argv) > 1:
        random_index = int(sys.argv[1])
    else:
        # Default: the value from your SUCCESS message
        random_index = 4900706925914211
    
    print(f"Random Index from GPU: {random_index:,}")
    print()
    
    # Decode to password
    password = decode_random_index(random_index)
    
    print("=" * 70)
    print("RESULT")
    print("=" * 70)
    print()
    print(f"Password: '{password}'")
    print()
    
    # Verify by hashing
    hash_result = sha256_hash(password)
    print("Verification:")
    print(f"  SHA-256('{password}') = {hash_result}")
    print()
    
    # Show character breakdown
    print("Character Breakdown:")
    for i, char in enumerate(password):
        print(f"  [{i}] = '{char}' (ASCII {ord(char)})")
    print()
    
    print("=" * 70)
    print("✅ SUCCESS! Password recovered from random index.")
    print("=" * 70)

