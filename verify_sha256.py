#!/usr/bin/env python3
"""
Verify SHA-256 implementation details
"""
import hashlib
import struct

def sha256_details(text):
    """Show SHA-256 details"""
    print(f"Input: '{text}'")
    print(f"Input bytes: {text.encode().hex()}")
    print()
    
    # Calculate hash
    hash_obj = hashlib.sha256(text.encode())
    hash_bytes = hash_obj.digest()
    hash_hex = hash_obj.hexdigest()
    
    print(f"SHA-256 hash (hex): {hash_hex}")
    print(f"SHA-256 hash (bytes): {hash_bytes.hex()}")
    print()
    
    # Convert to u32 array (big-endian)
    u32_array = []
    for i in range(8):
        chunk = hash_bytes[i*4:(i+1)*4]
        u32_val = struct.unpack('>I', chunk)[0]  # big-endian
        u32_array.append(f"{u32_val:08x}")
    
    print(f"As u32 array (big-endian): [{', '.join(u32_array)}]")
    print()
    
    # Show padding for single character
    if len(text) == 1:
        print("SHA-256 padding for 1-byte message:")
        data = bytearray(64)
        data[0] = ord(text[0])
        data[1] = 0x80  # Padding bit
        # Length in bits at end (big-endian)
        bit_len = 8  # 1 byte = 8 bits
        data[62] = 0
        data[63] = bit_len
        print(f"Padded block (hex): {data.hex()}")

if __name__ == "__main__":
    print("=" * 70)
    sha256_details("a")
    print("=" * 70)

