#!/usr/bin/env python3
"""
Simple test to verify dashboard can find and display password
"""
import subprocess
import re
from smart_mapper import smart_encode

print("=" * 70)
print("Dashboard Engine Test")
print("=" * 70)
print()

# Run ChaosWalker engine directly
print("Running engine...")
result = subprocess.run(
    ["./target/release/chaos_walker"],
    capture_output=True,
    text=True,
    timeout=10
)

print("Engine output:")
print(result.stdout)
print()

# Check if success was found
if "SUCCESS" in result.stdout:
    print("✅ SUCCESS detected in output!")
    
    # Extract index
    match = re.search(r'Random Index:\s*(\d+)', result.stdout)
    if match:
        index = int(match.group(1))
        print(f"✅ Index extracted: {index}")
        
        # Decode password
        password = smart_encode(index)
        print(f"✅ Password decoded: '{password}'")
        print()
        print("=" * 70)
        print(f"🎉 RESULT: The password is '{password}'")
        print("=" * 70)
    else:
        print("❌ Could not extract index from output")
else:
    print("❌ No SUCCESS found in output")
    print("This might take longer - the password may not be found quickly")
