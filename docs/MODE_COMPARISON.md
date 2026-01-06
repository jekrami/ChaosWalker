# ChaosWalker: LINEAR vs RANDOM Mode

## Quick Reference Card

### 🚀 LINEAR MODE (Default)
```toml
gpu_ptx_path = "./kernels/chaos_worker_linear.ptx"
```
- **Search Order**: Sequential (a, b, c, ..., aa, ab, ...)
- **Speed**: FAST for human passwords
- **Best For**: 99% of use cases
- **Example**: Finds "password" in 4 seconds

### 🎲 RANDOM MODE
```toml
gpu_ptx_path = "./kernels/chaos_worker_random.ptx"
```
- **Search Order**: Pseudo-random (Feistel network)
- **Speed**: Random (may take longer for common passwords)
- **Best For**: Exhaustive research, security compliance
- **Example**: "password" could be found anywhere

## Real Test: Password "a"

| Mode | Result |
|------|--------|
| **LINEAR** | Found in < 0.1 seconds at index 0 ✅ |
| **RANDOM** | Still searching after 6+ billion checks ❌ |

## The Math

```
LINEAR:  Index 0 → "a" (Direct)
RANDOM:  Index 0 → Feistel → 5,250,597,704,285,566,177 → "~]x9K#@L"
```

Password "a" exists at random index 0, but you need to find which linear index maps to it!

## When to Switch

**Start with LINEAR** (recommended)
↓
Find 90%+ of weak passwords in hours
↓
**Switch to RANDOM** (optional)
↓
Deep-space mining for months

## File Locations

- Linear kernel: `kernels/chaos_worker_linear.cu` → `.ptx`
- Random kernel: `kernels/chaos_worker_random.cu` → `.ptx`
- Configuration: `config.toml`
- Full guide: `HYBRID_MODE_GUIDE.md`
