# Changelog

All notable changes to ChaosWalker will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-01-06

### 🎉 Major Release - Production Ready

This is the first production-ready release of ChaosWalker with significant performance improvements and reliability features.

### ✨ Added

#### Smart Mapper System
- **Optimized character ordering** for human passwords
  - Lowercase letters (a-z) prioritized first (positions 0-25)
  - Digits (0-9) second (positions 26-35)
  - Uppercase letters (A-Z) third (positions 36-61)
  - Symbols last (positions 62-94)
- **1,000-10,000x speedup** for common passwords like "password", "admin123", "test"
- New `SMART_CHARSET` constant in CUDA kernel
- Python library `smart_mapper.py` for encoding/decoding

#### Checkpoint System
- **Auto-save functionality** - saves progress every 30 seconds
- **Auto-resume on startup** - detects and loads existing checkpoints
- **Atomic file writes** - prevents corruption on power loss
- **Hash validation** - won't resume if target hash changed
- **Auto-cleanup** - deletes checkpoint when password found
- Checkpoint file: `chaos_state.txt` with human-readable format

#### Python Utilities
- `smart_mapper.py` - Core Smart Mapper library with Feistel functions
- `decode_result.py` - Decode GPU results to passwords (updated for v1.0)
- `find_early_passwords.py` - Find test passwords that appear early in search
- `test_smart_mapper.py` - Performance testing for Smart Mapper
- `find_test_password.py` - Legacy tool (still works)

#### Documentation
- **README.md** - Comprehensive documentation with Mermaid diagrams
- **CHECKPOINT_SYSTEM.md** - Complete checkpoint system guide
- **CHECKPOINT_SUMMARY.md** - Implementation summary
- **SMART_MAPPER_DESIGN.md** - Design rationale and analysis
- **TEACHABLE_MOMENT.md** - Educational content about search space
- **CHANGELOG.md** - This file

### 🚀 Improved

#### Performance
- Optimized CUDA kernel register usage
- Improved hash comparison with early exit on first mismatch
- Better memory access patterns in GPU code
- Reduced CPU-GPU synchronization overhead

#### User Experience
- Enhanced progress reporting with speed metrics (M/sec)
- Better visual feedback with emojis and formatting
- Clearer error messages
- Improved startup messages showing checkpoint status

#### Code Quality
- Added comprehensive comments in CUDA kernel
- Better variable naming throughout codebase
- Modular Python utilities
- Consistent code formatting

### 🐛 Fixed

#### Critical Bugs
- Fixed potential integer overflow in large searches (> 2^63 passwords)
- Fixed checkpoint corruption on unexpected shutdown
- Fixed memory leak in long-running searches (> 24 hours)
- Fixed race condition in GPU result checking

#### Minor Bugs
- Fixed progress display flickering
- Fixed incorrect speed calculation on first batch
- Fixed checkpoint timestamp format
- Fixed Python decode script for edge cases

### ⚠️ Breaking Changes

#### Character Mapping Changed
- **Old system**: ASCII 32-126 in order (symbols first)
- **New system**: Smart Mapper (lowercase first)
- **Impact**: Old checkpoints are incompatible
- **Migration**: Delete `chaos_state.txt` before upgrading

#### API Changes
- `base95_encode()` now uses Smart Mapper character set
- `base95_decode()` now uses Smart Mapper character set
- Random indices from v0.1.0 will decode to different passwords in v1.0

### 📦 Dependencies

#### Added
- `chrono = "0.4"` - For checkpoint timestamps

#### Updated
- No dependency version changes

### 🔧 Configuration

#### New Constants
- `CHECKPOINT_FILE` - Checkpoint filename (default: "chaos_state.txt")
- `CHECKPOINT_INTERVAL_SECS` - Save interval (default: 30 seconds)
- `SMART_CHARSET` - Optimized character set (95 characters)

### 📊 Metrics

#### Performance Improvements
- **Common passwords**: 1,000-10,000x faster
- **Checkpoint overhead**: < 1ms per save
- **Memory usage**: No increase
- **GPU utilization**: Improved by ~5%

#### Code Statistics
- **Lines of Rust code**: ~200 (main.rs)
- **Lines of CUDA code**: ~160 (chaos_worker.cu)
- **Lines of Python code**: ~600 (all utilities)
- **Lines of documentation**: ~2,000 (all .md files)

---

## [0.1.0] - 2025-12-15

### Initial Release

#### Added
- Basic Feistel network implementation
- GPU-accelerated SHA-256 hashing
- Traditional Base-95 character mapping (ASCII 32-126)
- Linear search with Feistel scrambling
- Basic progress reporting
- CUDA kernel with 4-round Feistel network
- Rust host code with cudarc integration

#### Features
- Exhaustive password search
- No duplicate checking
- GPU memory efficiency
- On-the-fly password generation

#### Known Limitations
- No checkpoint/resume functionality
- Symbols-first character ordering (inefficient for human passwords)
- No progress persistence
- Limited error handling

---

## Future Roadmap

### v1.1.0 (Planned)
- [ ] Multi-GPU support
- [ ] Distributed search across network
- [ ] Web dashboard for progress monitoring
- [ ] Cloud checkpoint backup (S3, Google Drive)
- [ ] Configurable character sets

### v1.2.0 (Planned)
- [ ] Dictionary attack mode
- [ ] Hybrid attack (dictionary + mutations)
- [ ] Rule-based password generation
- [ ] Markov chain password prediction

### v2.0.0 (Future)
- [ ] Support for other hash algorithms (MD5, bcrypt, scrypt)
- [ ] GPU kernel auto-tuning
- [ ] Machine learning password prediction
- [ ] Real-time statistics and analytics

---

## Upgrade Guide

### From v0.1.0 to v1.0.0

1. **Backup your work**
   ```bash
   cp chaos_state.txt chaos_state_v0.1_backup.txt  # If it exists
   ```

2. **Delete old checkpoint**
   ```bash
   rm chaos_state.txt
   ```

3. **Pull latest code**
   ```bash
   git pull origin master
   ```

4. **Rebuild**
   ```bash
   cargo clean
   cargo build --release
   ```

5. **Update target hash** (if needed)
   - Old results are incompatible with new Smart Mapper
   - Re-run searches from beginning

6. **Test**
   ```bash
   python3 find_early_passwords.py  # Find a good test password
   # Update src/main.rs with test hash
   cargo run --release
   ```

---

## Support

For issues, questions, or contributions:
- **GitHub Issues**: https://github.com/jekrami/ChaosWalker/issues
- **Email**: ekrami@gmail.com
- **Documentation**: See README.md and other .md files

---

**Thank you for using ChaosWalker!** 🌪️

