# ChaosWalker Checkpoint System: "The Save Point"

## Overview

The checkpoint system ensures you **never lose progress** during long-running password searches. If your computer reboots, crashes, or you need to stop the search, you can resume exactly where you left off.

## How It Works

### Automatic Saving
- **Every 30 seconds**, the system saves your progress to `chaos_state.txt`
- Saves are **atomic** (write to temp file, then rename) to prevent corruption
- No performance impact - saves happen between batches

### Automatic Resuming
- On startup, checks for `chaos_state.txt`
- If found and target hash matches, **resumes from saved position**
- If target hash changed, starts fresh (prevents searching for wrong password)

### Automatic Cleanup
- When password is found, checkpoint file is **automatically deleted**
- Prevents accidentally resuming a completed search

## The Checkpoint File Format

```
# ChaosWalker Checkpoint File
# DO NOT EDIT MANUALLY
# Generated: 2026-01-06 15:30:45
current_linear_index=50000000
total_passwords_checked=50000000
target_hash=43171370af809a1ddb703b976848eae3a4c4157a781724ced6c03a403ebf6be8
```

### Fields Explained

- **current_linear_index**: Where to resume searching (next batch starts here)
- **total_passwords_checked**: Total passwords checked so far (for statistics)
- **target_hash**: The hash being searched for (validates checkpoint matches current target)

## Usage Examples

### Normal Operation (First Run)

```bash
$ cargo run --release

🆕 No checkpoint found. Starting from beginning.

Target loaded. Engine started.
Batch Size: 10000000 keys/cycle
Checkpoint: Saving every 30 seconds to chaos_state.txt

Checked: 50.0 M | Speed: 1234.56 M/sec | Offset: 50000000 [💾 Saved]
```

### Resuming After Interruption

```bash
# You stopped the program (Ctrl+C) or it crashed
$ cargo run --release

📂 CHECKPOINT FOUND!
   Resuming from: 50000000
   Already checked: 50000000 passwords

Target loaded. Engine started.
Batch Size: 10000000 keys/cycle
Checkpoint: Saving every 30 seconds to chaos_state.txt

Checked: 60.0 M | Speed: 1234.56 M/sec | Offset: 60000000 [💾 Saved]
```

### Success (Checkpoint Auto-Deleted)

```bash
!!! SUCCESS !!!
Target Found at Random Index: 4900706925914211
(Use: python3 decode_result.py 4900706925914211 to get the password)

✅ Checkpoint deleted (search complete)
```

## Manual Checkpoint Management

### View Current Checkpoint

```bash
cat chaos_state.txt
```

### Delete Checkpoint (Start Fresh)

```bash
rm chaos_state.txt
```

### Manually Edit Checkpoint (Advanced)

⚠️ **Not recommended**, but possible:

```bash
# Edit the file
nano chaos_state.txt

# Change current_linear_index to jump to a specific position
current_linear_index=1000000000

# Save and run
cargo run --release
```

## Configuration

Edit `src/main.rs` to change checkpoint behavior:

```rust
const CHECKPOINT_FILE: &str = "chaos_state.txt";
const CHECKPOINT_INTERVAL_SECS: u64 = 30; // Save every 30 seconds
```

### Recommended Settings

| Use Case | Interval | Reason |
|----------|----------|--------|
| Testing | 10 seconds | See saves happen quickly |
| Normal | 30 seconds | Good balance |
| High-speed GPU | 60 seconds | Reduce I/O overhead |
| Slow storage | 120 seconds | Minimize disk writes |

## Distributed Search Support

The checkpoint system enables **distributed searching** across multiple machines:

### Strategy 1: Partition by Range

```bash
# Machine 1: Search 0 to 1 billion
current_linear_index=0

# Machine 2: Search 1 billion to 2 billion
current_linear_index=1000000000

# Machine 3: Search 2 billion to 3 billion
current_linear_index=2000000000
```

### Strategy 2: Interleaved Search

```bash
# Machine 1: Every 3rd batch (0, 30M, 60M, ...)
current_linear_index=0
# Modify code to increment by BATCH_SIZE * 3

# Machine 2: Every 3rd batch (10M, 40M, 70M, ...)
current_linear_index=10000000

# Machine 3: Every 3rd batch (20M, 50M, 80M, ...)
current_linear_index=20000000
```

## Safety Features

### Atomic Writes
- Writes to `chaos_state.txt.tmp` first
- Then renames to `chaos_state.txt`
- Prevents corruption if power loss during save

### Hash Validation
- Checkpoint includes target hash
- Won't resume if you changed the target
- Prevents wasting time on wrong search

### Error Handling
- If checkpoint save fails, prints warning but continues
- If checkpoint load fails, starts fresh
- Never crashes due to checkpoint issues

## Performance Impact

- **Save time**: < 1 millisecond (atomic file write)
- **Load time**: < 1 millisecond (small text file)
- **Disk space**: < 1 KB per checkpoint
- **Network**: None (local file only)

**Result**: Zero measurable performance impact on GPU throughput.

## Troubleshooting

### Checkpoint Not Saving

```bash
# Check file permissions
ls -la chaos_state.txt

# Check disk space
df -h .

# Check for errors in output
cargo run --release 2>&1 | grep -i checkpoint
```

### Checkpoint Not Loading

```bash
# Verify file exists
cat chaos_state.txt

# Check format is correct
grep "current_linear_index" chaos_state.txt

# Delete and start fresh if corrupted
rm chaos_state.txt
```

### Wrong Resume Position

```bash
# View checkpoint
cat chaos_state.txt

# Manually set position
echo "current_linear_index=0" > chaos_state.txt
echo "total_passwords_checked=0" >> chaos_state.txt
echo "target_hash=YOUR_HASH_HERE" >> chaos_state.txt
```

## Best Practices

1. **Backup checkpoints** for very long searches:
   ```bash
   cp chaos_state.txt chaos_state_backup_$(date +%Y%m%d_%H%M%S).txt
   ```

2. **Monitor progress** with a script:
   ```bash
   watch -n 5 'cat chaos_state.txt | grep current_linear_index'
   ```

3. **Estimate completion time**:
   ```python
   current = 50_000_000  # From checkpoint
   speed = 1_234_560_000  # Hashes per second
   target = 1_000_000_000  # Estimated total
   remaining = target - current
   seconds = remaining / speed
   print(f"ETA: {seconds/3600:.1f} hours")
   ```

## Future Enhancements

Potential improvements to the checkpoint system:

- [ ] Cloud backup (S3, Google Drive)
- [ ] Multiple checkpoint files (keep last N)
- [ ] Checkpoint compression
- [ ] Network sync for distributed search
- [ ] Web dashboard showing progress
- [ ] Email/SMS alerts on completion

## Conclusion

The checkpoint system transforms ChaosWalker from a **one-shot tool** into a **long-running campaign**. You can now:

- ✅ Search for days/weeks/months without fear
- ✅ Survive reboots, crashes, power outages
- ✅ Pause and resume at will
- ✅ Distribute work across machines
- ✅ Never lose progress

**The war against entropy just got a save point.** 💾

