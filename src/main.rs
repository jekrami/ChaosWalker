use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use std::convert::TryInto;
use std::time::{Instant, Duration};
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;

// --- CONFIGURATION ---
use serde::Deserialize;

// --- CONFIGURATION STRUCT ---
#[derive(Deserialize, Clone)]
struct Config {
    target_hash: String,
    batch_size: usize,
    gpu_ptx_path: String,
    checkpoint_file: String,
    checkpoint_interval_secs: u64,
    known_password_length: usize,
}

// --- LENGTH OPTIMIZATION ---

/// Calculate the starting index for passwords of a specific length
/// This allows skipping all shorter passwords when length is known
fn calculate_start_offset(known_length: usize) -> u64 {
    if known_length <= 1 {
        return 0;
    }
    
    let mut offset = 0u64;
    for len in 1..known_length {
        // Use saturating_add to prevent overflow for very large lengths
        offset = offset.saturating_add(95u64.pow(len as u32));
    }
    offset
}

// --- CHECKPOINT SYSTEM ---

/// Save the current search state to disk
fn save_checkpoint(filename: &str, current_index: u64, total_checked: u64, target_hex: &str) -> io::Result<()> {
    let checkpoint_data = format!(
        "# ChaosWalker Checkpoint File\n\
         # DO NOT EDIT MANUALLY\n\
         # Generated: {}\n\
         current_linear_index={}\n\
         total_passwords_checked={}\n\
         target_hash={}\n",
        chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
        current_index,
        total_checked,
        target_hex
    );

    // Write atomically: write to temp file, then rename
    let temp_file = format!("{}.tmp", filename);
    fs::write(&temp_file, checkpoint_data)?;
    fs::rename(&temp_file, filename)?;

    Ok(())
}

/// Load the checkpoint from disk, if it exists
fn load_checkpoint(filename: &str) -> Option<(u64, u64, String)> {
    if !Path::new(filename).exists() {
        return None;
    }

    let content = match fs::read_to_string(filename) {
        Ok(c) => c,
        Err(_) => return None,
    };

    let mut current_index = None;
    let mut total_checked = None;
    let mut target_hash = None;

    for line in content.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }

        if let Some(value) = line.strip_prefix("current_linear_index=") {
            current_index = value.parse().ok();
        } else if let Some(value) = line.strip_prefix("total_passwords_checked=") {
            total_checked = value.parse().ok();
        } else if let Some(value) = line.strip_prefix("target_hash=") {
            target_hash = Some(value.to_string());
        }
    }

    match (current_index, total_checked, target_hash) {
        (Some(idx), Some(total), Some(hash)) => Some((idx, total, hash)),
        _ => None,
    }
}

fn worker_thread(
    device_id: usize,
    config: Config,
    global_index: Arc<AtomicU64>,
    found_flag: Arc<AtomicBool>,
    result_store: Arc<Mutex<Option<u64>>>,
    ptx_content: String,
) -> Result<(), anyhow::Error> {
    // 1. Connect to specific GPU
    let ctx = CudaContext::new(device_id as usize)?;
    let stream = ctx.default_stream();
    
    // 2. Load Kernel
    let ptx = Ptx::from_src(&ptx_content);
    let module = ctx.load_module(ptx)?;
    let kernel = module.load_function("crack_kernel")?;

    // 3. Prepare Target
    let target_bytes = hex::decode(&config.target_hash)?;
    let mut target_u32 = [0u32; 8];
    for i in 0..8 {
        let chunk = &target_bytes[i*4..(i+1)*4];
        target_u32[i] = u32::from_be_bytes(chunk.try_into()?);
    }

    // 4. Allocate Memory
    // Initialize to sentinel value (max u64) so we can detect when index 0 is found
    let sentinel = vec![u64::MAX];
    let mut dev_found = stream.clone_htod(&sentinel)?;
    let dev_target = stream.clone_htod(&target_u32)?;
    let cfg = LaunchConfig::for_num_elems(config.batch_size as u32);

    let batch_size_u64 = config.batch_size as u64;

    // 5. Work Loop
    loop {
        if found_flag.load(Ordering::Relaxed) {
            break;
        }

        // Grab a batch of work
        let start_index = global_index.fetch_add(batch_size_u64, Ordering::Relaxed);

        // Run Kernel
        unsafe {
            stream.launch_builder(&kernel)
                .arg(&start_index)
                .arg(&mut dev_found)
                .arg(&dev_target)
                .arg(&(config.batch_size as i32))
                .launch(cfg)?;
        }

        // IMPORTANT: Synchronize context before reading result!
        ctx.synchronize()?;

        // Check Result
        let found_val = stream.clone_dtoh(&dev_found)?;

        // Check if value changed from sentinel (meaning password was found)
        if found_val[0] != u64::MAX {
            // Found it!
            found_flag.store(true, Ordering::SeqCst);
            let mut lock = result_store.lock().unwrap();
            *lock = Some(found_val[0]);
            break;
        }
    }

    Ok(())
}

fn main() -> Result<(), anyhow::Error> {
    println!("--- Project ChaosWalker v1.2: Multi-GPU Edition ---");

    // 0. LOAD CONFIGURATION
    let config_content = fs::read_to_string("config.toml")
        .map_err(|_| anyhow::anyhow!("Could not find config.toml"))?;
    let config: Config = toml::from_str(&config_content)?;
    
    // 1. DISCOVER GPUS
    // We try to connect to device 0 to see if it works, then count 
    // In cudarc 0.18, we can just try to iterate.
    // Helper via nvml or driver is safer, but let's assume at least 1 and try to connect.
    // Actually cudarc doesn't expose device_count() directly on top level easily without init.
    // We will just use a loop to detect.
    
    let mut device_ids = Vec::new();
    for i in 0..8 {
        if CudaContext::new(i).is_ok() {
            device_ids.push(i);
        } else {
            break;
        }
    }

    if device_ids.is_empty() {
        anyhow::bail!("No CUDA devices found!");
    }

    println!("Detected {} CUDA Device(s)", device_ids.len());

    // 2. LENGTH OPTIMIZATION
    let length_offset = calculate_start_offset(config.known_password_length);
    if config.known_password_length > 0 {
        println!("🎯 Length optimization enabled: {} characters", config.known_password_length);
        println!("   Skipping first {} passwords (lengths 1-{})", 
                 length_offset, config.known_password_length - 1);
        
        // Calculate time saved
        let seconds_saved = length_offset as f64 / 1_300_000_000.0; // @ 1.3 GH/s
        if seconds_saved < 60.0 {
            println!("   Time saved: {:.1} seconds", seconds_saved);
        } else if seconds_saved < 3600.0 {
            println!("   Time saved: {:.1} minutes", seconds_saved / 60.0);
        } else if seconds_saved < 86400.0 {
            println!("   Time saved: {:.1} hours", seconds_saved / 3600.0);
        } else {
            println!("   Time saved: {:.1} days", seconds_saved / 86400.0);
        }
    }

    // 3. CHECKPT / STATE
    let (start_linear_index, previous_total_checked) = if let Some((saved_index, saved_total, saved_hash)) = load_checkpoint(&config.checkpoint_file) {
        if saved_hash == config.target_hash {
            println!("📂 Resuming from checkpoint: {}", saved_index);
            (saved_index, saved_total)
        } else {
            println!("⚠️ Target changed. Starting fresh.");
            (0, 0)
        }
    } else {
        (0, 0)
    };

    // Shared State (apply length offset)
    let actual_start_index = start_linear_index + length_offset;
    let global_index = Arc::new(AtomicU64::new(actual_start_index));
    let found_flag = Arc::new(AtomicBool::new(false));
    let result_store = Arc::new(Mutex::new(None));

    // Pre-load PTX content once
    // Note: Ptx::from_file actually reads file content. We need to pass content or path to threads.
    // Ptx::from_file returns a Ptx struct. We can't clone it easily to threads if it holds pointers.
    // Best to read file locally and pass string content.
    // Wait, Ptx::from_file simply reads the file. Let's read it manually to be safe.
    let ptx_src = fs::read_to_string(&config.gpu_ptx_path)?;

    // 3. SPAWN WORKERS
    let mut handles = vec![];
    for dev_id in device_ids {
        let cfg = config.clone();
        let idx = global_index.clone();
        let f = found_flag.clone();
        let res = result_store.clone();
        let src = ptx_src.clone();
        
        let handle = thread::spawn(move || {
            if let Err(e) = worker_thread(dev_id, cfg, idx, f, res, src) {
                eprintln!("GPU {} Error: {}", dev_id, e);
            }
        });
        handles.push(handle);
    }

    println!("Engine started. {} workers active.", handles.len());

    // 4. MONITOR LOOP
    let start_time = Instant::now();
    let mut last_ckpt = Instant::now();
    
    // We need to track how much we've done for speed calc
    // global_index moves forward.
    
    loop {
        thread::sleep(Duration::from_millis(500));

        if found_flag.load(Ordering::Relaxed) {
             break;
        }

        let current = global_index.load(Ordering::Relaxed);
        let done_since_start = current - actual_start_index;
        let total = previous_total_checked + done_since_start;
        
        // Stats
        let elapsed = start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            let speed = done_since_start as f64 / elapsed / 1_000_000.0;
            print!("\rChecked: {:.1} M | Speed: {:.2} M/sec | Offset: {}   ",
                total as f64 / 1_000_000.0,
                speed,
                current
            );
            io::stdout().flush().ok();
        }

        // Checkpoint
        if last_ckpt.elapsed() >= Duration::from_secs(config.checkpoint_interval_secs) {
            let _ = save_checkpoint(&config.checkpoint_file, current, total, &config.target_hash);
            print!(" [💾]");
            io::stdout().flush().ok();
            last_ckpt = Instant::now();
        }
    }

    // 5. FINISH
    // Wait for threads (they should exit quickly after found_flag is true)
    for h in handles {
        let _ = h.join();
    }

    let guard = result_store.lock().unwrap();
    if let Some(winning_idx) = *guard {
        println!("\n\n!!! SUCCESS !!!");
        println!("Target Found at Random Index: {}", winning_idx);
        println!("(Use: python3 decode_result.py {} to get the password)", winning_idx);
        let _ = fs::remove_file(&config.checkpoint_file);
    } else {
        println!("\nStopped without finding target.");
    }

    Ok(())
}
