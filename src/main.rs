use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use std::convert::TryInto;
use std::time::{Instant, Duration};
use std::fs;
use std::io::{self, Write};
use std::path::Path;

// --- CONFIGURATION ---
const BATCH_SIZE: usize = 10_000_000; // Keys to check per GPU cycle
const GPU_PTX_PATH: &str = "./kernels/chaos_worker.ptx";
const CHECKPOINT_FILE: &str = "chaos_state.txt";
const CHECKPOINT_INTERVAL_SECS: u64 = 30; // Save every 30 seconds

// --- CHECKPOINT SYSTEM ---

/// Save the current search state to disk
fn save_checkpoint(current_index: u64, total_checked: u64, target_hex: &str) -> io::Result<()> {
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
    let temp_file = format!("{}.tmp", CHECKPOINT_FILE);
    fs::write(&temp_file, checkpoint_data)?;
    fs::rename(&temp_file, CHECKPOINT_FILE)?;

    Ok(())
}

/// Load the checkpoint from disk, if it exists
fn load_checkpoint() -> Option<(u64, u64, String)> {
    if !Path::new(CHECKPOINT_FILE).exists() {
        return None;
    }

    let content = match fs::read_to_string(CHECKPOINT_FILE) {
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

fn main() -> Result<(), anyhow::Error> {
    println!("--- Project ChaosWalker: Initiating High-Performance Mode ---");

    // 1. CONNECT TO GPU (Using CudaContext for cudarc 0.18+)
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    println!("GPU Connected: RTX 3090 active.");

    // 2. LOAD KERNEL
    // We load the compiled PTX file.
    // Ensure your build.rs has compiled the NEW C++ code before running this!
    let ptx = Ptx::from_file(GPU_PTX_PATH);
    let module = ctx.load_module(ptx)?;
    let kernel = module.load_function("crack_kernel")?;

    // 3. DEFINE TARGET
    // ChaosWalker v1.0 with Smart Mapper
    // Test password: "VDKdrAQ5" (will be found at ~119K linear index)
    let target_hex = "c30c9a521a08ba8613d80d866ed07f91d347ceb1c2dafe5f358ef9244918b3d4";
    let target_bytes = hex::decode(target_hex)?;

    // Convert target to u32 array (Big Endian)
    let mut target_u32 = [0u32; 8];
    for i in 0..8 {
        let chunk = &target_bytes[i*4..(i+1)*4];
        target_u32[i] = u32::from_be_bytes(chunk.try_into()?);
    }

    // 4. ALLOCATE MEMORY ON GPU
    // We only need buffers for the Target (Input) and the Result (Output).
    // We DO NOT need a buffer for the indices anymore (saving massive bandwidth).
    let mut dev_found = stream.alloc_zeros::<u64>(1)?;
    let dev_target = stream.clone_htod(&target_u32)?;

    // 5. CHECK FOR CHECKPOINT (Resume from previous run)
    let (mut current_linear_index, mut total_checked) = if let Some((saved_index, saved_total, saved_hash)) = load_checkpoint() {
        if saved_hash == target_hex {
            println!();
            println!("📂 CHECKPOINT FOUND!");
            println!("   Resuming from: {}", saved_index);
            println!("   Already checked: {} passwords", saved_total);
            println!();
            (saved_index, saved_total)
        } else {
            println!();
            println!("⚠️  Checkpoint found but target hash changed. Starting fresh.");
            println!();
            (0, 0)
        }
    } else {
        println!();
        println!("🆕 No checkpoint found. Starting from beginning.");
        println!();
        (0, 0)
    };

    println!("Target loaded. Engine started.");
    println!("Batch Size: {} keys/cycle", BATCH_SIZE);
    println!("Checkpoint: Saving every {} seconds to {}", CHECKPOINT_INTERVAL_SECS, CHECKPOINT_FILE);
    println!();

    let start_time = Instant::now();
    let mut last_checkpoint_time = Instant::now();

    let cfg = LaunchConfig::for_num_elems(BATCH_SIZE as u32);

    // --- THE ATTACK LOOP ---
    loop {
        // Launch the Kernel
        // The GPU calculates the Feistel permutation itself now.
        unsafe {
            stream.launch_builder(&kernel)
                .arg(&current_linear_index)    // Arg 1: Start Index (Scalar)
                .arg(&mut dev_found)           // Arg 2: Found Flag (Pointer)
                .arg(&dev_target)              // Arg 3: Target Hash (Pointer)
                .arg(&(BATCH_SIZE as i32))     // Arg 4: Count (Scalar)
                .launch(cfg)?;
        }

        // Check Results (Async copy back)
        // We only read back 8 bytes per million keys! Very fast.
        let found_val = stream.clone_dtoh(&dev_found)?;
        
        if found_val[0] != 0 {
            let winning_random_index = found_val[0];
            println!("\n\n!!! SUCCESS !!!");
            println!("Target Found at Random Index: {}", winning_random_index);
            println!("(Use: python3 decode_result.py {} to get the password)", winning_random_index);

            // Delete checkpoint on success
            let _ = fs::remove_file(CHECKPOINT_FILE);
            println!("\n✅ Checkpoint deleted (search complete)");
            break;
        }

        // Progress
        total_checked += BATCH_SIZE as u64;
        current_linear_index += BATCH_SIZE as u64;

        // Auto-save checkpoint every N seconds
        if last_checkpoint_time.elapsed() >= Duration::from_secs(CHECKPOINT_INTERVAL_SECS) {
            if let Err(e) = save_checkpoint(current_linear_index, total_checked, target_hex) {
                eprintln!("\n⚠️  Warning: Failed to save checkpoint: {}", e);
            } else {
                print!(" [💾 Saved]");
                io::stdout().flush().ok();
            }
            last_checkpoint_time = Instant::now();
        }

        // Progress display
        if total_checked % (BATCH_SIZE as u64 * 50) == 0 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let speed = total_checked as f64 / elapsed / 1_000_000.0;
            print!("\rChecked: {:.1} M | Speed: {:.2} M/sec | Offset: {}",
                total_checked as f64 / 1_000_000.0,
                speed,
                current_linear_index
            );
            io::stdout().flush().ok();
        }
    }

    Ok(())
}