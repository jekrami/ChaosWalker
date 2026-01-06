use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use std::convert::TryInto;
use std::time::Instant;

// --- CONFIGURATION ---
const BATCH_SIZE: usize = 10_000_000; // Keys to check per GPU cycle
const GPU_PTX_PATH: &str = "./kernels/chaos_worker.ptx";

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
    // Example: SHA256("testpass") -> 17912ee2...
    let target_hex = "17912ee268297e742817c187b5a1b3240247657954930379462509d37575209f";
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

    println!("Target loaded. Engine started.");
    println!("Batch Size: {} keys/cycle", BATCH_SIZE);

    let start_time = Instant::now();
    let mut total_checked: u64 = 0;
    
    // We track the linear progress (0, 1M, 2M...), but the GPU 
    // transforms these into random hops internally.
    let mut current_linear_index: u64 = 0;

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
            println!("(Use a Python script to convert this index back to the password string)");
            break;
        }

        // Progress
        total_checked += BATCH_SIZE as u64;
        current_linear_index += BATCH_SIZE as u64;

        if total_checked % (BATCH_SIZE as u64 * 50) == 0 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let speed = total_checked as f64 / elapsed / 1_000_000.0;
            print!("\rChecked: {:.1} M | Speed: {:.2} M/sec | Offset: {}", 
                total_checked as f64 / 1_000_000.0, 
                speed,
                current_linear_index
            );
        }
    }

    Ok(())
}