use std::process::Command;

fn main() {
    // 1. Tell Cargo to look for CUDA libraries in standard places
    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cudart");

    // 2. Automatically compile the CUDA Kernels
    // Compile LINEAR mode kernel (fast, Smart Mapper optimized)
    println!("cargo:rerun-if-changed=kernels/chaos_worker_linear.cu");
    let status_linear = Command::new("nvcc")
        .args(&[
            "-ptx", 
            "kernels/chaos_worker_linear.cu", 
            "-o", 
            "kernels/chaos_worker_linear.ptx", 
            "-arch=sm_86" // Target RTX 3090
        ])
        .status()
        .expect("Failed to run nvcc. Is CUDA Toolkit installed?");

    if !status_linear.success() {
        panic!("NVCC compilation failed for linear kernel!");
    }

    // Compile RANDOM mode kernel (exhaustive, Feistel-based)
    println!("cargo:rerun-if-changed=kernels/chaos_worker_random.cu");
    let status_random = Command::new("nvcc")
        .args(&[
            "-ptx", 
            "kernels/chaos_worker_random.cu", 
            "-o", 
            "kernels/chaos_worker_random.ptx", 
            "-arch=sm_86" // Target RTX 3090
        ])
        .status()
        .expect("Failed to run nvcc. Is CUDA Toolkit installed?");

    if !status_random.success() {
        panic!("NVCC compilation failed for random kernel!");
    }

    println!("cargo:warning=Successfully compiled both LINEAR and RANDOM kernels");
}