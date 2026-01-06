use std::process::Command;

fn main() {
    // 1. Tell Cargo to look for CUDA libraries in standard places
    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cudart");

    // 2. Automatically compile the CUDA Kernel
    // This runs 'nvcc' every time you change 'chaos_worker.cu'
    println!("cargo:rerun-if-changed=kernels/chaos_worker.cu");

    let status = Command::new("nvcc")
        .args(&[
            "-ptx", 
            "kernels/chaos_worker.cu", 
            "-o", 
            "kernels/chaos_worker.ptx", 
            "-arch=sm_86" // Target RTX 3090
        ])
        .status()
        .expect("Failed to run nvcc. Is CUDA Toolkit installed?");

    if !status.success() {
        panic!("NVCC compilation failed!");
    }
}