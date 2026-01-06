// Length-Filtered Kernel for ChaosWalker
// This kernel only checks passwords of a specific length

#define REQUIRED_LENGTH 8  // Set this to your known password length

// In the main kernel, after generating the password:
extern "C" __global__ void crack_kernel_length_filtered(
    uint64_t start_index,
    uint64_t* found_idx,
    const uint32_t* target_hash,
    int num_items
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_items) return;

    uint64_t my_linear_id = start_index + idx;
    uint64_t password_index = my_linear_id;  // or feistel_encrypt(my_linear_id) for random

    // Generate password (same as before)
    uint64_t temp = password_index;
    int len = 0;
    uint8_t temp_pass[16];

    if (temp == 0) {
        temp_pass[0] = SMART_CHARSET[0];
        len = 1;
    } else {
        while (temp > 0 && len < 16) {
            temp_pass[len++] = SMART_CHARSET[temp % 95];
            temp /= 95;
        }
    }

    // LENGTH FILTER: Skip if not the required length
    if (len != REQUIRED_LENGTH) {
        return;  // Skip this password, don't hash it
    }

    // Continue with SHA-256 and checking (only for correct length)
    // ... rest of kernel code ...
}
