#include <stdint.h>
#include <cuda_runtime.h>

// --- SHA-256 CONSTANTS ---
__constant__ uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// SHA-256 Macros
#define ROTRIGHT(a,b) (((a) >> (b)) | ((a) << (32-(b))))
#define CH(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTRIGHT(x,2) ^ ROTRIGHT(x,13) ^ ROTRIGHT(x,22))
#define EP1(x) (ROTRIGHT(x,6) ^ ROTRIGHT(x,11) ^ ROTRIGHT(x,25))
#define SIG0(x) (ROTRIGHT(x,7) ^ ROTRIGHT(x,18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x,17) ^ ROTRIGHT(x,19) ^ ((x) >> 10))

// --- HELPER: GPU SHA256 TRANSFORM ---
__device__ void sha256_transform(uint32_t *state, const uint8_t *data) {
    uint32_t a, b, c, d, e, f, g, h, i, j, t1, t2, m[64];

    for (i = 0, j = 0; i < 16; ++i, j += 4)
        m[i] = (data[j] << 24) | (data[j + 1] << 16) | (data[j + 2] << 8) | (data[j + 3]);
    for (; i < 64; ++i)
        m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];

    a = state[0]; b = state[1]; c = state[2]; d = state[3];
    e = state[4]; f = state[5]; g = state[6]; h = state[7];

    for (i = 0; i < 64; ++i) {
        t1 = h + EP1(e) + CH(e, f, g) + K[i] + m[i];
        t2 = EP0(a) + MAJ(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

// --- CORE: FEISTEL ROUND FUNCTION (Simplified for GPU Speed) ---
// Uses a MurmurHash-style mix instead of full SHA256 to save registers
__device__ uint32_t feistel_round(uint32_t val, int round_num, uint32_t key_seed) {
    uint32_t k = val * 0xcc9e2d51;
    k = (k << 15) | (k >> 17);
    k *= 0x1b873593;
    uint32_t h = key_seed ^ k;
    h = (h << 13) | (h >> 19);
    h = h * 5 + 0xe6546b64 + round_num;
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

// --- MAIN KERNEL ---
extern "C" __global__ void crack_kernel(
    uint64_t start_index,   // The base index for this batch
    uint64_t* found_idx, 
    const uint32_t* target_hash,
    int num_items
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_items) return;

    // 1. CALCULATE MY LINEAR ID
    uint64_t my_linear_id = start_index + idx;

    // 2. FEISTEL ENCRYPT (Linear ID -> Random Unique ID)
    // 4 Rounds of Feistel Network
    uint32_t left = (uint32_t)(my_linear_id >> 32);
    uint32_t right = (uint32_t)(my_linear_id & 0xFFFFFFFF);
    uint32_t key_seed = 0xDEADBEEF; // Simple static seed for speed

    #pragma unroll
    for(int r=0; r<4; r++) {
        uint32_t temp = right;
        right = left ^ feistel_round(right, r, key_seed);
        left = temp;
    }
    uint64_t random_point = ((uint64_t)left << 32) | right;

    // 3. MAPPER: BASE-95 CONVERSION (Random ID -> Password)
    uint8_t data[64];
    #pragma unroll
    for(int k=0; k<64; k++) data[k] = 0;

    uint64_t temp = random_point;
    int len = 0;
    uint8_t temp_pass[16]; 
    
    if (temp == 0) {
        temp_pass[0] = 32; len = 1;
    } else {
        while (temp > 0 && len < 16) {
            temp_pass[len++] = (temp % 95) + 32; 
            temp /= 95;
        }
    }

    // Copy & Pad (Standard SHA256 Padding)
    for(int i=0; i<len; i++) data[i] = temp_pass[i];
    data[len] = 0x80;
    uint64_t bit_len = len * 8;
    data[63] = bit_len & 0xFF;
    data[62] = (bit_len >> 8) & 0xFF;

    // 4. HASHER
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    sha256_transform(state, data);

    // 5. CHECKER
    if (state[0] == target_hash[0] && state[1] == target_hash[1] && 
        state[2] == target_hash[2] && state[3] == target_hash[3] &&
        state[4] == target_hash[4] && state[5] == target_hash[5] &&
        state[6] == target_hash[6] && state[7] == target_hash[7]) 
    {
        *found_idx = random_point;
    }
}