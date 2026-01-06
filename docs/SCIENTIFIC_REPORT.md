# Project ChaosWalker: A Stateless, Randomized Cryptographic Search Engine on GPU Architecture

**Date:** January 06, 2026  
**Author:** Jafar (IT Manager & Lead Engineer)  
**System Version:** 1.0 (Hybrid Rust/CUDA/Python Stack)

---

## Abstract

Project ChaosWalker presents a specialized cryptanalysis engine designed to explore the SHA-256 keyspace using massive parallel computing. Traditional brute-force tools are frequently constrained by I/O bottlenecks and sequential search latency, limiting their efficacy against complex passwords deep within the keyspace. ChaosWalker addresses these limitations through a novel Stateless "All-In-GPU" Architecture combined with a Chaotic Traversal Strategy. By migrating the entire candidate generation logic to the GPU and randomizing the search index via a cryptographic permutation, the system eliminates PCIe bus constraints and decouples password retrieval time from lexical complexity. Benchmarks on a single NVIDIA RTX 3090 workstation demonstrate a throughput exceeding 1.52 Billion Hashes per Second (1.52 GH/s) while ensuring uniform probability coverage across the 64-bit keyspace.

---

## 1. Introduction & Problem Statement

High-performance password recovery and cryptographic auditing face two fundamental limitations relative to modern hardware capabilities: the I/O Bottleneck and the Sequential Latency Problem.

### 1.1 The I/O Bottleneck
In traditional architectures, the host CPU acts as the primary generator of candidate passwords, which are subsequently transferred to the GPU for hashing. This model fails to saturate modern GPU interconnects. Even with high-bandwidth PCIe Gen 4.0 interfaces, the CPU cannot generate and transmit data fast enough to match the varying throughput of thousands of CUDA cores, leading to significant idle times on the computing device.

### 1.2 The Sequential Latency Problem
Standard brute-force tools scan the keyspace linearly (e.g., `aaaa`, `aaab`, `aaac`). This deterministic approach creates a "Worst Case" scenario for targets located deep in the lexicographical order (e.g., passwords starting with high-value ASCII characters or symbols). Consequently, complex passwords are only tested after exhaustive scanning of simpler candidates, resulting in potentially years of latency before distinct patterns are reached.

---

## 2. System Architecture

Project ChaosWalker introduces a "Zero-Traffic" architecture that fundamentally inverts the traditional control model.

### 2.1 The "Zero-Traffic" Paradigm
The CPU is relegated to the role of a state manager rather than a generator. Instead of transmitting gigabytes of candidate data, the Host sends a minimal control signal: a single 64-bit integer acting as the `Start_Index`.

*   **Data Transferred per Batch:** 8 Bytes
*   **Operations Triggered:** 10,000,000 (typical batch size)
*   **Bus Efficiency:** ~500,000x reduction in traffic compared to traditional methods.

This architecture ensures 100% GPU compute utilization by eliminating memory stalls associated with host-to-device data transfer.

### 2.2 Hybrid Implementation Stack
The system is implemented using a high-performance hybrid stack:
*   **Rust (Host):** Handles safe memory management, concurrency, and GPU driver automated orchestration.
*   **CUDA (Device):** Executes the compute kernels, including the Feistel network, Base-95 mapping, and SHA-256 hashing directly in hardware registers.
*   **Python:** Provides auxiliary analysis and visualization tools for result decoding and probability heatmaps.

---

## 3. Mathematical Methodology: The "Chaos" Paradigm

To address the Sequential Latency Problem, ChaosWalker employs a Non-Sequential, Chaotic Search Strategy. This is the engine's defining innovation, allowing it to traverse the keyspace stochastically while guaranteeing exhaustiveness.

### 3.1 The Feistel Network (Chaotic Traversal)
The engine utilizes a custom 4-round Feistel Network to map the linear search index to a randomized key space. For any given linear index $N$, the system applies a cryptographic bijection $P(N)$:

$$ F(\text{Index}) \leftrightarrow \text{Key}_{\text{Random}} $$

**Mechanism:**
1.  **Identity Calculation:** Each thread calculates a linear ID based on its block and thread index.
2.  **Permutation:** The linear ID is passed through a lightweight Feistel network using MurmurHash-style mixing functions.
3.  **Bijective Guarantee:** The mapping is strictly one-to-one, ensuring that every point in the 64-bit keyspace is visited exactly once, but in a pseudo-random order effectively indistinguishable from white noise.

### 3.2 In-Register Computation and Mapping
Following the permutation, the randomized 64-bit integer is converted into a plaintext candidate directly within the GPU registers:
*   **Base-95 Mapper:** The integer is mapped to a printable ASCII charset (byte range [32-126]) to generate the candidate string (e.g., `59201` $\rightarrow$ `"abc"`).
*   **Register-Based Hashing:** The candidate is hashed using a register-optimized SHA-256 implementation, avoiding global memory latency entirely.

---

## 4. Performance Evaluation

Performance benchmarks were conducted on a workstation equipped with a single NVIDIA RTX 3090.

### 4.1 Throughput Analysis
*   **Sustained Hash Rate:** 1.52 GH/s (Billion Hashes per Second).
*   **PCIe Utilization:** Near 0%. The bus remains idle during computation, allowing for trivial scalability to multi-GPU configurations without bandwidth saturation.

### 4.2 Latency Distribution
Unlike linear scanners where time-to-discovery is proportional to lexicographical position, ChaosWalker provides a flat probability distribution. A complex password (e.g., `~9z#P_nq`) has the same probability of being discovered in the first minute as a simple one (e.g., `aaaaaaab`). This significantly reduces the expected time-to-discovery for hard targets.

---

## 5. Conclusion

Project ChaosWalker demonstrates that moving the entire generation and traversal logic to the GPU is not only viable but superior to traditional host-generated approaches for high-speed cryptanalysis. By combining a "Zero-Traffic" architecture with a Feistel-based chaotic traversal, the system solves both the I/O bottleneck and the sequential latency problem, representing a significant advancement in the field of stateless cryptographic search engines.

---

## References

1.  **NIST FIPS 180-4**, "Secure Hash Standard (SHS)", March 2012.
2.  **Feistel, H.**, "Cryptography and Computer Privacy", Scientific American, Vol. 228, No. 5, 1973.
3.  **Luby, M., & Rackoff, C.**, "How to Construct Pseudorandom Permutations from Pseudorandom Functions", SIAM Journal on Computing, 1988.
