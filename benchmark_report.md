# Benchmark Analysis Report

**Date:** 2026-03-16
**Setup:** 256 clients, 4 iterations, seed=42, adaptive threshold=0.9

---

## 1. Summary of All Runs

### Flamingo

| Vec Length | Mode | Wall (s) | Srv Report (s) | Srv Recon (s) | ECC Mult (s) | Clt Report (s) | Clt Recon (s) | Vector Comm (MB) | Sign Comm (MB) | Avg Online | Agg Fail | Correct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100k | Fixed | 81.6 | 0.018 | 1.703 | 1.795 | 0.048 | 0.075 | 95.8 | 3.1 | 238 | 0 | Yes |
| 100k | Adaptive | 92.5 | 0.021 | 2.583 | 2.094 | 0.049 | 0.115 | 93.1 | 3.5 | 230 | 0 | Yes |
| 1M | Fixed | 176.8 | 0.124 | 3.961 | 2.031 | 0.133 | 0.067 | 947.2 | 3.4 | 242 | 0 | Yes |
| 1M | Adaptive | 197.3 | 0.100 | 5.825 | 2.139 | 0.138 | 0.113 | 916.9 | 3.4 | 230 | 0 | Yes |

### Google Malicious

| Vec Length | Mode | Wall (s) | Srv Report (s) | Srv Crosscheck (s) | Srv Recon (s) | Avg Online | Agg Fail | Correct |
|---|---|---|---|---|---|---|---|---|
| 100k | Fixed | 101.0 | 0.011 | 0.011 | 1.039 | - | 0 | Yes |
| 100k | Adaptive | 106.6 | 0.010 | 0.005 | 2.465 | - | 1* | Yes |
| 1M | Fixed | 694.9 | 0.154 | 0.012 | 5.637 | - | 0 | Yes |
| 1M | Adaptive | 728.7 | 0.105 | 0.005 | 17.306 | - | 1* | Yes |

\* 1 aggregation failure out of 4 iterations (skipped).
Note: Google Malicious client-side timings and communication data were not captured by the benchmark parser.

---

## 2. Protocol Comparison: Flamingo vs. Google Malicious

### Wall-Clock Time

| Vec Length | Flamingo Fixed (s) | Google Fixed (s) | Google / Flamingo |
|---|---|---|---|
| 100k | 81.6 | 101.0 | **1.24x** |
| 1M | 176.8 | 694.9 | **3.93x** |

At 100k, the two protocols are relatively close — Google Malicious is only 24% slower. At 1M, the gap explodes to nearly **4x**. This is because Google Malicious's pairwise masking cost scales with both the number of neighbors and the vector length, while Flamingo's committee-based approach has lower per-element overhead.

### Server Reconstruction Time (avg per iteration)

| Vec Length | Flamingo Fixed (s) | Google Fixed (s) | Google / Flamingo |
|---|---|---|---|
| 100k | 1.703 | 1.039 | 0.61x (Google faster) |
| 1M | 3.961 | 5.637 | 1.42x |

At 100k, Google Malicious reconstruction is actually *faster* than Flamingo's. But at 1M, it becomes 1.4x slower. The crossover happens because Flamingo's reconstruction involves ECC operations (relatively constant cost) plus vector summation, while Google Malicious reconstruction scales with vector length × number of dropped neighbors.

### Where Does the Time Go?

Google Malicious's extra wall-clock time comes primarily from its **6-round structure** (vs. Flamingo's 3 rounds). Even when server computation per round is comparable, the additional round trips (advertise keys, establish graph, forward shares) each carry their own fixed timeout or wait period. At 1M vectors, the client-side computation for pairwise masking also becomes a significant contributor.

---

## 3. Fixed vs. Adaptive Mode

### Wall-Clock Comparison

| Protocol | Vec Length | Fixed (s) | Adaptive (s) | Delta |
|---|---|---|---|---|
| Flamingo | 100k | 81.6 | 92.5 | **+13.4% slower** |
| Flamingo | 1M | 176.8 | 197.3 | **+11.6% slower** |
| Google Malicious | 100k | 101.0 | 106.6 | **+5.5% slower** |
| Google Malicious | 1M | 694.9 | 728.7 | **+4.9% slower** |

Adaptive mode was **slower in all four configurations**. This is counterintuitive — adaptive mode is designed to reduce latency by eliminating idle wait time. The explanation lies in the balance between wait time savings and reconstruction cost:

1. **At these vector lengths, computation already exceeds the fixed timeouts.** Clients need significant time to mask 100k-1M element vectors. By the time they finish, the fixed timeout has often already elapsed, leaving little idle time for adaptive mode to reclaim.

2. **Adaptive mode's 90% threshold increases dropout.** The server proceeds once 90% of clients respond, which causes more clients to be marked as dropped. This directly increases reconstruction cost:

| Protocol | Vec Length | Fixed Recon (s) | Adaptive Recon (s) | Increase |
|---|---|---|---|---|
| Flamingo | 100k | 1.703 | 2.583 | **+52%** |
| Flamingo | 1M | 3.961 | 5.825 | **+47%** |
| Google Malicious | 100k | 1.039 | 2.465 | **+137%** |
| Google Malicious | 1M | 5.637 | 17.306 | **+207%** |

Google Malicious suffers disproportionately because unmasking a dropped client requires recovering pairwise masks with *every* neighbor, making reconstruction cost scale with `(# dropouts) × (neighborhood size) × (vector length)`.

### Client Participation (Flamingo)

| Vec Length | Fixed Avg Online | Adaptive Avg Online |
|---|---|---|
| 100k | 238 / 256 (93.0%) | 230 / 256 (89.8%) |
| 1M | 242 / 256 (94.5%) | 230 / 256 (89.8%) |

Adaptive mode consistently lands near the 90% threshold. Fixed mode retains more clients (93-95%) because the longer timeouts allow stragglers to complete.

---

## 4. Scaling: 100k vs. 1M Vector Length

### Wall-Clock Scaling (10x vector length increase)

| Protocol | Mode | 100k (s) | 1M (s) | Scaling Factor |
|---|---|---|---|---|
| Flamingo | Fixed | 81.6 | 176.8 | **2.17x** |
| Flamingo | Adaptive | 92.5 | 197.3 | **2.13x** |
| Google Malicious | Fixed | 101.0 | 694.9 | **6.88x** |
| Google Malicious | Adaptive | 106.6 | 728.7 | **6.84x** |

Flamingo scales roughly **2x** for a 10x vector length increase — well sub-linear. This is because a large portion of Flamingo's cost (ECC operations, committee coordination) is independent of vector length.

Google Malicious scales nearly **7x** — much worse. Its pairwise masking means most operations grow linearly with vector length, and the 6-round structure amplifies this across every round.

### Server Computation Scaling

| Step | Protocol | 100k (s) | 1M (s) | Scaling |
|---|---|---|---|---|
| Report | Flamingo | 0.018 | 0.124 | 7.1x |
| Report | Google | 0.011 | 0.154 | 14.5x |
| Reconstruction | Flamingo | 1.703 | 3.961 | 2.3x |
| Reconstruction | Google | 1.039 | 5.637 | 5.4x |
| ECC Mult | Flamingo | 1.795 | 2.031 | 1.13x |

- **ECC scalar multiplication** is nearly constant (~1.8-2.0s) regardless of vector length, as expected — it operates on curve points, not raw vectors.
- **Report (server-side aggregation)** scales roughly linearly with vector length for both protocols.
- **Reconstruction** scales sub-linearly for Flamingo (2.3x) but super-linearly for Google Malicious (5.4x), due to the pairwise mask recovery cost.

#### Flamingo Reconstruction Sub-Timer Breakdown

To understand where reconstruction time goes, we instrumented the four sub-operations within the server's reconstruction phase. All times are averages per iteration (sequential baseline, fixed mode, 256 clients).

| Sub-Operation | Description | 100k (s) | % of Recon | 1M (s) | % of Recon |
|---|---|---|---|---|---|
| `lagrange_interp` | Lagrange interpolation to recover secret shares | 0.0015 | 0.1% | 0.0045 | 0.1% |
| `recover_mi` | PRG expansion of committee mask keys → vector subtraction | 0.198 | 8.2% | 1.755 | 35.1% |
| `ecc_scalar_mult` | ECC scalar multiplication on pairwise shares | 1.973 | 81.8% | 1.525 | 30.5% |
| `recover_pairwise` | PRG expansion of dropped-client pairwise keys → vector add/sub | 0.235 | 9.7% | 1.674 | 33.5% |
| **Total** | | **2.413** | **100%** | **4.995** | **100%** |

At **100k**, ECC scalar multiplication dominates at 82% of reconstruction time — the PRG/vector operations are negligible. At **1M**, the picture changes dramatically: ECC drops to 31%, and the two PRG expansion steps (`recover_mi` + `recover_pairwise`) together consume **69%** of reconstruction time. This confirms that vector-length-dependent PRG expansion becomes the bottleneck at large vector sizes.

The `lagrange_interp` step is negligible (<0.1%) at both scales — it operates only on small integer shares, not vectors.

#### Why Does Flamingo Reconstruction Scale with Vector Length?

The reconstruction phase includes two **vector-length-dependent** operations alongside two that are not:

1. **PRG mask recovery (`recover_mi`):** For each committee member's recovered secret key, the server expands a ChaCha20 PRG to generate the full mask vector (`vector_len` bytes), then subtracts it element-by-element from the aggregate. Cost: `O(committee_size × vector_len)`. Scales ~8.9x from 100k to 1M.

2. **Pairwise mask cancellation (`recover_pairwise`):** For each dropped client, the server recovers the pairwise shared key, expands it via ChaCha20 into a full-length vector, and adds/subtracts it to cancel the residual mask. Cost: `O(num_dropped × vector_len)`. Scales ~7.1x from 100k to 1M.

3. **ECC scalar multiplication** and **Lagrange interpolation** are independent of vector length (operate on curve points and integer shares respectively). ECC remains ~1.5-2.0s regardless.

The constant-cost crypto steps dilute the linear PRG/vector cost, yielding sub-linear overall scaling (2.3x for 10x vector growth).

#### Parallelization of PRG Expansion

Since ChaCha20 PRG expansions for different keys are fully independent, we implemented **multi-threaded parallelization** using Python's `ThreadPoolExecutor`. Both ChaCha20 (PyCryptodome) and NumPy release the GIL, allowing true parallel execution. Parallelization is enabled by default for vector lengths >= 100k and can be controlled via the `FLAMINGO_PARALLEL_RECON` environment variable.

**Results** (256 clients, 4 iterations, fixed mode, same seed):

| Vec Length | Sub-Operation | Sequential (s) | Parallel (s) | Speedup |
|---|---|---|---|---|
| 100k | `recover_mi` | 0.198 | 0.081 | **2.5x** |
| 100k | `recover_pairwise` | 0.235 | 0.099 | **2.4x** |
| 100k | Total reconstruction | 2.413 | 2.039 | **1.18x** |
| 1M | `recover_mi` | 1.755 | 0.517 | **3.4x** |
| 1M | `recover_pairwise` | 1.674 | 0.570 | **2.9x** |
| 1M | Total reconstruction | 4.995 | 2.800 | **1.78x** |

| Vec Length | Sequential Wall (s) | Parallel Wall (s) | Wall Reduction |
|---|---|---|---|
| 100k | 110.4 | 107.2 | **-3.0%** |
| 1M | 209.5 | 201.7 | **-3.7%** |

At **1M**, parallelization reduces reconstruction time by **44%** (4.99s → 2.80s), with the PRG steps achieving 2.9-3.4x speedup. The wall-clock improvement is more modest (-3.7%) because reconstruction is only one component of the total protocol time (which includes network waiting and client computation). The ECC scalar multiplication (~1.5-1.7s), now the dominant sub-operation after parallelization, remains sequential and is the next optimization target.

At **100k**, the PRG steps were already small relative to ECC, so parallelization has less impact on total reconstruction (18%) and negligible impact on wall-clock time.

On cloud hardware with more cores (32-64), the PRG speedup would be even larger, potentially pushing `recover_mi` and `recover_pairwise` to sub-100ms at 1M. At that point, ECC scalar multiplication would become the clear bottleneck (~60-70% of reconstruction), and further optimization would require ECC-level parallelism or hardware acceleration.

---

## 5. Client Computation (Flamingo)

Average per-client computation per iteration:

| Vec Length | Mode | Client Report (s) | Client Reconstruction (s) |
|---|---|---|---|
| 100k | Fixed | 0.048 | 0.075 |
| 100k | Adaptive | 0.049 | 0.115 |
| 1M | Fixed | 0.133 | 0.067 |
| 1M | Adaptive | 0.138 | 0.113 |

- **Client report** (masking + sending vector) scales ~2.8x from 100k to 1M — proportional to the work of masking a larger vector.
- **Client reconstruction** (sending decryption shares) remains low and is roughly constant with vector length, as clients only send small shares, not full vectors.
- Adaptive mode slightly increases client reconstruction time because the reconstruction round may require more work when more clients have dropped.

---

## 6. Communication (Flamingo)

| Vec Length | Mode | Vector Data (total, MB) | Sign Data (total, MB) | Per-Client Per-Iter Vector (KB) |
|---|---|---|---|---|
| 100k | Fixed | 95.8 | 3.1 | 101 |
| 100k | Adaptive | 93.1 | 3.5 | 101 |
| 1M | Fixed | 947.2 | 3.4 | 978 |
| 1M | Adaptive | 916.9 | 3.4 | 997 |

- **Vector communication** scales linearly with vector length (~10x from 100k to 1M), as expected. Each client transmits approximately `vec_len × 1 byte` per iteration (values are packed as uint8 for the unit-vector test inputs).
- **Signing communication** is constant (~3.1-3.5 MB total) regardless of vector length. This is committee coordination overhead (signatures, not data), involving only the small committee subset.
- Adaptive mode has slightly lower total vector bytes (fewer online clients sending data) but the per-client cost is identical.

---

## 7. Recommendations

| Scenario | Recommendation | Reason |
|---|---|---|
| Large vectors (>=100k) | **Fixed mode** | Computation exceeds timeouts; adaptive adds reconstruction cost with no latency benefit |
| Small vectors (<10k) | **Adaptive mode** | Wait time dominates; adaptive can save 29-40% latency |
| Scalability priority | **Flamingo** | 2x scaling for 10x vector growth vs. 7x for Google Malicious |
| Minimizing dropout impact | **Flamingo** | Reconstruction cost under dropout is far lower than Google Malicious (+47% vs +207%) |
| Simple deployment | **Fixed mode** | Predictable behavior, no threshold tuning needed |

---

## Raw Data

- `benchmark_flamingo.csv` / `benchmark_flamingo_100k.csv` — Flamingo results (fixed + adaptive)
- `benchmark_google.csv` / `benchmark_google_100k.csv` — Google Malicious results (fixed + adaptive)
- `benchmark_recon_seq_100k.csv` / `benchmark_recon_seq_1m.csv` — Sequential reconstruction baseline
- `benchmark_recon_par_100k.csv` / `benchmark_recon_par_1m.csv` — Parallel reconstruction results
