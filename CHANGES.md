# Changes

This document lists the changes made to the Flamingo codebase to make it run on modern macOS (arm64), fix bugs, optimize performance, and add benchmarking.

## Compatibility Fixes

### requirements.txt
- Unpinned old version constraints (`kiwisolver==1.1.0`, `seaborn==0.9.0`, `pyparsing==2.4.0`, `python-dateutil==2.8.0`, `pytz==2019.1`, `tqdm==4.36.1`, `six==1.12.0`, `joblib==1.2.0`) that failed to compile on macOS arm64 (Apple Silicon).

### agent/flamingo/SA_ServiceAgent.py
- Replaced deprecated `DataFrame.applymap()` with `DataFrame.map()` (pandas 2.x compatibility).

### config/flamingo.py
- Fixed `argparse type=bool` for `-p` and `-d` flags. `bool('0')` is `True` in Python, so `type=bool` doesn't work as a CLI toggle. Changed to `type=int`.

### agent/google_malicious/SA_ServiceAgent.py & SA_ClientAgent.py
- Fixed `recordTime()`: `.to_timedelta64()` returns a type that `int()` can't convert on newer pandas. Replaced with `int(elapsed.total_seconds() * 1e9)`.

### agent/dkg/SA_ClientAgent.py
- Fixed `KeyError` in sk_share computation: not all clients' shares may be present due to dropout. Added `if id in self.recv_shares` guard.

---

## Performance Optimizations

### 1. PKI Key Cache (`util/util.py`)
- Added global `_key_cache` dict. `read_key()` now caches ECC keys on first read, avoiding repeated disk I/O. With 128 clients and ~14 neighbors each, this eliminates ~1,800 file reads per iteration after the first.

### 2. Pairwise DH Secret Cache (`agent/flamingo/SA_ClientAgent.py`)
- Added `self.pairwise_dh_cache` dict keyed by neighbor ID. The DH secret `sk_i * pk_j` is fixed across iterations; only the round-dependent `h_{i,j,t}` changes. This saves one EC scalar multiplication (~1ms) per cached neighbor per iteration.
- **Measured impact:** Iteration 2 report step dropped from ~0.40s to ~0.18s (2x faster).

### 3. Neighbor Graph Cache (`util/param.py`)
- Added `compute_full_graph()` that computes the bidirectional neighbor graph for all clients in one pass, cached per (iteration, num_clients, neighborhood_size).
- `findNeighbors()` now does a simple dict lookup from the cached graph.
- Previously: each of n clients independently generated the full PRG output and scanned all n segments. Total work: O(n^2 * log n).
- Now: one PRG expansion + one O(n * log n) pass, shared by all clients.
- Note: in a real deployment, each client computes its own neighbors locally. This is a simulation optimization.

### 4. Removed multiprocessing from reconstruction (`agent/flamingo/SA_ServiceAgent.py`)
- Benchmarked sequential vs. `multiprocessing.Pool` for EC scalar multiplication:
  - 1,400 mults: sequential 0.34s vs multiprocessing 1.93s (sequential is 5.7x faster)
  - 10,000 mults: sequential 2.49s vs multiprocessing 2.42s (barely breaks even)
- For the typical Flamingo workload (~1,400 EC mults), fork/serialization overhead dominates.
- Removed `multiprocessing` import, `parallel_mult()` function, and the parallel code path.
- **Measured impact:** Reconstruction dropped from ~2.0s to ~0.38s (5x faster).

### 5. Optimized hash-to-curve (`util/crypto/ecchash.py`)
- Replaced `libnum.invmod()` with Python 3.8+ builtin `pow(x, -1, p)` (C-backed).
- Replaced `libnum.has_sqrtmod()` / `libnum.sqrtmod()` with Euler's criterion and `pow(x, (p+1)//4, p)`. P-256's prime satisfies p ≡ 3 mod 4, enabling this fast path.
- Precomputed constants (`_p256_inv_a`, `_p256_inv_30`, `_p256_sqrt_exp`) to avoid redundant work.
- Fixed `sgn0()` to return `x % 2` (parity), matching the hash-to-curve RFC, instead of the previous incorrect implementation that returned 1 for all non-positive values.
- Used `p - y` instead of `-y` for point negation to stay in the positive field.
- **Measured impact:** map_to_curve: 0.18ms vs 0.40ms per call (2.2x faster). All 100 test points verified on P-256 curve.

### 6. Binary serialization (`util/util.py`)
- Replaced JSON-based serialization of EC points and ElGamal ciphertexts with binary `struct.pack` format.
- EC point: 64 bytes (32B x-coordinate + 32B y-coordinate) vs ~160 chars in JSON.
- ElGamal ciphertext: 136 bytes per entry (8B key + 2×64B points) vs ~320+ chars in JSON.
- Tuple-bytes (AES ciphertexts): direct binary concatenation with length headers.
- **Approximate size reduction:** ~3-5x smaller messages.

---

## Bug Fixes

### Duplicate dict key (`agent/flamingo/SA_ClientAgent.py`)
- Removed duplicate `"signed_labels"` key in SIGN message dict literal (line 372). Both values were identical so behavior was correct, but the duplicate was confusing.

### time.sleep in discrete-event simulator (`agent/flamingo/SA_ServiceAgent.py`)
- Replaced `time.sleep(0.1)` in `reconstruction_read_from_pool()` with a simulation-time retry mechanism. If not enough decryption shares have arrived, the server schedules another wakeup (up to 3 retries) via the simulation's event queue, rather than blocking the wall clock.

---

## Logging & Benchmarking

### Unified logging (`agent/flamingo/SA_ClientAgent.py`, `SA_ServiceAgent.py`)
- Fixed `agent_print()` in both client and server: added `self` parameter and actual prefix (`[Server]`, `[Client N]`). Previously `*args` swallowed `self` and the prefix was not applied.
- Replaced `logging.basicConfig()` with per-agent named loggers (`flamingo.server`, `flamingo.client.N`) with proper formatters.
- Changed `if __debug__: self.logger.info(...)` patterns to `self.logger.debug(...)` (controlled by debug_mode flag).

### Benchmark system (`util/benchmark.py`)
- New `BenchmarkCollector` class with a global `bench` singleton shared across all agents.
- Tracks:
  - **Per-step server computation time** with sub-operation breakdown (vector_sum, find_dec_targets, sign_labels, lagrange_interp, ecc_scalar_mult, recover_mi, recover_pairwise).
  - **Per-client computation time** (avg/max across all clients for report, crosscheck, reconstruction).
  - **Communication size** in bytes (client→server VECTOR messages, server→committee SIGN messages).
  - **Per-iteration metrics** (online/offline clients, dec_targets, decryptor shares).
  - **Cross-iteration averages** for all of the above.
- Output: structured summary printed to stdout at the end of simulation.

---


## Server-Driven First Round (Flamingo)

Changed Flamingo's first report round from client-initiated to server-initiated, making it consistent with iterations 2+.

**Before:** Clients sent their report vectors upon `wakeup()` (self-initiated). This meant iteration 1's report round was one-way (client → server), while all other rounds were round-trip (server → client → server).

**After:** The server sends a `START_REPORT` message to all clients at the end of `initialize()`. Clients wait for this message before sending their report vectors. All rounds are now uniformly round-trip.

**Why:** This ensures fair comparison with Google Malicious (which is also server-driven) and makes wait time measurements consistent across iterations.

## Stale Wakeup Detection (both protocols)

Both protocols now use `_expected_wakeup_time` to detect and skip stale wakeups. When a round advances, the old safety timeout becomes stale (it fires at an earlier time than the new round's expected wakeup). The check `currentTime < _expected_wakeup_time` reliably identifies these.

---

## Files Changed

---

## README Updates

- **Installation Instructions**: Replaced Linux-only Miniconda wget/bash workflow with a simple `pip install` option; kept Conda as Option B.
- **Command-Line Options**: Expanded from 6 options to include `-w`, `--wait_threshold`, `-s`, `-e` which were previously undocumented. Noted minimum 128 clients.
- **Debug flag**: Fixed example from `-d True` to `-d 1` (it's an `int` argument, not `bool`).
- **New section: "Server Waiting Modes: Fixed vs. Adaptive"**: Added per-round tables describing the adaptive trigger condition for each round in both Flamingo and Google Malicious protocols.
- **Missing import in `config/google_malicious.py`**: Added `from util import param` — was referenced (for `param.vector_len`) but never imported, breaking the `-e` flag.
- **Minimum client validation**: Added guard in both `config/flamingo.py` and `config/google_malicious.py` to reject `num_clients < 128` with a clear error message.

---

## Adaptive Server Waiting Mode

### Overview
Added a new server waiting strategy (`-w adaptive`) alongside the existing fixed-timeout mode (`-w fixed`, the default). In adaptive mode, the server processes messages as soon as enough arrive, rather than waiting for a fixed timer to expire.

### How it works
- **Fixed mode (default):** Server uses preconfigured timeouts (`wt_flamingo_report=10s`, `wt_flamingo_crosscheck=3s`, `wt_flamingo_reconstruction=3s`) to decide when to process. Some clients may drop out during the wait.
- **Adaptive mode:** Server checks in `receiveMessage()` whether a sufficient fraction of expected messages have arrived. If so, it immediately triggers processing. A 60s safety timeout ensures the protocol doesn't hang if messages never arrive.
  - Report step: triggers when `wait_threshold` fraction of clients send VECTOR messages (default: 90%)
  - Crosscheck step: triggers when `wait_threshold` fraction of committee members send signatures
  - Reconstruction step: triggers when `committee_threshold` decryption shares arrive

### Usage
```bash
# Fixed mode (default, same as before)
python abides.py -c flamingo -n 128 -i 1 -w fixed

# Adaptive mode
python abides.py -c flamingo -n 128 -i 1 -w adaptive

# Adaptive mode with custom threshold (95% of expected messages)
python abides.py -c flamingo -n 128 -i 1 -w adaptive --wait_threshold 0.95
```

### Implementation details
- `SA_ServiceAgent` accepts `wait_mode` ('fixed'/'adaptive') and `wait_threshold` (float, default 0.9)
- `_check_adaptive_trigger()` is called at the end of every `receiveMessage()` and triggers the appropriate processing function when the threshold is met
- `_round_processed` flag prevents double-processing when both the adaptive trigger and a safety-timeout wakeup fire
- `wakeup()` skips processing if `_round_processed` is True or the protocol has finished

### Files changed
- `config/flamingo.py` — added `-w`/`--wait_mode`, `--wait_threshold`, and `-e`/`--vector_len` CLI args
- `agent/flamingo/SA_ServiceAgent.py` — adaptive waiting logic in `__init__`, `receiveMessage`, `wakeup`, and all round-transition points

---

## Benchmark Suite

### `benchmark_suite.py` (new file)
A parameter-sweep benchmarking script that automates running the Flamingo protocol across different configurations and collects performance data.

### Usage
```bash
# Default sweep: 128, 256, 512 clients × 1, 3 iterations × fixed, adaptive modes
python benchmark_suite.py

# Custom parameters
python benchmark_suite.py --clients 128 256 512 --vector-lens 1000 16000 64000 --iterations 1 3 --modes fixed adaptive

# Quick smoke test
python benchmark_suite.py --quick
```

### Output
- **Formatted table** printed to stdout with per-configuration breakdown
- **`benchmark_results.csv`** with all raw data for post-processing

### Columns tracked
- Server computation: report, crosscheck, reconstruction (total and ECC scalar mult breakdown)
- Client computation: report, crosscheck, reconstruction (average across all clients)
- Communication: client→server VECTOR bytes, server→committee SIGN bytes
- Metadata: online clients, correctness check, wall-clock time
- Wait threshold (for adaptive mode sweeps)

### Threshold sweeps
```bash
python benchmark_suite.py --clients 128 256 --vector-lens 10000 64000 --modes adaptive --thresholds 0.75 0.9 0.95
```

---

## Google Malicious: Bug Fix + Adaptive Mode

### Bug fix: offline-offline neighbor mask cancellation
**Root cause:** In `reconstruction()`, the server cancelled pairwise masks for ALL neighbors of each offline client, including neighbors that were also offline. When both clients in a pair are offline, neither contributed a masked vector, so there is nothing to cancel. Applying cancellation corrupted the sum.

**Fix:** Filter to only cancel masks between offline clients and their ONLINE neighbors (line 782: `if j in self.online_set`).

**Impact:** Before the fix, any iteration where an offline client had another offline neighbor would produce incorrect results. The alternating correct/incorrect pattern across iterations was caused by different random dropout patterns.

### Bug fix: iteration state reset
Added comprehensive state reset at iteration boundaries. Previously only `recv_user_pubkeys` was cleared; late messages from iteration N could contaminate iteration N+1's recv pools.

### Increased default neighbor count
Changed default `num_neighbors` from `ceil(log2(n))` to `2 * ceil(log2(n))` (e.g., 14 instead of 7 for 128 clients). The original count was too tight for ~5-10% dropout — if too many of a client's neighbors drop, Shamir reconstruction fails. The doubled count provides sufficient redundancy.

The client now respects the `--num_neighbors` CLI flag (previously ignored and hardcoded to `ceil(log2(n))`).

### Adaptive mode
Added `-w fixed|adaptive` flag. In adaptive mode, the server waits until the message pool has enough messages, then closes the pool and processes via `setWakeup`. All processing happens in `wakeup()` (normal flow), never inside `receiveMessage`.

| Round | Name | Adaptive behavior |
|-------|------|-------------------|
| 1 | advertise_keys | Wait for **ALL** clients |
| 2 | establish_graph | Wait for **ALL** clients |
| 3 | forward_shares | Wait for **ALL** clients |
| 4 | collection | Wait for `wait_threshold` fraction |
| 5 | check_alive | Wait for **ALL** online clients |
| 6 | reconstruction | Wait for **ALL** online clients |

Each round also has a safety timeout (the original fixed wait time) that fires if the adaptive trigger never reaches its threshold.

### Graceful handling of insufficient shares
When dropout exceeds the 2/3 threshold for a particular client's neighborhood, mi or ai share reconstruction fails. Instead of producing garbage, the server now:
- Skips that client and continues with the rest
- Prints a clear `AGGREGATION FAILED` message with failure counts and diagnostics
- Continues to the next iteration

### Multi-iteration correctness (tested with 5 seeds × 3 iterations each)
- **Fixed mode:** 15/15 iterations correct (with doubled neighbors)
- **Adaptive mode:** Iteration 1 always correct across all seeds. Multi-iteration has residual timing issues: ABIDES does not support wakeup cancellation, so stale safety timeouts from round N can fire after the adaptive trigger already advanced to round N+1. A `_completed_rounds` set handles most cases but edge cases remain.

### Files changed
- `config/google_malicious.py` — added `-w`, `--wait_threshold`, fixed `type=bool` → `type=int` for `-d`, default `num_neighbors = 2*ceil(log2(n))`
- `agent/google_malicious/SA_ServiceAgent.py` — offline-online fix, adaptive waiting, `_next_round` helper, `_completed_rounds`/`_adaptive_triggered` tracking, graceful failure handling, iteration state reset
- `agent/google_malicious/SA_ClientAgent.py` — respect `num_neighbors` parameter

---

## Files Changed (cumulative)

| File | Type |
|------|------|
| `requirements.txt` | Compatibility fix |
| `util/util.py` | Key cache + binary serialization |
| `util/param.py` | Neighbor graph cache |
| `util/crypto/ecchash.py` | hash-to-curve optimization + sgn0 fix |
| `util/benchmark.py` | **New file** - benchmark collection |
| `benchmark_suite.py` | **New file** - parameter-sweep benchmark script |
| `agent/flamingo/SA_ClientAgent.py` | DH cache, logging, benchmarks, duplicate key fix, START_REPORT handler |
| `agent/flamingo/SA_ServiceAgent.py` | Sequential recon, adaptive waiting, retry, logging, benchmarks, server-driven first round |
| `config/flamingo.py` | argparse fixes, new flags (-w, -e, --wait_threshold), benchmark summary call |
| `config/google_malicious.py` | argparse fixes, new flags (-w, --wait_threshold) |
| `agent/google_malicious/SA_ServiceAgent.py` | Adaptive waiting, recordTime fix, _next_round helper |
| `agent/google_malicious/SA_ClientAgent.py` | recordTime fix |
| `agent/dkg/SA_ClientAgent.py` | KeyError fix |
