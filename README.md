# Flamingo

Flamingo is a system built for privacy-preserving federated learning, where individual training weights are aggregated using secure aggregation.
This implementation accompanies our [paper](https://eprint.iacr.org/2023/486) by Yiping Ma, Jess Woods, Sebastian Angel, Antigoni Polychroniadou and Tal Rabin at IEEE S&P (Oakland) 2023.

WARNING: This is an academic proof-of-concept prototype and is not production-ready.

## Overview
We integrate our code into [ABIDES](https://github.com/jpmorganchase/abides-jpmc-public), an open-source highfidelity simulator designed for AI research in financial markets (e.g., stock exchanges).
The simulator supports tens of thousands of clients interacting with a server to facilitate transactions (and in our case to compute sums).
It also supports configurable pairwise network latencies.

Flamingo protocol works by steps (i.e., round trips).
A step includes waiting and processing messages.
The waiting time is set according to the network latency distribution and a target dropout rate.
See more details in Section 8 in our paper.

The `main` branch contains the code for private sum protocol and the `fedlearn` branch contains the code for private training of machine learning models.

## Installation Instructions

Requires Python 3.9+. You can use pip directly or set up a virtual environment.

**Option A: pip (simplest)**
```
pip install -r requirements.txt
```

**Option B: Conda**
```
conda create --name flamingo-v0 python=3.9.12
conda activate flamingo-v0
pip install -r requirements.txt
```

## Private Sum
The code is in branch `main`.

First enter into folder `pki_files`, and run
```
python setup_pki.py
```

### Command-Line Options

```
-c [protocol name]                  flamingo or google_malicious
-n [number of clients]              power of 2, minimum 128 (e.g., 128, 256, 512)
-i [number of iterations]           number of protocol iterations
-p [parallel mode]                  1=on, 0=off
-o [neighborhood size]              multiplicative factor of log(n)
-s [random seed]                    for reproducibility (optional)
-e [vector length]                  override input vector length (optional)
-w [wait mode]                      fixed or adaptive (default: fixed)
--wait_threshold [fraction]         threshold for adaptive mode (default: 0.9)
-d [debug mode]                     1=on, 0=off
```

Example commands:
```
python abides.py -c flamingo -n 128 -i 1 -p 1
python abides.py -c flamingo -n 256 -i 1 -w adaptive --wait_threshold 0.9
python abides.py -c google_malicious -n 256 -i 1 -w adaptive -e 10000
```

If you want to print out information of every agent, add `-d 1` to the above command.

NOTE: For ease of benchmarking, we separate the setup phase (folder `dkg`) and the private sum phase (folder `flamingo`). You can execute the command above directly as we provide the shares of the secret key to decryptors (a small random subset of clients) before the summation begins. If you wish to benchmark the setup independently, run `python abides.py -c dkg -n [number of decryptors]`.

## Server Waiting Modes: Fixed vs. Adaptive

The server supports two waiting modes that control when it proceeds from one round to the next.

- **Fixed mode** (`-w fixed`): The server waits for a preconfigured timeout in each round (set in `util/param.py`), regardless of how many messages have arrived. Simple and predictable, but wastes time if messages arrive early.

- **Adaptive mode** (`-w adaptive`): The server proceeds as soon as enough messages arrive, based on per-round trigger conditions. A 60-second safety timeout prevents indefinite waiting. This significantly reduces end-to-end latency (typically 29-40% faster) with only a marginal decrease in participation.

The two protocols have different round structures, so adaptive mode behaves differently for each.

### Flamingo Protocol

Flamingo has 3 rounds per iteration (after initialization). In adaptive mode, **all 3 rounds are threshold-based** — the server proceeds once it receives a sufficient fraction of expected messages.

| Round | Name | Fixed Timeout | Adaptive Trigger |
|-------|------|---------------|------------------|
| 1 | `report` | 10s | `wait_threshold` of `num_clients` vectors received (default 90%) |
| 2 | `crosscheck` | 3s | 2/3 of committee signatures received |
| 3 | `reconstruction` | 3s | `committee_threshold` decryption shares received (determined by protocol parameters) |

Round 3 uses a hard threshold (`committee_threshold`) derived from the committee size and secret-sharing fraction, not the configurable `wait_threshold`.

### Google Malicious Protocol

Google Malicious has 6 rounds per iteration: 3 setup rounds followed by 3 aggregation rounds. In adaptive mode, **only 1 round uses the configurable threshold**; the rest wait for all expected messages.

| Round | Name | Fixed Timeout | Adaptive Trigger |
|-------|------|---------------|------------------|
| 1 | `advertise_keys` | 10s | All `num_clients` pubkeys received |
| 2 | `establish_graph` | 10s | All graph choices received |
| 3 | `forward_shares` | 30s | All backup shares received |
| 4 | `collection` | 10s | `wait_threshold` of vectors received (default 90%) |
| 5 | `check_alive` | 3s | All ACKs from online clients |
| 6 | `reconstruction` | 2s | All reconstruction shares from online clients |

Rounds 1-3 are setup phases where every client's data is needed for correct pairwise mask construction, so they require 100% participation. Only **Round 4 (collection)** applies the configurable `wait_threshold`. Rounds 5-6 wait for all *online* clients — here "online" refers to the set of clients that successfully submitted their masked vector in Round 4 (collection). Any client that did not respond in Round 4 is considered offline/dropped out, and Rounds 5-6 only expect messages from the remaining online set.

> **Note on Rounds 5-6 threshold:** The BBGLR protocol theoretically supports waiting for only a subset of clients in Rounds 5 and 6. However, doing so requires the server to track every client's neighbor list and maintain per-client state about which of their neighbors are online versus offline. This adds significant bookkeeping complexity to the implementation. In practice, waiting for all online clients in these rounds is a much simpler approach and is what we implement here.

Despite most rounds waiting for all messages, adaptive mode still provides large speedups because the server proceeds **immediately** when all messages arrive rather than waiting out the remaining fixed timeout.

## Machine Learning Applications
The code is in branch `fedlearn`.
The machine learning model we use in this repository is a multi-layer perceptron classifier (`MLPClassfier` in `sklearn`) that can pull a variety of different datasets from the [pmlb website](https://epistasislab.github.io/pmlb/index.html). Users might wish to implement more complex models themselves.

Beyond the aforementioned configs, we provide machine learning training configs below.
```
-t [dataset name]
-s [random seed (optional)]
-e [input vector length]
-x [float-as-int encoding constant (optional)]
-y [float-as-int multiplier (optional)]
```

Example command:
```
python abides.py -c flamingo -n 128 -i 5 -p 1 -t mnist
```

## Benchmark Suite

The benchmark suite (`benchmark_suite.py`) sweeps over different parameter combinations and collects performance data into a CSV file with a printed summary table.

```
python benchmark_suite.py                     # Run default sweep
python benchmark_suite.py --quick             # Quick smoke test (128 clients, 1 iteration, both modes)
python benchmark_suite.py --help              # Show all options
```

### Benchmark Options

```
--protocols [names]       Protocols to test: flamingo, google_malicious (default: flamingo)
--clients [counts]        Client counts to sweep (default: 128 256 512)
--vector-lens [lengths]   Vector lengths to sweep (default: use param.vector_len)
--iterations [counts]     Iteration counts to sweep (default: 1 3)
--modes [modes]           Wait modes to test: fixed, adaptive (default: fixed adaptive)
--thresholds [fractions]  Thresholds for adaptive mode (default: 0.9)
--seed [int]              Random seed for reproducibility (default: 42)
--output [file]           Output CSV file (default: benchmark_results.csv)
--quick                   Quick smoke test: 128 clients, 1 iteration, both modes
```

### Examples

Sweep over multiple client counts with both protocols:
```
python benchmark_suite.py --protocols flamingo google_malicious --clients 128 256 512
```

Test adaptive mode with different thresholds:
```
python benchmark_suite.py --clients 256 --modes adaptive --thresholds 0.8 0.9 0.95
```

The suite automatically validates correctness by checking that aggregated sums are consistent across vector elements. If a bug is detected (e.g., incorrect mask cancellation), the benchmark stops immediately and reports the issue. Aggregation failures due to insufficient shares (from client dropout) are reported separately and are not treated as bugs.

### Output

- **`benchmark_results.csv`** — Raw results for all runs, including server/client timings, communication costs, and online client counts.
- **Summary table** — Printed to stdout with columns for protocol, client count, vector length, iterations, wait mode, wall-clock time, server/client step timings, communication bytes, and correctness.

## Timeline Analysis

Each experiment run produces a `timeline_<protocol>.csv` file recording per-iteration events with timestamps. The `analyze_timeline.py` script parses these CSVs and prints per-round wait times, server computation time, and total iteration time.

### Usage

```bash
# Analyze a single file
python analyze_timeline.py timeline_flamingo.csv

# Compare two protocols side-by-side
python analyze_timeline.py timeline_flamingo.csv timeline_google_malicious.csv

# Compare across client counts
python analyze_timeline.py timeline_flamingo_128.csv timeline_flamingo_256.csv timeline_flamingo_512.csv
```

### Example Output (n=512, adaptive mode, threshold=0.9)

**Flamingo** (3 rounds per iteration):

| Iter | Report Wait | CC Wait | Recon Wait | Srv Comp | Total |
|------|------------|---------|------------|----------|-------|
| 1 | 4.951s | 0.213s | 0.203s | 5.054s | 10.42s |
| 2 | 5.984s | 0.397s | 0.155s | 4.949s | 11.49s |
| 3 | 5.055s | 0.311s | 0.184s | 4.784s | 10.33s |
| 4 | 5.586s | 0.766s | 0.207s | 4.729s | 11.29s |
| **AVG** | **5.394s** | **0.422s** | **0.187s** | **4.879s** | **10.88s** |

**Google Malicious** (6 rounds per iteration):

| Iter | R1 adkey | R2 graph | R3 share | R4 coll | R5 alive | R6 recon | Srv Comp | Total |
|------|---------|---------|---------|---------|---------|---------|----------|-------|
| 1 | 3.000s | 10.000s | 29.982s | 1.626s | 2.998s | 1.996s | 2.409s | 52.01s |
| 2 | 10.000s | 10.000s | 14.234s | 1.731s | 2.997s | 1.995s | 2.475s | 43.43s |
| 3 | 10.000s | 10.000s | 21.608s | 1.721s | 2.998s | 1.995s | 2.661s | 50.98s |
| 4 | 10.000s | 10.000s | 16.333s | 1.595s | 2.998s | 1.995s | 2.608s | 45.53s |
| **AVG** | **8.250s** | **10.000s** | **20.539s** | **1.668s** | **2.998s** | **1.995s** | **2.538s** | **47.99s** |

(Benchmarked on a laptop, server compute time is much slower than standard case)

**Speedup: 4.4x** (Google 47.99s vs Flamingo 10.88s)

### Column Definitions

- **Report/Round Wait** — Time from round start until the server begins processing (i.e., network wait for sufficient messages).
- **Srv Comp** — Total server-side computation across all rounds (processing received messages).
- **Total** — End-to-end time per iteration (sum of all waits + server computation).

### Using AI Agents for Analysis

The timeline CSV files contain detailed per-event timestamps that are well-suited for deeper analysis with AI coding agents (e.g., Claude Code, Cursor, GitHub Copilot). Example prompts:

- *"Read timeline_flamingo_512.csv and explain why iteration 2 was slower than iteration 3."*
- *"Compare the wait time distributions across these timeline CSVs and plot a chart."*
- *"What is the 90th percentile network latency implied by the report wait times?"*

The CSV schema is simple: `iteration, simtime, offset_s, event` — any tool that reads CSV can process it.

## Additional Information
The server waiting time is set in `util/param.py` according to a target dropout rate (1%).
Specifically, for a target dropout rate, we set the waiting time according to the network latency (see `model/LatencyModel.py`). For each iteration, server total time = server waiting time + server computation time.

## Acknowledgement
We thank authors of [MicroFedML](https://eprint.iacr.org/2022/714.pdf) for providing an example template of ABIDES framework.
