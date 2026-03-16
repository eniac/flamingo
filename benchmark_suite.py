#!/usr/bin/env python3
"""
Benchmark suite for the Flamingo secure aggregation protocol.

Sweeps over different parameters (num_clients, vector_len, num_iterations, wait_mode)
and collects performance data into a CSV file and prints a summary table.

Usage:
    python benchmark_suite.py                     # Run default sweep
    python benchmark_suite.py --clients 128 256   # Custom client counts
    python benchmark_suite.py --quick             # Quick smoke test
    python benchmark_suite.py --help              # Show all options

Output:
    benchmark_results.csv   - Raw results for all runs
    Printed summary table   - Formatted to stdout
"""

import argparse
import csv
import importlib
import io
import os
import re
import subprocess
import sys
import time


def run_single(num_clients, vector_len, num_iterations, wait_mode, wait_threshold=0.9, seed=42, protocol="flamingo"):
    """Run a single experiment and parse the output."""
    cmd = [
        sys.executable, "abides.py",
        "-c", protocol,
        "-n", str(num_clients),
        "-i", str(num_iterations),
        "-w", wait_mode,
        "--wait_threshold", str(wait_threshold),
        "-s", str(seed),
    ]
    if vector_len is not None:
        cmd += ["-e", str(vector_len)]

    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    wall_clock = time.perf_counter() - start

    stdout = result.stdout
    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})")
        print(result.stderr[-500:] if result.stderr else "no stderr")
        return None

    # Parse output
    data = {
        "protocol": protocol,
        "num_clients": num_clients,
        "vector_len": vector_len if vector_len else 16000,
        "num_iterations": num_iterations,
        "wait_mode": wait_mode,
        "wait_threshold": wait_threshold,
        "wall_clock_s": round(wall_clock, 2),
    }

    # Parse final sums to verify correctness (both "[Server] final sum:" and "final sum =")
    sums = re.findall(r"final sum\s*[=:]\s*\[(\d+)", stdout)
    fails = stdout.count("AGGREGATION FAILED")
    data["sums_correct"] = (len(sums) > 0 and all(int(s) > 0 for s in sums) and fails == 0)
    data["agg_failures"] = fails

    # Parse server timings from benchmark summary
    def parse_time(pattern):
        m = re.search(pattern, stdout)
        return float(m.group(1)) if m else 0.0

    # Parse from "Average across N iterations" section or single iteration
    # Try average section first
    avg_section = stdout[stdout.rfind("Average across"):]  if "Average across" in stdout else stdout

    def parse_timedelta_str(td_str):
        """Parse 'HH:MM:SS.ffffff' or 'X days HH:MM:SS' to seconds."""
        parts = td_str.split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        return 0.0

    # Parse server timings - works for both flamingo and google_malicious
    for step in ["report", "crosscheck", "reconstruction"]:
        val = 0.0
        m = re.search(rf"Server {step}\s+([\d.]+)s", avg_section)
        if m:
            val = float(m.group(1))
        else:
            m = re.search(rf"^\s+{step}\s+([\d.]+)s", stdout, re.MULTILINE)
            if m:
                val = float(m.group(1))
            else:
                # google_malicious legacy: "Report step:  0 days 00:00:00.002330"
                m = re.search(rf"{step.capitalize()} step:\s+.*?(\d+:\d+:[\d.]+)", stdout)
                if m:
                    val = parse_timedelta_str(m.group(1))
        data[f"srv_{step}_s"] = val

    for step in ["report", "crosscheck", "reconstruction"]:
        m = re.search(rf"Client {step}\s+avg=([\d.]+)s", avg_section)
        if not m:
            m = re.search(rf"^\s+{step}\s+avg=([\d.]+)s", stdout, re.MULTILINE)
        data[f"clt_{step}_avg_s"] = float(m.group(1)) if m else 0.0

    # Parse ECC scalar mult time from last iteration's detail
    m = re.search(r"ecc_scalar_mult\s+([\d.]+)s", stdout)
    data["srv_ecc_mult_s"] = float(m.group(1)) if m else 0.0

    # Parse communication
    m = re.search(r"client->server \(VECTOR\)\s+total=\s*([\d,]+)", stdout)
    data["comm_vector_bytes"] = int(m.group(1).replace(",", "")) if m else 0

    m = re.search(r"server->committee \(SIGN\)\s+total=\s*([\d,]+)", stdout)
    data["comm_sign_bytes"] = int(m.group(1).replace(",", "")) if m else 0

    # Parse online clients
    online = re.findall(r"online_clients: (\d+)", stdout)
    data["avg_online"] = round(sum(int(x) for x in online) / len(online)) if online else 0

    return data


def print_table(results, title="Benchmark Results"):
    """Print a formatted table."""
    if not results:
        print("No results to display.")
        return

    print(f"\n{'=' * 120}")
    print(f" {title}")
    print(f"{'=' * 120}")

    header = (f"{'Protocol':>16} {'Clients':>7} {'VecLen':>7} {'Iters':>5} {'Mode':>8} {'Thresh':>6} "
              f"{'Wall(s)':>8} "
              f"{'SrvReport':>10} {'SrvRecon':>10} {'ECCMult':>10} "
              f"{'CltReport':>10} {'CltRecon':>10} "
              f"{'CommVec(KB)':>12} {'CommSign(KB)':>13} "
              f"{'Online':>6} {'OK':>3}")
    print(header)
    print("-" * 145)

    for r in results:
        th_str = f"{r.get('wait_threshold', 0.9):.2f}" if r['wait_mode'] == 'adaptive' else "  -"
        line = (f"{r.get('protocol','flamingo'):>16} {r['num_clients']:>7} {r['vector_len']:>7} {r['num_iterations']:>5} {r['wait_mode']:>8} {th_str:>6} "
                f"{r['wall_clock_s']:>8.1f} "
                f"{r['srv_report_s']:>10.4f} {r['srv_reconstruction_s']:>10.4f} {r['srv_ecc_mult_s']:>10.4f} "
                f"{r['clt_report_avg_s']:>10.4f} {r['clt_reconstruction_avg_s']:>10.4f} "
                f"{r['comm_vector_bytes']/1024:>12.0f} {r['comm_sign_bytes']/1024:>13.0f} "
                f"{r['avg_online']:>6} {'Y' if r['sums_correct'] else 'N':>3}")
        print(line)

    print(f"{'=' * 145}")


def main():
    parser = argparse.ArgumentParser(description="Flamingo benchmark suite")
    parser.add_argument("--protocols", type=str, nargs="+", default=["flamingo"],
                        choices=["flamingo", "google_malicious"],
                        help="Protocols to test (default: flamingo)")
    parser.add_argument("--clients", type=int, nargs="+", default=[128, 256, 512],
                        help="Number of clients to test (default: 128 256 512)")
    parser.add_argument("--vector-lens", type=int, nargs="+", default=None,
                        help="Vector lengths to test (default: use param.vector_len)")
    parser.add_argument("--iterations", type=int, nargs="+", default=[1, 3],
                        help="Number of iterations to test (default: 1 3)")
    parser.add_argument("--modes", type=str, nargs="+", default=["fixed", "adaptive"],
                        choices=["fixed", "adaptive"],
                        help="Wait modes to test (default: fixed adaptive)")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.9],
                        help="Wait thresholds to test for adaptive mode (default: 0.9)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--output", type=str, default="benchmark_results.csv",
                        help="Output CSV file (default: benchmark_results.csv)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 128 clients, 1 iteration, both modes")
    args = parser.parse_args()

    if args.quick:
        args.clients = [128]
        args.iterations = [1]
        args.vector_lens = None
        args.modes = ["fixed", "adaptive"]
        args.thresholds = [0.9]

    # Build list of configurations
    configs = []
    vec_lens = args.vector_lens if args.vector_lens else [None]  # None = use default
    for proto in args.protocols:
        for n in args.clients:
            for vl in vec_lens:
                for iters in args.iterations:
                    for mode in args.modes:
                        if mode == "fixed":
                            configs.append((proto, n, vl, iters, mode, 0.9))
                        else:
                            for th in args.thresholds:
                                configs.append((proto, n, vl, iters, mode, th))

    print(f"Benchmark Suite")
    print(f"===============")
    print(f"Configurations to run: {len(configs)}")
    print(f"Protocols: {args.protocols}")
    print(f"Client counts: {args.clients}")
    print(f"Vector lengths: {vec_lens}")
    print(f"Iterations: {args.iterations}")
    print(f"Wait modes: {args.modes}")
    print(f"Thresholds: {args.thresholds}")
    print(f"Seed: {args.seed}")
    print()

    results = []
    for i, (proto, n, vl, iters, mode, th) in enumerate(configs):
        vl_str = str(vl) if vl else "default"
        th_str = f" th={th}" if mode == "adaptive" else ""
        print(f"[{i+1}/{len(configs)}] {proto} n={n} veclen={vl_str} iters={iters} mode={mode}{th_str} ... ", end="", flush=True)
        data = run_single(n, vl, iters, mode, wait_threshold=th, seed=args.seed, protocol=proto)
        if data:
            results.append(data)
            print(f"done ({data['wall_clock_s']:.1f}s, sum_ok={data['sums_correct']})")
        else:
            print("FAILED")

    # Print summary table
    print_table(results)

    # Write CSV
    if results:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
