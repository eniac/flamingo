#!/usr/bin/env python3
"""Analyze timeline CSV files produced by Flamingo / Google Malicious experiments.

Reads timeline_*.csv files and prints per-round wait times, server computation,
and total iteration time in a formatted table.

Usage:
    python analyze_timeline.py timeline_flamingo.csv
    python analyze_timeline.py timeline_flamingo.csv timeline_google_malicious.csv
    python analyze_timeline.py timeline_flamingo_128.csv timeline_flamingo_256.csv timeline_google_128.csv
"""
import argparse
import json
import sys
import pandas as pd


def read_timeline_csv(filepath):
    """Read a timeline CSV, extracting the optional JSON params header.

    Returns (params_dict, DataFrame).
    """
    params = {}
    with open(filepath, 'r') as f:
        first_line = f.readline()
        if first_line.startswith("# "):
            try:
                params = json.loads(first_line[2:])
            except json.JSONDecodeError:
                pass
    df = pd.read_csv(filepath, comment="#")
    return params, df


def detect_protocol(df):
    """Detect whether a timeline CSV is from flamingo or google_malicious."""
    events = df["event"].str.lower()
    if events.str.contains("advertise_keys").any():
        return "google_malicious"
    elif events.str.contains("report").any():
        return "flamingo"
    else:
        raise ValueError("Cannot detect protocol from timeline events.")


def find_event(events, keyword):
    """Find offset_s for the first event containing keyword."""
    match = [(t, e) for e, t in events if keyword in e]
    if match:
        return match[0][0]
    return None


def analyze_flamingo(df):
    """Analyze a Flamingo timeline CSV."""
    rounds = ["report", "crosscheck", "reconstruction"]
    header = f"{'Iter':>4} {'Report Wait':>12} {'CC Wait':>12} {'Recon Wait':>12} {'Srv Comp':>10} {'Total':>8}"
    print(header)
    print("-" * len(header))

    all_data = []
    for it in sorted(df["iteration"].unique()):
        it_df = df[df["iteration"] == it]
        events = list(zip(it_df["event"], it_df["offset_s"]))

        starts, finishes = {}, {}
        for evt, t in events:
            for r in rounds:
                if f"starts processing {r}" in evt:
                    starts[r] = t
                if f"finishes {r}" in evt:
                    finishes[r] = t

        report_wait = starts.get("report", 0)
        cc_wait = starts.get("crosscheck", 0) - finishes.get("report", 0)
        recon_wait = starts.get("reconstruction", 0) - finishes.get("crosscheck", 0)

        srv_comp = sum(finishes.get(r, 0) - starts.get(r, 0) for r in rounds)
        total = it_df["offset_s"].max()

        all_data.append({
            "report_wait": report_wait, "cc_wait": cc_wait,
            "recon_wait": recon_wait, "srv_comp": srv_comp, "total": total
        })
        print(f"{it:>4} {report_wait:>12.3f} {cc_wait:>12.3f} {recon_wait:>12.3f} {srv_comp:>10.3f} {total:>8.2f}")

    if len(all_data) > 1:
        avg = {k: sum(d[k] for d in all_data) / len(all_data) for k in all_data[0]}
        print(f" AVG {avg['report_wait']:>12.3f} {avg['cc_wait']:>12.3f} {avg['recon_wait']:>12.3f} {avg['srv_comp']:>10.3f} {avg['total']:>8.2f}")

    return all_data


def analyze_google(df):
    """Analyze a Google Malicious timeline CSV."""
    rounds = ["advertise_keys", "graph", "forward_shares", "collection", "check_alive", "reconstruction"]
    short = ["R1 adkey", "R2 graph", "R3 share", "R4 coll", "R5 alive", "R6 recon"]

    header = f"{'Iter':>4}" + "".join(f" {s:>10}" for s in short) + f" {'Srv Comp':>10} {'Total':>8}"
    print(header)
    print("-" * len(header))

    all_data = []
    for it in sorted(df["iteration"].unique()):
        it_df = df[df["iteration"] == it]
        events = list(zip(it_df["event"], it_df["offset_s"]))

        starts, finishes = {}, {}
        for evt, t in events:
            for r in rounds:
                if f"starts processing {r}" in evt:
                    starts[r] = t
                if f"finishes {r}" in evt:
                    finishes[r] = t

        waits = [starts.get(rounds[0], 0)]
        for i in range(1, 6):
            waits.append(starts.get(rounds[i], 0) - finishes.get(rounds[i - 1], 0))

        srv_comp = sum(finishes.get(r, 0) - starts.get(r, 0) for r in rounds)
        total = it_df["offset_s"].max()

        row = {f"wait_{i}": waits[i] for i in range(6)}
        row["srv_comp"] = srv_comp
        row["total"] = total
        all_data.append(row)

        line = f"{it:>4}" + "".join(f" {w:>10.3f}" for w in waits) + f" {srv_comp:>10.3f} {total:>8.2f}"
        print(line)

    if len(all_data) > 1:
        avg_waits = [sum(d[f"wait_{i}"] for d in all_data) / len(all_data) for i in range(6)]
        avg_sc = sum(d["srv_comp"] for d in all_data) / len(all_data)
        avg_t = sum(d["total"] for d in all_data) / len(all_data)
        line = f" AVG" + "".join(f" {w:>10.3f}" for w in avg_waits) + f" {avg_sc:>10.3f} {avg_t:>8.2f}"
        print(line)

    return all_data


def main():
    parser = argparse.ArgumentParser(description="Analyze timeline CSV files from Flamingo / Google Malicious experiments.")
    parser.add_argument("files", nargs="+", help="Timeline CSV file(s) to analyze")
    args = parser.parse_args()

    summaries = []

    for fname in args.files:
        params, df = read_timeline_csv(fname)
        protocol = params.get("protocol") or detect_protocol(df)
        n_iters = df["iteration"].nunique()
        n_clients = params.get("num_clients", "?")

        # Fallback: infer client count from event text
        if n_clients == "?":
            for evt in df["event"]:
                if "report_vectors" in evt or "pubkeys" in evt:
                    try:
                        n_clients = evt.split("/")[-1].rstrip(")")
                        break
                    except (IndexError, ValueError):
                        pass

        print(f"\n{'=' * 70}")
        print(f"  {fname}")
        info_parts = [f"Protocol: {protocol}", f"Clients: {n_clients}", f"Iterations: {n_iters}"]
        if params.get("wait_mode"):
            info_parts.append(f"Mode: {params['wait_mode']}")
        if params.get("wait_threshold"):
            info_parts.append(f"Threshold: {params['wait_threshold']}")
        if params.get("vector_len"):
            info_parts.append(f"Vec len: {params['vector_len']}")
        if params.get("seed") is not None:
            info_parts.append(f"Seed: {params['seed']}")
        print(f"  {' | '.join(info_parts)}")
        print(f"{'=' * 70}\n")

        if protocol == "flamingo":
            data = analyze_flamingo(df)
        else:
            data = analyze_google(df)

        avg_total = sum(d["total"] for d in data) / len(data)
        summaries.append((fname, protocol, n_clients, avg_total))

    if len(summaries) > 1:
        print(f"\n{'=' * 70}")
        print("  COMPARISON SUMMARY")
        print(f"{'=' * 70}")
        print(f"  {'File':<40} {'Protocol':<18} {'n':>5} {'Avg Total':>10}")
        print(f"  {'-' * 75}")
        for fname, proto, n, avg in summaries:
            print(f"  {fname:<40} {proto:<18} {n:>5} {avg:>9.2f}s")

        # If we have both protocols, compute speedup
        fl = [s for s in summaries if s[1] == "flamingo"]
        go = [s for s in summaries if s[1] == "google_malicious"]
        if fl and go:
            print(f"\n  Speedup (Google / Flamingo): {go[0][3] / fl[0][3]:.1f}x")


if __name__ == "__main__":
    main()
