#!/usr/bin/env python3
"""Plot per-round server wait time: Flamingo vs BBGLR with Morandi color palette."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# ── Morandi color palette ──────────────────────────────────────────────────
MORANDI = {
    'sage':      '#9CAF88',   # Threshold of N clients
    'steel':     '#7B9CBF',   # Committee only (~60 members)
    'rose':      '#C9908E',   # Wait for ALL N clients
    'taupe':     '#C4B6A6',   # Wait for ALL online clients
}

COLORS = [MORANDI['sage'], MORANDI['steel'], MORANDI['rose'], MORANDI['taupe']]
LABELS = [
    'Threshold of N clients',
    'Committee only (~60 members)',
    'Wait for ALL N clients',
    'Wait for ALL online clients',
]

# ── Round-category mapping ─────────────────────────────────────────────────
# Flamingo: Report=threshold, Crosscheck=committee, Reconstruction=committee
FLAMINGO_ROUND_CAT = {
    'report': 0,        # threshold
    'crosscheck': 1,    # committee
    'reconstruction': 1, # committee
}
# BBGLR: advertise_keys/graph/forward_shares=all N, collection=threshold,
#         check_alive/reconstruction=all online
BBGLR_ROUND_CAT = {
    'advertise_keys': 2,    # all N
    'graph': 2,              # all N
    'forward_shares': 2,     # all N
    'collection': 0,         # threshold
    'check_alive': 3,        # all online
    'reconstruction': 3,     # all online
}

# ── Parse timeline CSV ─────────────────────────────────────────────────────
def parse_timeline(path):
    """Return list of (iteration, offset_s, event) tuples."""
    df = pd.read_csv(path)
    return list(zip(df['iteration'], df['offset_s'], df['event']))

def compute_wait_times(events, protocol):
    """Compute per-round wait times for each iteration.

    Returns dict: {iteration: {category_index: total_wait_seconds}}
    Wait time per round = (start processing) - (finish previous round/iter start).
    """
    cat_map = FLAMINGO_ROUND_CAT if protocol == 'flamingo' else BBGLR_ROUND_CAT

    # Group events by iteration
    iters = {}
    for it, offset, ev in events:
        iters.setdefault(it, []).append((offset, ev))

    result = {}
    for it, ev_list in iters.items():
        ev_list.sort(key=lambda x: x[0])
        cats = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}

        last_send_time = 0.0  # when server finished previous round & sent requests

        for offset, ev in ev_list:
            if ev.startswith('server starts processing '):
                round_name = ev.replace('server starts processing ', '')
                wait = offset - last_send_time
                if round_name in cat_map:
                    cats[cat_map[round_name]] += wait

            if ev.startswith('server finishes '):
                last_send_time = offset

        result[it] = cats

    return result


def avg_wait_by_category(wait_dict):
    """Average wait times across iterations, return [cat0, cat1, cat2, cat3]."""
    n_iters = len(wait_dict)
    totals = [0.0, 0.0, 0.0, 0.0]
    for cats in wait_dict.values():
        for i in range(4):
            totals[i] += cats[i]
    return [t / n_iters for t in totals]


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    base = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(base)

    ns = [128, 256, 512]
    flamingo_data = []
    bbglr_data = []

    for n in ns:
        fl_path = os.path.join(root, f'timeline_flamingo_{n}.csv')
        gm_path = os.path.join(root, f'timeline_google_malicious_{n}.csv')

        fl_events = parse_timeline(fl_path)
        gm_events = parse_timeline(gm_path)

        fl_waits = compute_wait_times(fl_events, 'flamingo')
        gm_waits = compute_wait_times(gm_events, 'google_malicious')

        flamingo_data.append(avg_wait_by_category(fl_waits))
        bbglr_data.append(avg_wait_by_category(gm_waits))

    # ── Plot ───────────────────────────────────────────────────────────────
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    fig, ax = plt.subplots(figsize=(14, 7))

    n_groups = len(ns)
    bar_width = 0.32
    group_gap = 1.2

    for gi, n in enumerate(ns):
        fl = flamingo_data[gi]
        gm = bbglr_data[gi]
        fl_total = sum(fl)
        gm_total = sum(gm)

        x_fl = gi * group_gap
        x_gm = gi * group_gap + bar_width + 0.05

        # Stacked bars — Flamingo
        bottom = 0
        for ci in range(4):
            if fl[ci] > 0.01:
                ax.bar(x_fl, fl[ci], bar_width, bottom=bottom,
                       color=COLORS[ci], edgecolor='white', linewidth=0.5)
            bottom += fl[ci]

        # Stacked bars — BBGLR
        bottom = 0
        for ci in range(4):
            if gm[ci] > 0.01:
                ax.bar(x_gm, gm[ci], bar_width, bottom=bottom,
                       color=COLORS[ci], edgecolor='white', linewidth=0.5)
            bottom += gm[ci]

        # Total labels
        ax.text(x_fl, fl_total + 0.8, f'{fl_total:.1f}s',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
        ax.text(x_gm, gm_total + 0.8, f'{gm_total:.1f}s',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

        # Speedup ratio
        ratio = gm_total / fl_total
        mid_x = (x_fl + x_gm) / 2
        ax.text(mid_x, max(fl_total, gm_total) + 3.5, f'{ratio:.1f}x',
                ha='center', va='bottom', fontsize=10.5, color='#666666',
                fontstyle='italic')

    # X-axis
    tick_positions = []
    tick_labels_list = []
    for gi, n in enumerate(ns):
        x_fl = gi * group_gap
        x_gm = gi * group_gap + bar_width + 0.05
        tick_positions.extend([x_fl, x_gm])
        tick_labels_list.extend(['Flamingo', 'BBGLR'])

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels_list, fontsize=11)

    # Group labels (n = ...)
    for gi, n in enumerate(ns):
        mid = gi * group_gap + (bar_width + 0.05) / 2
        ax.text(mid, -5.5, f'n = {n}', ha='center', fontsize=12, fontweight='bold')

    ax.set_ylabel('Server Wait Time per Iteration (s)', fontsize=13)
    ax.set_xlabel('Number of Clients (n)', fontsize=13, labelpad=30)
    ax.set_title('Per-Round Server Wait Time: Flamingo vs. BBGLR', fontsize=15, fontweight='bold', pad=15)
    ax.set_ylim(0, 55)

    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[i], edgecolor='white')
               for i in range(4)]
    ax.legend(handles, LABELS, loc='upper left', framealpha=0.9, fontsize=10.5,
              edgecolor='#cccccc')

    plt.tight_layout()

    out_png = os.path.join(base, 'wait_time_comparison.png')
    out_pdf = os.path.join(base, 'wait_time_comparison.pdf')
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f'Saved: {out_png}')
    print(f'Saved: {out_pdf}')
    plt.close()


if __name__ == '__main__':
    main()
