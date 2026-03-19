#!/usr/bin/env python3
"""
Generate two charts for the Flamingo talk:
  Chart 1: Pure network wait time comparison (Flamingo vs BBGLR)
  Chart 2: Total per-iteration time breakdown (client compute + network wait + server compute)

Key insight: the simulation sets setComputationDelay(0) for flamingo clients and uses
defaultComputationDelay=0.1s for google clients. Real client computation (measured by
time.perf_counter()) does NOT affect simulation time. So timeline wait ≈ pure network wait,
with a small correction for the default delays:
  - Flamingo: 0.1s per round (server default only)
  - Google:   0.2s per round (server + client default)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# ── Morandi color palette ──────────────────────────────────────────────────
MORANDI = {
    'sage':       '#9CAF88',
    'steel':      '#7B9CBF',
    'rose':       '#C9908E',
    'taupe':      '#C4B6A6',
    'lavender':   '#B8A9C9',
    'dusty_pink': '#D4A5A5',
    'olive':      '#A3B18A',
    'slate':      '#8E9AAF',
    'clay':       '#C9B99A',
    'mist':       '#B5C4B1',
}

# ── Round-category mapping (same as original) ─────────────────────────────
FLAMINGO_ROUND_CAT = {
    'report': 0,         # threshold
    'crosscheck': 1,     # committee
    'reconstruction': 1, # committee
}
BBGLR_ROUND_CAT = {
    'advertise_keys': 2,     # all N
    'graph': 2,              # all N
    'forward_shares': 2,     # all N
    'collection': 0,         # threshold
    'check_alive': 3,        # all online
    'reconstruction': 3,     # all online
}

FLAMINGO_ROUNDS = ['report', 'crosscheck', 'reconstruction']
GOOGLE_ROUNDS = ['advertise_keys', 'graph', 'forward_shares',
                 'collection', 'check_alive', 'reconstruction']

# Default computation delay per round (from Kernel config)
# Flamingo: server 0.1s + client 0s = 0.1s per round
# Google:   server 0.1s + client 0.1s = 0.2s per round
DEFAULT_DELAY_FLAMINGO = 0.1  # per round
DEFAULT_DELAY_GOOGLE = 0.2    # per round

# ── Client computation data (from microbenchmarks) ────────────────────────
# Flamingo: from benchmark_wait_study.csv (adaptive mode, vec_len=10000)
# Values are per-client average per iteration.
FLAMINGO_CLT = {
    128: {'report': 0.0361, 'crosscheck': 0.0002, 'reconstruction': 0.0464},
    256: {'report': 0.0392, 'crosscheck': 0.0002, 'reconstruction': 0.109},
    512: {'report': 0.0424, 'crosscheck': 0.0002, 'reconstruction': 0.237},
}

# Google: from experiments (adaptive mode, vec_len=10000, seed=42)
# adkey/graph/share are raw sums across all clients -> divide by N
# collection/crosscheck/reconstruction are already per-client
GOOGLE_CLT_RAW = {
    128: {
        'advertise_keys': 0.016665 / 128,
        'graph': 0.007900 / 128,
        'forward_shares': 0.350997 / 128,
        'collection': 0.017745,
        'crosscheck': 0.005099,
        'reconstruction': 0.018007,
    },
    256: {
        'advertise_keys': 0.037600 / 256,
        'graph': 0.016801 / 256,
        'forward_shares': 1.034602 / 256,
        'collection': 0.022012,
        'crosscheck': 0.006309,
        'reconstruction': 0.022355,
    },
    512: {
        'advertise_keys': 0.069048 / 512,
        'graph': 0.037813 / 512,
        'forward_shares': 2.955675 / 512,
        'collection': 0.026260,
        'crosscheck': 0.007445,
        'reconstruction': 0.026681,
    },
}


# ── Parse timeline CSV ─────────────────────────────────────────────────────
def parse_timeline(path):
    df = pd.read_csv(path)
    return list(zip(df['iteration'], df['offset_s'], df['event']))


def compute_per_round_times(events, protocol):
    """Compute per-round wait time and server computation for each iteration.

    Returns dict: {iteration: {round_name: {'wait': s, 'comp': s}}}
    """
    iters = {}
    for it, offset, ev in events:
        iters.setdefault(it, []).append((offset, ev))

    result = {}
    for it, ev_list in iters.items():
        ev_list.sort(key=lambda x: x[0])
        rounds = {}
        last_send_time = 0.0

        i = 0
        while i < len(ev_list):
            offset, ev = ev_list[i]
            if ev.startswith('server starts processing '):
                round_name = ev.replace('server starts processing ', '')
                wait = offset - last_send_time
                # Find matching finish event
                comp = 0.0
                for j in range(i + 1, len(ev_list)):
                    if ev_list[j][1].startswith('server finishes '):
                        comp = ev_list[j][0] - offset
                        break
                rounds[round_name] = {'wait': wait, 'comp': comp}

            if ev.startswith('server finishes '):
                last_send_time = offset
            i += 1

        result[it] = rounds
    return result


def avg_per_round(per_round_dict):
    """Average wait & comp across iterations for each round."""
    all_rounds = set()
    for it_data in per_round_dict.values():
        all_rounds.update(it_data.keys())

    avg = {}
    n_iters = len(per_round_dict)
    for rnd in all_rounds:
        total_wait = sum(d.get(rnd, {}).get('wait', 0) for d in per_round_dict.values())
        total_comp = sum(d.get(rnd, {}).get('comp', 0) for d in per_round_dict.values())
        avg[rnd] = {'wait': total_wait / n_iters, 'comp': total_comp / n_iters}
    return avg


# ── Compute everything ─────────────────────────────────────────────────────
def gather_data():
    base = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(base)
    ns = [128, 256, 512]

    data = {'flamingo': {}, 'google': {}}

    for n in ns:
        # ── Flamingo ──
        fl_events = parse_timeline(os.path.join(root, f'timeline_flamingo_{n}.csv'))
        fl_rounds = avg_per_round(compute_per_round_times(fl_events, 'flamingo'))

        n_fl_rounds = len(FLAMINGO_ROUNDS)
        fl_total_wait_raw = sum(fl_rounds.get(r, {}).get('wait', 0) for r in FLAMINGO_ROUNDS)
        fl_pure_network = fl_total_wait_raw - n_fl_rounds * DEFAULT_DELAY_FLAMINGO
        fl_server_comp = sum(fl_rounds.get(r, {}).get('comp', 0) for r in FLAMINGO_ROUNDS)
        fl_client_comp = sum(FLAMINGO_CLT[n].values())

        data['flamingo'][n] = {
            'network_wait': fl_pure_network,
            'server_comp': fl_server_comp,
            'client_comp': fl_client_comp,
            'rounds': fl_rounds,
            # Per-category wait (for chart 1)
            'cat_wait': compute_cat_wait(fl_rounds, 'flamingo', n),
        }

        # ── Google ──
        gm_events = parse_timeline(os.path.join(root, f'timeline_google_malicious_{n}.csv'))
        gm_rounds = avg_per_round(compute_per_round_times(gm_events, 'google'))

        n_gm_rounds = len(GOOGLE_ROUNDS)
        gm_total_wait_raw = sum(gm_rounds.get(r, {}).get('wait', 0) for r in GOOGLE_ROUNDS)
        gm_pure_network = gm_total_wait_raw - n_gm_rounds * DEFAULT_DELAY_GOOGLE
        gm_server_comp = sum(gm_rounds.get(r, {}).get('comp', 0) for r in GOOGLE_ROUNDS)
        gm_client_comp = sum(GOOGLE_CLT_RAW[n].values())

        data['google'][n] = {
            'network_wait': gm_pure_network,
            'server_comp': gm_server_comp,
            'client_comp': gm_client_comp,
            'rounds': gm_rounds,
            'cat_wait': compute_cat_wait(gm_rounds, 'google', n),
        }

    return ns, data


def compute_cat_wait(rounds, protocol, n):
    """Compute wait by category (threshold/committee/allN/allOnline), corrected."""
    cat_map = FLAMINGO_ROUND_CAT if protocol == 'flamingo' else BBGLR_ROUND_CAT
    delay = DEFAULT_DELAY_FLAMINGO if protocol == 'flamingo' else DEFAULT_DELAY_GOOGLE
    cats = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    for rnd, info in rounds.items():
        if rnd in cat_map:
            corrected = max(0, info['wait'] - delay)
            cats[cat_map[rnd]] += corrected
    return [cats[i] for i in range(4)]


# ── Chart 1: Pure Network Wait Time ───────────────────────────────────────
def plot_chart1(ns, data):
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    COLORS = [MORANDI['sage'], MORANDI['steel'], MORANDI['rose'], MORANDI['taupe']]
    LABELS = [
        'Threshold of N clients',
        'Committee only (~60 members)',
        'Wait for ALL N clients',
        'Wait for ALL online clients',
    ]

    fig, ax = plt.subplots(figsize=(14, 7))
    bar_width = 0.32
    group_gap = 1.2

    for gi, n in enumerate(ns):
        fl = data['flamingo'][n]['cat_wait']
        gm = data['google'][n]['cat_wait']
        fl_total = sum(fl)
        gm_total = sum(gm)

        x_fl = gi * group_gap
        x_gm = gi * group_gap + bar_width + 0.05

        # Stacked bars
        for vals, x_pos in [(fl, x_fl), (gm, x_gm)]:
            bottom = 0
            for ci in range(4):
                if vals[ci] > 0.01:
                    ax.bar(x_pos, vals[ci], bar_width, bottom=bottom,
                           color=COLORS[ci], edgecolor='white', linewidth=0.5)
                bottom += vals[ci]

        # Total labels
        ax.text(x_fl, fl_total + 0.8, f'{fl_total:.1f}s',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
        ax.text(x_gm, gm_total + 0.8, f'{gm_total:.1f}s',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

        # Speedup ratio
        ratio = gm_total / fl_total if fl_total > 0 else 0
        mid_x = (x_fl + x_gm) / 2
        ax.text(mid_x, max(fl_total, gm_total) + 3.5, f'{ratio:.1f}x',
                ha='center', va='bottom', fontsize=10.5, color='#666666',
                fontstyle='italic')

    # X-axis
    tick_positions, tick_labels = [], []
    for gi, n in enumerate(ns):
        x_fl = gi * group_gap
        x_gm = gi * group_gap + bar_width + 0.05
        tick_positions.extend([x_fl, x_gm])
        tick_labels.extend(['Flamingo', 'BBGLR'])

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=11)

    for gi, n in enumerate(ns):
        mid = gi * group_gap + (bar_width + 0.05) / 2
        ax.text(mid, -5.5, f'n = {n}', ha='center', fontsize=12, fontweight='bold')

    ax.set_ylabel('Pure Network Wait Time per Iteration (s)', fontsize=13)
    ax.set_xlabel('Number of Clients (n)', fontsize=13, labelpad=30)
    ax.set_title('Per-Round Server Network Wait: Flamingo vs. BBGLR', fontsize=15,
                 fontweight='bold', pad=15)
    ax.set_ylim(0, 55)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[i], edgecolor='white')
               for i in range(4)]
    ax.legend(handles, LABELS, loc='upper left', framealpha=0.9, fontsize=10.5,
              edgecolor='#cccccc')

    plt.tight_layout()
    return fig


# ── Chart 2: Total Time Breakdown ─────────────────────────────────────────
def plot_chart2(ns, data):
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    # 3 components: client computation, network wait, server computation
    COLORS_3 = [MORANDI['lavender'], MORANDI['steel'], MORANDI['rose']]
    LABELS_3 = ['Client Computation', 'Pure Network Wait', 'Server Computation']

    fig, ax = plt.subplots(figsize=(14, 7.5))
    bar_width = 0.32
    group_gap = 1.2

    for gi, n in enumerate(ns):
        for pi, (proto_key, proto_label) in enumerate([('flamingo', 'Flamingo'), ('google', 'BBGLR')]):
            d = data[proto_key][n]
            vals = [d['client_comp'], d['network_wait'], d['server_comp']]
            total = sum(vals)
            srv_total_wait = vals[0] + vals[1]  # client comp + network wait

            x = gi * group_gap + pi * (bar_width + 0.05)

            bottom = 0
            for ci in range(3):
                ax.bar(x, vals[ci], bar_width, bottom=bottom,
                       color=COLORS_3[ci], edgecolor='white', linewidth=0.5)
                bottom += vals[ci]

            # Total label on top
            ax.text(x, total + 0.5, f'{total:.1f}s',
                    ha='center', va='bottom', fontweight='bold', fontsize=11)

            # Side bracket for "Server Total Wait" (client comp + network wait)
            bracket_x = x + bar_width / 2 + 0.03
            bracket_bottom = 0
            bracket_top = srv_total_wait
            # Draw bracket line
            ax.plot([bracket_x, bracket_x + 0.02, bracket_x + 0.02, bracket_x],
                    [bracket_bottom, bracket_bottom, bracket_top, bracket_top],
                    color='#888888', linewidth=0.8, clip_on=False)
            # Label: server total wait value
            ax.text(bracket_x + 0.04, (bracket_bottom + bracket_top) / 2,
                    f'wait\n{srv_total_wait:.1f}s',
                    ha='left', va='center', fontsize=7.5, color='#666666')

            # Label server computation on top segment if visible
            if vals[2] > total * 0.04:
                ax.text(x, srv_total_wait + vals[2] / 2,
                        f'srv\n{vals[2]:.1f}s',
                        ha='center', va='center', fontsize=7.5, color='#444444')

        # Speedup ratios
        fl_d = data['flamingo'][n]
        gm_d = data['google'][n]
        fl_total = fl_d['client_comp'] + fl_d['network_wait'] + fl_d['server_comp']
        gm_total = gm_d['client_comp'] + gm_d['network_wait'] + gm_d['server_comp']

        ratio = gm_total / fl_total if fl_total > 0 else 0
        x_fl = gi * group_gap
        x_gm = gi * group_gap + bar_width + 0.05
        mid_x = (x_fl + x_gm) / 2
        ax.text(mid_x, max(fl_total, gm_total) + 3.0, f'{ratio:.1f}x total',
                ha='center', va='bottom', fontsize=10, color='#666666',
                fontstyle='italic')

        # Also show wait-only ratio
        fl_wait = fl_d['client_comp'] + fl_d['network_wait']
        gm_wait = gm_d['client_comp'] + gm_d['network_wait']
        wait_ratio = gm_wait / fl_wait if fl_wait > 0 else 0
        ax.text(mid_x, max(fl_total, gm_total) + 5.5, f'{wait_ratio:.1f}x wait',
                ha='center', va='bottom', fontsize=9, color='#999999')

    # X-axis
    tick_positions, tick_labels = [], []
    for gi, n in enumerate(ns):
        x_fl = gi * group_gap
        x_gm = gi * group_gap + bar_width + 0.05
        tick_positions.extend([x_fl, x_gm])
        tick_labels.extend(['Flamingo', 'BBGLR'])

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=11)

    for gi, n in enumerate(ns):
        mid = gi * group_gap + (bar_width + 0.05) / 2
        ax.text(mid, -5.5, f'n = {n}', ha='center', fontsize=12, fontweight='bold')

    ax.set_ylabel('Time per Iteration (s)', fontsize=13)
    ax.set_xlabel('Number of Clients (n)', fontsize=13, labelpad=30)
    ax.set_title('Per-Iteration Time Breakdown: Flamingo vs. BBGLR', fontsize=15,
                 fontweight='bold', pad=15)

    # Dynamic y-limit
    max_val = max(
        data['google'][n]['client_comp'] + data['google'][n]['network_wait'] + data['google'][n]['server_comp']
        for n in ns
    )
    ax.set_ylim(0, max_val * 1.35)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLORS_3[i], edgecolor='white')
               for i in range(3)]
    ax.legend(handles, LABELS_3, loc='upper left', framealpha=0.9, fontsize=10.5,
              edgecolor='#cccccc')

    plt.tight_layout()
    return fig


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    base = os.path.dirname(os.path.abspath(__file__))
    ns, data = gather_data()

    # Print summary table
    print(f"{'Proto':<10} {'n':>4}  {'Client':>8} {'Network':>8} {'SrvComp':>8} {'Total':>8}  {'SrvWait':>8}")
    print("-" * 70)
    for proto_key, proto_label in [('flamingo', 'Flamingo'), ('google', 'BBGLR')]:
        for n in ns:
            d = data[proto_key][n]
            total = d['client_comp'] + d['network_wait'] + d['server_comp']
            srv_wait = d['client_comp'] + d['network_wait']
            print(f"{proto_label:<10} {n:>4}  {d['client_comp']:>8.3f} {d['network_wait']:>8.3f} "
                  f"{d['server_comp']:>8.3f} {total:>8.3f}  {srv_wait:>8.3f}")
    print()

    # Chart 1
    fig1 = plot_chart1(ns, data)
    fig1.savefig(os.path.join(base, 'pure_network_wait.png'), dpi=200, bbox_inches='tight')
    fig1.savefig(os.path.join(base, 'pure_network_wait.pdf'), bbox_inches='tight')
    print(f'Saved: pure_network_wait.png/pdf')
    plt.close(fig1)

    # Chart 2
    fig2 = plot_chart2(ns, data)
    fig2.savefig(os.path.join(base, 'total_time_breakdown.png'), dpi=200, bbox_inches='tight')
    fig2.savefig(os.path.join(base, 'total_time_breakdown.pdf'), bbox_inches='tight')
    print(f'Saved: total_time_breakdown.png/pdf')
    plt.close(fig2)


if __name__ == '__main__':
    main()
