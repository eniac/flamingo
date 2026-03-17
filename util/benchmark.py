"""Benchmark collection and reporting for Flamingo protocol."""

import time
import json
import pandas as pd
from collections import defaultdict


class BenchmarkCollector:
    """Collects per-step timing, communication sizes, and other metrics.

    Usage:
        bench = BenchmarkCollector()
        bench.start_timer("report", "ecc_mult")
        ... do work ...
        bench.stop_timer("report", "ecc_mult")
        bench.record_comm("client_to_server", num_bytes)
        bench.print_summary()
    """

    def __init__(self):
        # {iteration -> {step -> {sub_operation -> total_seconds}}}
        self.timers = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        # {iteration -> {step -> {sub_operation -> count}}}
        self.timer_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        # Active timers: (iteration, step, sub_op) -> start_time
        self._active = {}
        # Communication: {iteration -> {direction -> total_bytes}}
        self.comm = defaultdict(lambda: defaultdict(int))
        # Communication counts
        self.comm_counts = defaultdict(lambda: defaultdict(int))
        # Client-level timers: {iteration -> {step -> [per_client_seconds]}}
        self.client_timers = defaultdict(lambda: defaultdict(list))
        # Misc metrics
        self.metrics = defaultdict(dict)
        self.current_iteration = 1
        # Timeline: {iteration -> {event_name -> simtime (pd.Timestamp)}}
        self.timeline = defaultdict(dict)
        # Message arrival times: {iteration -> {round_name -> [simtimes]}}
        self.arrivals = defaultdict(lambda: defaultdict(list))
        # Expected message counts: {(iteration, round_name) -> expected_total}
        self._expected_counts = {}

    def set_iteration(self, iteration):
        self.current_iteration = iteration

    def start_timer(self, step, sub_op="total"):
        key = (self.current_iteration, step, sub_op)
        self._active[key] = time.perf_counter()

    def stop_timer(self, step, sub_op="total"):
        key = (self.current_iteration, step, sub_op)
        if key in self._active:
            elapsed = time.perf_counter() - self._active.pop(key)
            self.timers[self.current_iteration][step][sub_op] += elapsed
            self.timer_counts[self.current_iteration][step][sub_op] += 1
            return elapsed
        return 0.0

    def record_client_time(self, step, seconds):
        self.client_timers[self.current_iteration][step].append(seconds)

    def record_comm(self, direction, num_bytes):
        self.comm[self.current_iteration][direction] += num_bytes
        self.comm_counts[self.current_iteration][direction] += 1

    def record_metric(self, name, value):
        self.metrics[self.current_iteration][name] = value

    def record_timeline(self, event_name, simtime):
        """Record a simulation-time event for the timeline view."""
        self.timeline[self.current_iteration][event_name] = simtime

    def record_arrival(self, round_name, simtime, expected_total=None):
        """Record when a client message arrives at the server (simulation time).

        Args:
            round_name: name of the round (e.g., "report_vectors")
            simtime: simulation time of arrival
            expected_total: how many messages are expected (e.g., num_clients)
        """
        self.arrivals[self.current_iteration][round_name].append(simtime)
        if expected_total is not None:
            self._expected_counts[(self.current_iteration, round_name)] = expected_total

    def print_summary(self, num_clients=0, committee_size=0):
        print("\n" + "=" * 72)
        print("BENCHMARK SUMMARY")
        print("=" * 72)

        iterations = sorted(self.timers.keys())
        if not iterations:
            print("No benchmark data collected.")
            return

        for it in iterations:
            print(f"\n--- Iteration {it} ---")

            # Metrics
            if it in self.metrics:
                for name, val in self.metrics[it].items():
                    print(f"  {name}: {val}")

            # Server timings
            print(f"\n  Server computation time:")
            steps = self.timers[it]
            for step in ["report", "crosscheck", "reconstruction"]:
                if step in steps:
                    total = steps[step].get("total", 0)
                    print(f"    {step:20s}  {total:.4f}s")
                    for sub_op, t in sorted(steps[step].items()):
                        if sub_op != "total":
                            count = self.timer_counts[it][step][sub_op]
                            print(f"      {sub_op:18s}  {t:.4f}s  ({count} calls)")

            # Client timings
            if it in self.client_timers:
                print(f"\n  Client computation time (per client):")
                for step in ["report", "crosscheck", "reconstruction"]:
                    times = self.client_timers[it].get(step, [])
                    if times:
                        avg_t = sum(times) / len(times)
                        max_t = max(times)
                        print(f"    {step:20s}  avg={avg_t:.4f}s  max={max_t:.4f}s  ({len(times)} clients)")

            # Communication
            if it in self.comm:
                print(f"\n  Communication (bytes):")
                for direction in sorted(self.comm[it].keys()):
                    total_bytes = self.comm[it][direction]
                    count = self.comm_counts[it][direction]
                    avg = total_bytes / count if count else 0
                    print(f"    {direction:30s}  total={total_bytes:>12,}  msgs={count:>4}  avg={avg:>10,.0f}")

        # Cross-iteration summary
        if len(iterations) > 1:
            print(f"\n--- Average across {len(iterations)} iterations ---")
            for step in ["report", "crosscheck", "reconstruction"]:
                totals = [self.timers[it][step].get("total", 0) for it in iterations]
                if any(t > 0 for t in totals):
                    print(f"  Server {step:20s}  {sum(totals)/len(totals):.4f}s")

            # Average client times
            for step in ["report", "crosscheck", "reconstruction"]:
                all_times = []
                for it in iterations:
                    all_times.extend(self.client_timers[it].get(step, []))
                if all_times:
                    print(f"  Client {step:20s}  avg={sum(all_times)/len(all_times):.4f}s")

            # Average comm
            all_dirs = set()
            for it in iterations:
                all_dirs.update(self.comm[it].keys())
            for direction in sorted(all_dirs):
                total = sum(self.comm[it].get(direction, 0) for it in iterations)
                count = sum(self.comm_counts[it].get(direction, 0) for it in iterations)
                avg_per_iter = total / len(iterations)
                print(f"  Comm {direction:28s}  {avg_per_iter:>12,.0f} bytes/iter")

        print("=" * 72)

        # Print timeline view
        self.print_timeline()

    def print_timeline(self):
        """Print a chronological timeline from the server's perspective."""
        if not self.timeline and not self.arrivals:
            return

        print()
        print("=" * 80)
        print("TIMELINE (server view, simulation time)")
        print("=" * 80)

        iterations = sorted(set(list(self.timeline.keys()) + list(self.arrivals.keys())))

        for it in iterations:
            events = self.timeline.get(it, {})
            arrival_data = self.arrivals.get(it, {})

            iter_start = events.get("iter_start")
            if iter_start is None:
                continue

            print(f"\n--- Iteration {it} (start: {iter_start}) ---")

            # Build a single chronological list of all events
            timeline_entries = []  # [(simtime, description)]

            # Add server processing events
            for name, t in events.items():
                if name == "iter_start":
                    continue
                # Convert internal names to readable descriptions
                desc = name.replace("_", " ")
                timeline_entries.append((t, desc))

            # Add arrival count milestones as events
            for round_name, times in arrival_data.items():
                times = sorted(times)
                n_arrived = len(times)
                if n_arrived == 0:
                    continue
                n_expected = self._expected_counts.get((it, round_name), n_arrived)
                # Show when 50%, 90%, and all arrived messages reached the server
                for target_frac, label in [(0.50, "50%"), (0.90, "90%")]:
                    target_count = int(n_expected * target_frac)
                    if target_count <= n_arrived and target_count > 0:
                        timeline_entries.append((
                            times[target_count - 1],
                            f"server receives {label} of {round_name} ({target_count}/{n_expected} msgs)"
                        ))
                # Always show the last arrived message
                timeline_entries.append((
                    times[-1],
                    f"server receives last {round_name} ({n_arrived}/{n_expected} msgs)"
                ))

            # Sort by time and print
            timeline_entries.sort(key=lambda x: x[0])
            for t, desc in timeline_entries:
                delta = t - iter_start
                # Format delta as seconds for readability
                delta_s = delta.total_seconds()
                print(f"  +{delta_s:>10.3f}s  {desc}")

        print("=" * 80)

    def save_timeline_csv(self, filepath="timeline.csv"):
        """Save the timeline to a CSV file for later analysis.

        Columns: iteration, simtime, offset_s, event
        """
        import csv
        iterations = sorted(set(list(self.timeline.keys()) + list(self.arrivals.keys())))
        if not iterations:
            return

        rows = []
        for it in iterations:
            events = self.timeline.get(it, {})
            arrival_data = self.arrivals.get(it, {})
            iter_start = events.get("iter_start")
            if iter_start is None:
                continue

            # Server processing events
            for name, t in events.items():
                if name == "iter_start":
                    continue
                delta_s = (t - iter_start).total_seconds()
                rows.append((it, str(t), delta_s, name))

            # Arrival milestones
            for round_name, times in arrival_data.items():
                times = sorted(times)
                n_arrived = len(times)
                if n_arrived == 0:
                    continue
                n_expected = self._expected_counts.get((it, round_name), n_arrived)
                for target_frac, label in [(0.50, "50%"), (0.90, "90%")]:
                    target_count = int(n_expected * target_frac)
                    if target_count <= n_arrived and target_count > 0:
                        t = times[target_count - 1]
                        delta_s = (t - iter_start).total_seconds()
                        rows.append((it, str(t), delta_s,
                                     f"server receives {label} of {round_name} ({target_count}/{n_expected})"))
                # Last arrived
                t = times[-1]
                delta_s = (t - iter_start).total_seconds()
                rows.append((it, str(t), delta_s,
                             f"server receives last {round_name} ({n_arrived}/{n_expected})"))

        # Sort by iteration then offset
        rows.sort(key=lambda r: (r[0], r[2]))

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "simtime", "offset_s", "event"])
            writer.writerows(rows)

        print(f"\nTimeline saved to {filepath}")

    def to_dict(self):
        """Export all data as a plain dict for JSON serialization."""
        return {
            "timers": {str(k): dict(v) for k, v in self.timers.items()},
            "comm": {str(k): dict(v) for k, v in self.comm.items()},
            "client_timers": {str(k): dict(v) for k, v in self.client_timers.items()},
            "metrics": dict(self.metrics),
        }

    def save_json(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


# Global singleton - shared across all agents in the simulation
bench = BenchmarkCollector()
