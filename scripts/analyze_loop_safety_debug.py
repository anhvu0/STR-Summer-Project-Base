#!/usr/bin/env python3
"""Compact analyzer for rl_episode_metrics.csv + decision debug CSV."""
import argparse
import csv
from collections import Counter, defaultdict


def _to_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def _truthy(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "y"}


def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser(description="Analyze loop/override behavior from CSV logs")
    p.add_argument("--metrics", required=True, help="Path to rl_episode_metrics.csv")
    p.add_argument("--debug", required=True, help="Path to decision debug CSV")
    p.add_argument("--loop-threshold", type=float, default=0.75, help="Loop-events threshold for heavy episodes")
    args = p.parse_args()

    metrics_rows = load_rows(args.metrics)
    debug_rows = load_rows(args.debug)

    override_cause_counts = Counter()
    prefilter_signal_counts = Counter()
    reward_by_override = defaultdict(list)
    finalized_long_horizon = 0
    finalized_revisit_without_progress = 0
    heavy_episodes = set()

    for row in metrics_rows:
        try:
            if _to_float(row.get("loop_events", 0.0)) >= args.loop_threshold:
                heavy_episodes.add(str(row.get("episode", "")))
        except Exception:
            continue

    for row in debug_rows:
        cause = (row.get("override_cause") or "").strip()
        override_type = (row.get("override_type") or "").strip()
        reward = _to_float(row.get("reward", ""), default=0.0)

        if cause:
            override_cause_counts[cause] += 1
        if override_type:
            reward_by_override[override_type].append(reward)

        for signal_key in [
            "prefilter_short_cycle",
            "prefilter_aba_bounce",
            "prefilter_dead_end_reentry",
            "prefilter_long_horizon_loop",
            "prefilter_revisit_without_progress",
            "prefilter_trap_like_reversal",
            "prefilter_distance_worsen",
            "prefilter_distance_worsen_severe",
        ]:
            if _truthy(row.get(signal_key, 0)):
                prefilter_signal_counts[signal_key] += 1

        if _truthy(row.get("finalized", 0)):
            if _truthy(row.get("prefilter_long_horizon_loop", 0)):
                finalized_long_horizon += 1
            if _truthy(row.get("prefilter_revisit_without_progress", 0)):
                finalized_revisit_without_progress += 1

    heavy_episode_cause_counts = Counter()
    for row in debug_rows:
        if str(row.get("episode", "")) in heavy_episodes:
            cause = (row.get("override_cause") or "").strip()
            if cause:
                heavy_episode_cause_counts[cause] += 1

    print("=== Fallback/override count by override_cause ===")
    for cause, count in override_cause_counts.most_common():
        print(f"{cause}: {count}")

    print("\n=== Prefilter signal counts ===")
    for signal, count in prefilter_signal_counts.most_common():
        print(f"{signal}: {count}")

    print("\n=== Mean reward by override_type ===")
    for override_type, values in sorted(reward_by_override.items()):
        mean_value = (sum(values) / len(values)) if values else 0.0
        print(f"{override_type}: n={len(values)} mean_reward={mean_value:.4f}")

    print("\n=== Finalized decisions with long-horizon risk bits ===")
    print(f"finalized_with_prefilter_long_horizon_loop: {finalized_long_horizon}")
    print(f"finalized_with_prefilter_revisit_without_progress: {finalized_revisit_without_progress}")

    print("\n=== Loop-heavy episode dominance check ===")
    if not heavy_episodes:
        print("No loop-heavy episodes found with current threshold.")
    elif not heavy_episode_cause_counts:
        print("Loop-heavy episodes present, but no override causes were logged.")
    else:
        dominant_cause, dominant_count = heavy_episode_cause_counts.most_common(1)[0]
        total = sum(heavy_episode_cause_counts.values())
        share = dominant_count / max(total, 1)
        print(f"heavy_episodes={len(heavy_episodes)} dominant_cause={dominant_cause} share={share:.2%} total_overrides={total}")


if __name__ == "__main__":
    main()
