from collections import Counter, deque
from typing import Deque, Dict, Iterable, Optional


def is_aba_bounce(history: Iterable[str]) -> bool:
    seq = list(history)
    return len(seq) >= 3 and seq[-1] == seq[-3] and seq[-2] != seq[-1]


def has_short_cycle_repeat(history: Iterable[str], max_cycle_len: int = 4) -> bool:
    seq = list(history)
    n = len(seq)
    for cycle_len in range(2, min(max_cycle_len, n // 2) + 1):
        if seq[-cycle_len:] == seq[-2 * cycle_len:-cycle_len]:
            return True
    return False


def dead_end_reentry_count(
    history: Iterable[str],
    edge_out_degree: Dict[str, int],
    dead_end_max_out: int = 1,
) -> int:
    counts = Counter(history)
    return sum(
        max(count - 1, 0)
        for edge, count in counts.items()
        if edge_out_degree.get(edge, 0) <= dead_end_max_out
    )


def has_long_horizon_revisit(
    history: Iterable[str],
    min_horizon: int = 5,
    max_horizon: int = 12,
    subseq_len: int = 2,
) -> bool:
    seq = list(history)
    n = len(seq)
    if n < (min_horizon + 1):
        return False
    anchor = seq[-1]
    start = max(0, n - max_horizon - 1)
    end = max(0, n - min_horizon)
    if anchor in seq[start:end]:
        return True
    if subseq_len <= 0 or n < (subseq_len + min_horizon):
        return False
    tail = tuple(seq[-subseq_len:])
    for i in range(start, max(start, end - subseq_len + 1)):
        if tuple(seq[i:i + subseq_len]) == tail:
            return True
    return False


def has_revisit_without_progress(
    history: Iterable[str],
    edge_distance_lookup: Optional[Dict[str, float]] = None,
    progress_slack: float = 20.0,
) -> bool:
    if not edge_distance_lookup:
        return False
    seq = list(history)
    if len(seq) < 3:
        return False
    current = seq[-1]
    current_dist = edge_distance_lookup.get(current, float("inf"))
    if current_dist == float("inf"):
        return False
    prior_distances = [
        edge_distance_lookup.get(edge, float("inf"))
        for edge in seq[:-1]
        if edge == current
    ]
    if not prior_distances:
        return False
    best_seen = min(prior_distances)
    return (current_dist - best_seen) >= float(progress_slack)


def transition_signal(
    history: Deque[str],
    current_edge: str,
    edge_out_degree: Dict[str, int],
    edge_distance_lookup: Optional[Dict[str, float]] = None,
    progress_slack: float = 20.0,
) -> Dict[str, bool]:
    probe = deque(history, maxlen=history.maxlen)
    probe.append(current_edge)
    return {
        "aba_bounce": is_aba_bounce(probe),
        "short_cycle": has_short_cycle_repeat(probe),
        "dead_end_reentry": dead_end_reentry_count(probe, edge_out_degree) > 0,
        "long_horizon_loop": has_long_horizon_revisit(probe),
        "revisit_without_progress": has_revisit_without_progress(
            probe,
            edge_distance_lookup=edge_distance_lookup,
            progress_slack=progress_slack,
        ),
    }


def would_worsen_distance(
    current_distance: float,
    next_distance: float,
    slack: float,
) -> bool:
    if current_distance == float("inf"):
        return False
    if next_distance == float("inf"):
        return True
    return (next_distance - current_distance) > slack
