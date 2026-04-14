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


def transition_signal(
    history: Deque[str],
    current_edge: str,
    edge_out_degree: Dict[str, int],
) -> Dict[str, bool]:
    probe = deque(history, maxlen=history.maxlen)
    probe.append(current_edge)
    return {
        "aba_bounce": is_aba_bounce(probe),
        "short_cycle": has_short_cycle_repeat(probe),
        "dead_end_reentry": dead_end_reentry_count(probe, edge_out_degree) > 0,
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


def score_transition_risk(
    history: Deque[str],
    current_edge: str,
    next_edge: str,
    destination: str,
    current_distance: float,
    next_distance: float,
    edge_out_degree: Dict[str, int],
    distance_slack: float = 30.0,
) -> Dict[str, object]:
    """
    Shared transition risk scoring for training + inference.
    Returns a compact score card that can be used for hard filtering or Q down-ranking.
    """
    signals = transition_signal(history, next_edge, edge_out_degree=edge_out_degree)
    distance_worsen = would_worsen_distance(current_distance, next_distance, slack=distance_slack)
    trap_like = (
        next_edge != destination
        and edge_out_degree.get(next_edge, 0) <= 1
        and len(history) > 0
        and history[-1] == current_edge
    )

    score = 0.0
    reasons = []
    if signals["short_cycle"]:
        score += 5.0
        reasons.append("short_cycle")
    if signals["aba_bounce"]:
        score += 5.0
        reasons.append("aba_bounce")
    if signals["dead_end_reentry"]:
        score += 3.0
        reasons.append("dead_end_reentry")
    if distance_worsen:
        score += 2.0
        reasons.append("distance_worsen")
    if trap_like:
        score += 4.0
        reasons.append("trap_like")

    return {
        "score": float(score),
        "reasons": reasons,
        "short_cycle": bool(signals["short_cycle"]),
        "aba_bounce": bool(signals["aba_bounce"]),
        "dead_end_reentry": bool(signals["dead_end_reentry"]),
        "distance_worsen": bool(distance_worsen),
        "trap_like": bool(trap_like),
    }
