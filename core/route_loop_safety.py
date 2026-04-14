"""Loop-safety utilities for proactive routing decisions."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List


@dataclass(frozen=True)
class LoopSignals:
    """Signals describing short-horizon loop risk."""

    aba_bounce: bool
    short_cycle: bool
    dead_end_reentry: bool
    repeat_count: int



def is_aba_bounce(history: Iterable[str]) -> bool:
    seq = list(history)
    return len(seq) >= 3 and seq[-1] == seq[-3] and seq[-2] != seq[-1]



def has_short_cycle_repeat(history: Iterable[str], max_cycle_len: int = 4) -> bool:
    seq = list(history)
    n = len(seq)
    for cycle_len in range(2, min(max_cycle_len, n // 2) + 1):
        if seq[-cycle_len:] == seq[-2 * cycle_len : -cycle_len]:
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
    next_edge: str,
    edge_out_degree: Dict[str, int],
) -> Dict[str, bool]:
    probe = deque(history, maxlen=history.maxlen)
    probe.append(next_edge)
    return {
        "aba_bounce": is_aba_bounce(probe),
        "short_cycle": has_short_cycle_repeat(probe),
        "dead_end_reentry": dead_end_reentry_count(probe, edge_out_degree) > 0,
    }



def would_worsen_distance(current_distance: float, next_distance: float, slack: float) -> bool:
    if current_distance == float("inf"):
        return False
    if next_distance == float("inf"):
        return True
    return (next_distance - current_distance) > slack



def summarize_loop_risk(
    history: Deque[str],
    candidate_edge: str,
    edge_out_degree: Dict[str, int],
) -> LoopSignals:
    """Return compact loop-risk features for candidate action ranking."""

    probe = deque(history, maxlen=history.maxlen)
    probe.append(candidate_edge)
    counts = Counter(probe)
    return LoopSignals(
        aba_bounce=is_aba_bounce(probe),
        short_cycle=has_short_cycle_repeat(probe),
        dead_end_reentry=dead_end_reentry_count(probe, edge_out_degree) > 0,
        repeat_count=max(counts.values()) if counts else 0,
    )
