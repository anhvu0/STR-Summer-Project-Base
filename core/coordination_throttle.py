"""Saturation-aware detour coordination: Layer A throttle + Layer B reservation field.

Two cooperating mechanisms that address the price-of-anarchy / coordination failure
where MAPPO's "selfless" detours overload the alternatives at saturation and end up
worse than everyone taking the shortest path (see docs/coordination_throttle.md and the
selfless-routing diagnosis). Both are deliberately small, deterministic, and reuse
signals the pipeline already computes.

Layer A -- spare-capacity veto (``detour_should_fallback``)
    A deterministic guardrail applied AFTER the policy picks a route. If the chosen
    route is a detour (candidate index != 0) onto an alternative that is itself near
    capacity AND that detour does not actually relieve congestion relative to the
    shortest-path baseline (its blended relief is within snapshot noise), the choice is
    reverted to the shortest-path baseline (candidate index 0). This catches pointless /
    pile-on detours onto roads that are already full without buying anything.

    It deliberately does NOT veto merely because the network as a whole is saturated. The
    original design added a "network saturated -> shortest path is optimal (PoA ~= 1)"
    trigger, but the Phase 0 forced-detour probe measured the opposite in this regime:
    relieving detours help MOST under saturation (e.g. a catastrophic-congestion seed went
    1236s -> 636s once detours were allowed), and that blanket trigger vetoed ~100% of
    detours at 450/150, nullifying the learned policy (greedy == stochastic byte-for-byte).
    So the veto now keys on whether THIS detour relieves vs the baseline, not on how busy
    the network is.

Layer B -- anticipatory reservation field (``ReservationField``)
    When a vehicle commits to a route, its leading edges are "booked" in a decaying
    edge-level field. The route-candidate generator scores against an EFFECTIVE density
    (live count + reservations), so a vehicle deciding later in the same window sees an
    alternative's relief already eroded by the detours earlier vehicles committed to.
    Simultaneous independent picks become a damped sequential best-response instead of a
    pile-on. Reservations decay each step as the booked vehicles enter the live density.

Layer A catches "the alternative is already full". Layer B prevents "the alternative
will be full because we are all about to choose it". They compose: with Layer B on, the
effective density that Layer A inspects already reflects in-window bookings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np


# Relief blend mirrors _route_candidate_balance_components in rl_training_pipeline.py:
# weight the mean-density relief above the first-edge relief.
_RELIEF_MEAN_WEIGHT = 0.65
_RELIEF_FIRST_WEIGHT = 0.35

# Indices into the RouteCandidate feature vector (see route_candidate_generator.py).
_FEAT_MAX_DENSITY = 3
_FEAT_FIRST_EDGE_DENSITY = 4
_FEAT_MEAN_RELIEF = 9
_FEAT_FIRST_RELIEF = 10


def route_relief(features: Sequence[float]) -> float:
    """Blended density relief of a candidate vs. the shortest-path baseline."""
    feats = np.asarray(features, dtype=np.float32).reshape(-1)
    return (
        _RELIEF_MEAN_WEIGHT * float(feats[_FEAT_MEAN_RELIEF])
        + _RELIEF_FIRST_WEIGHT * float(feats[_FEAT_FIRST_RELIEF])
    )


@dataclass(frozen=True)
class DetourThrottleConfig:
    """Layer A thresholds. Defaults are conservative -- fire only in clear jams."""

    enabled: bool = True
    # Alternative counts as near-capacity when its max OR first-edge density >= this.
    jam_density: float = 0.50
    # A relief whose magnitude is below this is treated as snapshot noise; a near-capacity
    # detour whose blended relief vs the baseline is under this is vetoed as pointless.
    relief_deadband: float = 0.01
    # DEPRECATED / unused: the old "network saturated -> veto every detour" trigger. Kept
    # for backward-compatible construction only; the Phase 0 probe showed detours help most
    # under saturation, so the veto no longer keys on network-wide density. See module docstring.
    network_p95_trigger: float = 0.30


def detour_should_fallback(
    chosen_idx: int,
    candidate_features: Sequence[Sequence[float]],
    network_density_p95: float,
    config: DetourThrottleConfig,
) -> bool:
    """Layer A. True if the chosen detour should revert to the shortest-path baseline.

    A no-op unless the choice is a detour (idx != 0) onto an alternative that is itself
    near capacity. The veto then triggers only when that near-capacity detour does not
    actually relieve congestion relative to the shortest-path baseline (its blended relief
    is within snapshot noise) -- i.e. a pointless / pile-on detour. It no longer triggers
    on network-wide saturation: the Phase 0 probe showed relieving detours help most under
    saturation, and the old saturation trigger vetoed essentially every detour, hiding the
    learned policy. ``network_density_p95`` is retained in the signature for compatibility
    but no longer gates the veto.
    """
    if not config.enabled or int(chosen_idx) == 0:
        return False
    if int(chosen_idx) < 0 or int(chosen_idx) >= len(candidate_features):
        return False
    feats = np.asarray(candidate_features[int(chosen_idx)], dtype=np.float32).reshape(-1)
    alt_saturated = (
        float(feats[_FEAT_MAX_DENSITY]) >= config.jam_density
        or float(feats[_FEAT_FIRST_EDGE_DENSITY]) >= config.jam_density
    )
    if not alt_saturated:
        return False
    illusory_relief = route_relief(feats) < float(config.relief_deadband)
    return bool(illusory_relief)


@dataclass(frozen=True)
class ReservationFieldConfig:
    """Layer B knobs governing how committed routes inflate the effective density."""

    enabled: bool = True
    # Number of leading edges of a committed route to book (skips the current edge).
    route_horizon: int = 4
    # Weight of the i-th booked edge along the route = route_decay ** i.
    route_decay: float = 0.7
    # Per-step multiplicative decay of every reservation (books fade as the vehicle
    # enters the live density).
    time_decay: float = 0.85
    # Scale of a reservation's contribution to the effective density.
    density_weight: float = 0.5
    # Drop reservations that fall below this after a decay step.
    min_keep: float = 0.02


class ReservationField:
    """Layer B. Decaying edge-level booking field shared across a decision window."""

    def __init__(self, config: ReservationFieldConfig = None):
        self.config = config or ReservationFieldConfig()
        self._reservations: Dict[str, float] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def seed_route(self, route_edges: Sequence[str], weight: float = 1.0) -> int:
        """Book the leading edges of a committed route. Returns edges booked."""
        if not self.config.enabled or not route_edges:
            return 0
        horizon = max(int(self.config.route_horizon), 0)
        decay = float(self.config.route_decay)
        booked = 0
        # Skip index 0: the current edge is already reflected in the live density.
        for offset, edge_id in enumerate(list(route_edges)[1:1 + horizon]):
            self._reservations[edge_id] = (
                self._reservations.get(edge_id, 0.0) + float(weight) * (decay ** offset)
            )
            booked += 1
        return booked

    def reserved_count(self, edge_id: str) -> float:
        return float(self._reservations.get(edge_id, 0.0))

    def density_bonus(self, edge_id: str, lane_meters: float, density_scale: float) -> float:
        """Reservation contribution to ``edge_id`` in the same units as edge density."""
        reserved = self._reservations.get(edge_id, 0.0)
        if reserved <= 0.0 or not self.config.enabled:
            return 0.0
        return float(self.config.density_weight) * (
            (float(reserved) * float(density_scale)) / max(float(lane_meters), 5.0)
        )

    def decay(self) -> None:
        """Fade every reservation by ``time_decay`` and prune the negligible ones."""
        if not self._reservations:
            return
        factor = float(self.config.time_decay)
        keep = float(self.config.min_keep)
        self._reservations = {
            edge_id: value * factor
            for edge_id, value in self._reservations.items()
            if value * factor > keep
        }

    def clear(self) -> None:
        self._reservations.clear()
