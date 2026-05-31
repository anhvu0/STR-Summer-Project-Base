"""Unit tests for the saturation-aware detour coordination layers."""

import numpy as np

from core.coordination_throttle import (
    DetourThrottleConfig,
    ReservationField,
    ReservationFieldConfig,
    detour_should_fallback,
    route_relief,
)


def _features(max_density=0.0, first_edge_density=0.0, mean_relief=0.0, first_relief=0.0):
    """Build an 11-d RouteCandidate feature vector with the fields the gate reads."""
    feats = np.zeros(11, dtype=np.float32)
    feats[3] = max_density
    feats[4] = first_edge_density
    feats[9] = mean_relief
    feats[10] = first_relief
    return feats


# --------------------------------------------------------------------------- #
# Layer A -- detour_should_fallback
# --------------------------------------------------------------------------- #

def test_relief_blend_weights():
    assert route_relief(_features(mean_relief=1.0, first_relief=0.0)) == np.float32(0.65)
    assert route_relief(_features(mean_relief=0.0, first_relief=1.0)) == np.float32(0.35)


def test_baseline_choice_is_never_vetoed():
    cfg = DetourThrottleConfig()
    cands = [_features(max_density=0.9), _features(max_density=0.9)]
    # idx 0 is the shortest-path baseline -- always kept.
    assert detour_should_fallback(0, cands, network_density_p95=0.9, config=cfg) is False


def test_disabled_config_never_vetoes():
    cfg = DetourThrottleConfig(enabled=False)
    cands = [_features(), _features(max_density=0.9, first_edge_density=0.9)]
    assert detour_should_fallback(1, cands, network_density_p95=0.9, config=cfg) is False


def test_detour_with_slack_is_kept():
    cfg = DetourThrottleConfig()
    # Alternative is well below the jam threshold -> genuine spare capacity.
    cands = [_features(), _features(max_density=0.2, first_edge_density=0.2, mean_relief=0.3)]
    assert detour_should_fallback(1, cands, network_density_p95=0.9, config=cfg) is False


def test_saturated_alt_vetoed_when_network_saturated():
    cfg = DetourThrottleConfig()
    # Alt near capacity AND network saturated -> revert even though relief looks positive.
    cands = [_features(), _features(max_density=0.8, mean_relief=0.5, first_relief=0.5)]
    assert detour_should_fallback(1, cands, network_density_p95=0.4, config=cfg) is True


def test_saturated_alt_vetoed_on_illusory_relief():
    cfg = DetourThrottleConfig()
    # Network not saturated, but the relief is within the noise deadband -> revert.
    cands = [_features(), _features(first_edge_density=0.6, mean_relief=0.0, first_relief=0.0)]
    assert detour_should_fallback(1, cands, network_density_p95=0.1, config=cfg) is True


def test_saturated_alt_with_genuine_relief_and_loose_network_is_kept():
    cfg = DetourThrottleConfig()
    # Alt near capacity but network NOT saturated and relief is real -> keep the detour.
    cands = [_features(), _features(max_density=0.6, mean_relief=0.5, first_relief=0.5)]
    assert detour_should_fallback(1, cands, network_density_p95=0.1, config=cfg) is False


def test_out_of_range_index_is_safe():
    cfg = DetourThrottleConfig()
    cands = [_features(), _features(max_density=0.9)]
    assert detour_should_fallback(5, cands, network_density_p95=0.9, config=cfg) is False


# --------------------------------------------------------------------------- #
# Layer B -- ReservationField
# --------------------------------------------------------------------------- #

def test_seed_route_skips_current_edge_and_decays_along_route():
    field = ReservationField(ReservationFieldConfig(route_horizon=4, route_decay=0.7))
    booked = field.seed_route(["e0", "e1", "e2", "e3", "e4", "e5"])
    assert booked == 4
    assert field.reserved_count("e0") == 0.0          # current edge skipped
    assert field.reserved_count("e1") == 1.0          # 0.7**0
    assert abs(field.reserved_count("e2") - 0.7) < 1e-6
    assert abs(field.reserved_count("e3") - 0.49) < 1e-6
    assert abs(field.reserved_count("e4") - 0.343) < 1e-6
    assert field.reserved_count("e5") == 0.0          # beyond the horizon


def test_seed_route_accumulates_across_vehicles():
    field = ReservationField(ReservationFieldConfig(route_horizon=2, route_decay=1.0))
    field.seed_route(["a", "b", "c"])
    field.seed_route(["x", "b", "c"])
    assert field.reserved_count("b") == 2.0           # two vehicles booked edge b


def test_density_bonus_units_and_weight():
    field = ReservationField(ReservationFieldConfig(route_horizon=1, density_weight=0.5))
    field.seed_route(["cur", "next"])                 # reserved_count("next") == 1.0
    # bonus = density_weight * reserved * density_scale / lane_meters
    bonus = field.density_bonus("next", lane_meters=1000.0, density_scale=100.0)
    assert abs(bonus - 0.05) < 1e-6
    assert field.density_bonus("unbooked", 1000.0, 100.0) == 0.0


def test_decay_multiplies_and_prunes():
    field = ReservationField(
        ReservationFieldConfig(route_horizon=4, route_decay=0.7, time_decay=0.85, min_keep=0.5)
    )
    field.seed_route(["e0", "e1", "e2", "e3", "e4"])
    field.decay()
    # e1: 1.0*0.85=0.85 kept; e2: 0.7*0.85=0.595 kept; e3/e4 fall below min_keep -> pruned.
    assert abs(field.reserved_count("e1") - 0.85) < 1e-6
    assert abs(field.reserved_count("e2") - 0.595) < 1e-6
    assert field.reserved_count("e3") == 0.0
    assert field.reserved_count("e4") == 0.0


def test_disabled_field_is_inert():
    field = ReservationField(ReservationFieldConfig(enabled=False))
    assert field.seed_route(["a", "b", "c"]) == 0
    assert field.density_bonus("b", 1000.0, 100.0) == 0.0


def test_clear_empties_field():
    field = ReservationField(ReservationFieldConfig())
    field.seed_route(["a", "b", "c"])
    field.clear()
    assert field.reserved_count("b") == 0.0
