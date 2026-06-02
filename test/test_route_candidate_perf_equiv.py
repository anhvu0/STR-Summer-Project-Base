"""Guards for the bit-identical perf cleanup of the route-candidate / corridor code.

These optimizations (per-call density memoization in ``get_candidates`` and the
``np.clip``/``np.max`` -> scalar ``min``/``max`` substitutions) are intended to be
*behavior preserving*: identical outputs, just less work. The tests below pin the two
properties that make that true so a future edit can't silently change results.
"""
from types import SimpleNamespace

import numpy as np

from core.route_candidate_generator import RouteCandidateGenerator


def _make_generator():
    connection_info = SimpleNamespace(
        edge_list=["A", "B", "C", "D", "E"],
        outgoing_edges_dict={
            "A": {0: "B", 1: "C"},
            "B": {0: "D"},
            "C": {0: "D"},
            "D": {0: "E"},
            "E": {},
        },
        edge_length_dict={"A": 5.0, "B": 10.0, "C": 20.0, "D": 10.0, "E": 10.0},
    )
    return RouteCandidateGenerator(
        connection_info=connection_info, k_routes=4, oversample=6, max_route_length_m=100.0
    )


def test_scalar_clip_matches_numpy_bit_for_bit():
    """min(max(x, lo), hi) must equal float(np.clip(x, lo, hi)) for every scalar we clip."""
    bounds = [(0.0, 1.0), (0.0, 2.0), (-1.0, 1.0)]
    values = [
        -5.0, -1.0, -0.5, -1e-12, 0.0, 1e-12, 0.05, 0.1, 0.5, 0.999999,
        1.0, 1.0000001, 1.5, 2.0, 2.5, 1234.5, 1.0 / 3.0, 2.0 / 3.0,
    ]
    for lo, hi in bounds:
        for x in values:
            assert min(max(x, lo), hi) == float(np.clip(x, lo, hi)), (x, lo, hi)


def test_scalar_max_matches_numpy_bit_for_bit():
    """float(max(seq)) must equal float(np.max(seq)) for the python-float density lists."""
    for seq in ([0.0], [0.1, 0.1, 0.1], [0.0, 0.5, 0.25, 0.999], [1.0 / 3.0, 2.0 / 3.0, 0.2]):
        assert float(max(seq)) == float(np.max(seq)), seq


def test_get_candidates_is_deterministic_under_memoization():
    """Repeated identical calls return identical features (the memo is a pure cache)."""
    generator = _make_generator()
    densities = {"B": 1.0, "C": 0.3}
    density_fn = lambda edge_id: densities.get(edge_id, 0.0)

    first = generator.get_candidates("A", "E", density_fn)
    second = generator.get_candidates("A", "E", density_fn)

    assert [c.route_edges for c in first] == [c.route_edges for c in second]
    for a, b in zip(first, second):
        # Bit-for-bit identical feature vectors across runs.
        assert np.array_equal(a.features, b.features)


def test_density_fn_queried_at_most_once_per_edge_per_call():
    """The per-call memo collapses the many repeated density reads to one per edge.

    Includes an edge whose density is exactly 0.0 to confirm the sentinel (not a
    truthiness check) drives the cache, so legitimate zeros are not re-fetched.
    """
    generator = _make_generator()
    densities = {"B": 1.0, "C": 0.0, "D": 0.5}  # C is a real 0.0 reading
    call_counts: dict = {}

    def counting_density_fn(edge_id):
        call_counts[edge_id] = call_counts.get(edge_id, 0) + 1
        return densities.get(edge_id, 0.0)

    generator.get_candidates("A", "E", counting_density_fn)

    assert call_counts, "expected the generator to read at least one edge density"
    assert max(call_counts.values()) == 1, f"edge density re-fetched within one call: {call_counts}"
