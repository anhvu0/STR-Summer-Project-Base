from types import SimpleNamespace

from core.route_candidate_generator import (
    ROUTE_FEATURE_DIM,
    RouteCandidateGenerator,
    filter_candidates_by_first_edges,
    pack_route_candidate_features,
)


def test_route_candidates_include_density_aware_features_and_pack_cleanly():
    connection_info = SimpleNamespace(
        edge_list=["A", "B", "C", "D", "E"],
        outgoing_edges_dict={
            "A": {0: "B", 1: "C"},
            "B": {0: "D"},
            "C": {0: "D"},
            "D": {0: "E"},
            "E": {},
        },
        edge_length_dict={
            "A": 5.0,
            "B": 10.0,
            "C": 20.0,
            "D": 10.0,
            "E": 10.0,
        },
    )
    generator = RouteCandidateGenerator(
        connection_info=connection_info,
        k_routes=4,
        oversample=6,
        max_route_length_m=100.0,
    )
    densities = {"B": 1.0}
    candidates = generator.get_candidates(
        "A",
        "E",
        lambda edge_id: densities.get(edge_id, 0.0),
    )

    assert len(candidates) >= 2
    assert any(candidate.route_edges[:2] == ["A", "C"] for candidate in candidates)
    assert all(candidate.features.shape == (ROUTE_FEATURE_DIM,) for candidate in candidates)

    candidates_with_previous = generator.get_candidates(
        "A",
        "E",
        lambda edge_id: densities.get(edge_id, 0.0),
        prev_route_edges=["A", "B", "D", "E"],
    )
    assert any(
        candidate.route_edges[:2] == ["A", "C"] and candidate.features[6] > 0.0
        for candidate in candidates_with_previous
    )

    packed = pack_route_candidate_features(candidates, route_k=4)
    assert packed.shape == (4 * ROUTE_FEATURE_DIM,)

    filtered = filter_candidates_by_first_edges(candidates, {"C"})
    assert filtered
    assert all(candidate.route_edges[1] == "C" for candidate in filtered)
