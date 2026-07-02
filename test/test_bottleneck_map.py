"""Structural guarantees the selfless-routing experiment depends on
(docs/bottleneck_map_design.md). If these fail, retraining on the bottleneck
map is meaningless: the policy either cannot see the fork choice or the
selfish/selfless travel-time gap has been built out of the network.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sumolib

from core.Util import ConnectionInfo
from core.route_candidate_generator import RouteCandidateGenerator

NET_PATH = os.path.join(
    os.path.dirname(__file__), "..", "configurations", "maps", "bottleneck.net.xml"
)


def _net():
    return sumolib.net.readNet(NET_PATH)


def test_sources_and_sink_for_pattern_4():
    net = _net()
    sources = {e.getID() for e in net.getEdges() if not e.getFromNode().getIncoming()}
    sinks = {e.getID() for e in net.getEdges() if not e.getToNode().getOutgoing()}
    assert sources == {"in1", "in2", "in3"}
    assert sinks == {"out"}


def test_bottleneck_is_shortest_path_from_every_source():
    net = _net()
    for src in ("in1", "in2", "in3"):
        path, _ = net.getShortestPath(net.getEdge(src), net.getEdge("out"))
        assert path is not None
        edge_ids = [e.getID() for e in path]
        assert "a1" in edge_ids and "a2" in edge_ids, (
            "selfish shortest path from {} must run through the bottleneck".format(src)
        )


def test_detours_cost_more_but_carry_more_capacity():
    net = _net()
    def path_len(ids):
        return sum(net.getEdge(i).getLength() for i in ids)
    len_a = path_len(["a1", "a2"])
    len_b = path_len(["b1", "b2", "b3"])
    len_c = path_len(["c1", "c2", "c3"])
    assert len_a < len_b < len_c
    # The dilemma scale: detour sacrifice must be tens of seconds, not trivial
    # and not prohibitive, at the 13.89 m/s map speed.
    assert 15.0 < (len_b - len_a) / 13.89 < 60.0
    assert net.getEdge("a1").getLaneNumber() == 1
    assert net.getEdge("b1").getLaneNumber() >= 2
    assert net.getEdge("c1").getLaneNumber() >= 2


def test_fork_offers_all_three_routes_to_the_policy():
    connection_info = ConnectionInfo(NET_PATH)
    generator = RouteCandidateGenerator(connection_info, net=_net())
    allowed_first_edges = {"a1", "b1", "c1"}
    candidates = generator.get_candidates(
        "stage", "out", lambda edge_id: 0.0, allowed_first_edges=allowed_first_edges
    )
    first_edges = {c.route_edges[1] for c in candidates}
    assert allowed_first_edges.issubset(first_edges), (
        "route candidates from the staging edge must cover the bottleneck and both "
        "detours; got first edges {}".format(sorted(first_edges))
    )
