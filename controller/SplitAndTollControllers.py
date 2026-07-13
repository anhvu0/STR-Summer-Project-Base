"""R4 learning-necessity baselines for the chained-Braess map (REVISION_PLAN R4).

Three non-learning controllers run through the same StrSumo harness as MAPPO:

  FixedSplitPolicy   : deploys a FIXED per-diamond {braess,up,down} assignment
                       (default: R2's grid-search best, pb=0 with a 45/55
                       up/down split, interleaved deterministically like
                       scratch_braess/so_grid_fine.py). If this matches MAPPO,
                       learning is not necessary at nominal demand (decision D1).
  RandomSplitPolicy  : uniform per-diamond assignment via a stable hash of the
                       vehicle id. No tuning; the "any spreading helps?" control.
  TollDijkstraPolicy : Dijkstra on live travel time + a queue-proportional
                       penalty on bottleneck (VAR) edges, i.e. a non-learning
                       marginal-cost router using the same 1-lane/>=150 m rule
                       as the marginal reward.

All are Braess-specific in route construction (edge names of the chained-Braess
net) but harness-generic in interface: make_decisions returns complete edge
routes, which StrSumo applies via vehicle_set_route.
"""
import hashlib
import math

import traci

from controller.DijkstraController import DijkstraPolicy
from controller.RouteController import RouteController

_LEG = {
    ("braess", 1): ["f_up1", "cross1", "g_dn1"],
    ("up", 1): ["f_up1", "g_up1"],
    ("down", 1): ["f_dn1", "g_dn1"],
    ("braess", 2): ["f_up2", "cross2", "g_dn2"],
    ("up", 2): ["f_up2", "g_up2"],
    ("down", 2): ["f_dn2", "g_dn2"],
}


def combo_route(src, l1, l2):
    return [src, "stage"] + _LEG[(l1, 1)] + ["link1"] + _LEG[(l2, 2)] + ["out"]


def interleaved_legs(pb, up_share, n, offset):
    """Deterministic stationary interleave; identical rule to so_grid_fine.py."""
    nb = int(round(pb * n))
    rest = n - nb
    nup = int(round(up_share * rest))
    legs = ["braess"] * nb + ["up"] * nup + ["down"] * (rest - nup)
    return [legs[(i * 7 + offset) % n] for i in range(n)]


class _AssignedRoutePolicy(RouteController):
    """Shared plumbing: hold a per-vehicle (l1, l2) assignment, emit the fixed
    route (suffix from the vehicle's current edge) at every decision point."""

    def __init__(self, connection_info):
        super().__init__(connection_info)
        self._routes = {}   # vehicle_id -> full edge list (built on first sight)

    def _assignment(self, vehicle_id):
        raise NotImplementedError

    def make_decisions(self, vehicles, connection_info):
        decisions = {}
        for vehicle in vehicles:
            vid = vehicle.vehicle_id
            cur = vehicle.current_edge
            if vid not in self._routes:
                l1, l2 = self._assignment(vid)
                src = cur if cur in ("in1", "in2") else "in1"
                self._routes[vid] = combo_route(src, l1, l2)
            route = self._routes[vid]
            if cur in route:
                decisions[vid] = list(route[route.index(cur):])
            else:
                # Off the assigned path (should not happen on this map); leave
                # the current route in place by re-sending from the current edge
                # to the destination via the tail of the assignment.
                decisions[vid] = [cur, "out"] if cur != "out" else [cur]
        return decisions


class FixedSplitPolicy(_AssignedRoutePolicy):
    def __init__(self, connection_info, n_vehicles, pb=0.0, up_share=0.45):
        super().__init__(connection_info)
        self._legs1 = interleaved_legs(pb, up_share, n_vehicles, 0)
        self._legs2 = interleaved_legs(pb, up_share, n_vehicles, 3)
        self._order = {}    # vehicle_id -> dense index in first-seen order
        self._n = n_vehicles

    def _assignment(self, vehicle_id):
        # Vehicle ids are numeric strings in this harness; fall back to
        # first-seen order if not.
        try:
            idx = int(vehicle_id) % self._n
        except ValueError:
            idx = self._order.setdefault(vehicle_id, len(self._order)) % self._n
        return self._legs1[idx], self._legs2[idx]


class RandomSplitPolicy(_AssignedRoutePolicy):
    def _assignment(self, vehicle_id):
        h = hashlib.md5(str(vehicle_id).encode()).digest()
        return (["braess", "up", "down"][h[0] % 3],
                ["braess", "up", "down"][h[1] % 3])


class TollDijkstraPolicy(DijkstraPolicy):
    """Dijkstra-dynamic plus a queue-proportional toll on bottleneck edges."""

    def __init__(self, connection_info, toll_per_vehicle_s=2.0,
                 min_edge_length=150.0):
        super().__init__(connection_info, weight_mode="traveltime")
        self.toll_per_vehicle_s = float(toll_per_vehicle_s)
        self._var_edges = None
        self._min_edge_length = float(min_edge_length)

    def _bottleneck_edges(self):
        if self._var_edges is None:
            var = set()
            for edge_id, length in self.connection_info.edge_length_dict.items():
                try:
                    lanes = int(traci.edge.getLaneNumber(edge_id))
                except Exception:
                    continue
                if lanes == 1 and float(length) >= self._min_edge_length:
                    var.add(edge_id)
            self._var_edges = var
        return self._var_edges

    def _edge_weight(self, edge_id):
        weight = super()._edge_weight(edge_id)
        if edge_id in self._bottleneck_edges():
            try:
                queued = float(traci.edge.getLastStepVehicleNumber(edge_id))
            except Exception:
                queued = 0.0
            toll = self.toll_per_vehicle_s * queued
            if math.isfinite(toll) and toll > 0.0:
                weight += toll
        return weight
