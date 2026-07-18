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
import copy
import hashlib
import math

import traci

from controller.DijkstraController import DijkstraPolicy
from controller.RouteController import RouteController
from core.coordination_throttle import ReservationField, ReservationFieldConfig

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


class ReservationDijkstraPolicy(DijkstraPolicy):
    """Dijkstra-dynamic that also reads the anticipatory route-reservation field.

    This isolates the reservation channel from learning. It is the live-travel-time
    replanner (weight_mode="traveltime") plus the SAME decaying booking field the
    MAPPO controller uses (core.coordination_throttle.ReservationField): within one
    decision round the vehicles are routed in a fixed order, each committed route
    books its leading edges, and a later decider pays a penalty on an edge in
    proportion to how many earlier vehicles just committed to it. So simultaneous
    deciders no longer all pile onto the momentarily fastest alternative.

    The reservation ablation (Table on Braess) is the only stack component whose
    removal significantly hurts MAPPO; this arm answers whether a non-learning
    replanner reading the same field matches the learned policy or not.

    Penalty units: the field stores a decaying reserved count per edge (a freshly
    booked leading edge contributes ~1.0). We convert that to seconds by
    ``reservation_toll_s`` per reserved unit, in the spirit of TollDijkstra's
    queue toll, and add it to the live edge travel time used by Dijkstra.
    """

    def __init__(self, connection_info, reservation_toll_s=6.0,
                 route_horizon=4, route_decay=0.7, time_decay=0.85):
        super().__init__(connection_info, weight_mode="traveltime")
        self.reservation_toll_s = float(reservation_toll_s)
        self._field = ReservationField(ReservationFieldConfig(
            enabled=True, route_horizon=int(route_horizon),
            route_decay=float(route_decay), time_decay=float(time_decay)))

    def _edge_weight(self, edge_id):
        weight = super()._edge_weight(edge_id)
        reserved = self._field.reserved_count(edge_id)
        if reserved > 0.0:
            toll = self.reservation_toll_s * reserved
            if math.isfinite(toll) and toll > 0.0:
                weight += toll
        return weight

    def _edges_from_directions(self, current_edge, directions):
        edges = [current_edge]
        edge = current_edge
        for direction in directions:
            outgoing = self.connection_info.outgoing_edges_dict.get(edge, {})
            if direction not in outgoing:
                break
            edge = outgoing[direction]
            edges.append(edge)
        return edges

    def _shortest_path_directions(self, vehicle):
        """One-vehicle Dijkstra on reservation-penalized live travel time. Returns
        the list of SUMO directions to the destination (same encoding the parent
        make_decisions builds)."""
        unvisited = {edge: 1000000000 for edge in self.connection_info.edge_list}
        current_edge = vehicle.current_edge
        current_distance = self._edge_weight(current_edge)
        unvisited[current_edge] = current_distance
        path_lists = {edge: [] for edge in self.connection_info.edge_list}
        while True:
            if current_edge not in self.connection_info.outgoing_edges_dict:
                break
            for direction, outgoing_edge in \
                    self.connection_info.outgoing_edges_dict[current_edge].items():
                if outgoing_edge not in unvisited:
                    continue
                new_distance = current_distance + self._edge_weight(outgoing_edge)
                if new_distance < unvisited[outgoing_edge]:
                    unvisited[outgoing_edge] = new_distance
                    current_path = copy.deepcopy(path_lists[current_edge])
                    current_path.append(direction)
                    path_lists[outgoing_edge] = current_path
            if current_edge in unvisited:
                del unvisited[current_edge]
            if not unvisited:
                break
            if current_edge == vehicle.destination:
                break
            possible = [e for e in unvisited.items() if e[1]]
            if not possible:
                break
            current_edge, current_distance = sorted(possible, key=lambda x: x[1])[0]
        return path_lists.get(vehicle.destination, [])

    def make_decisions(self, vehicles, connection_info):
        # Fade last round's books before this round's deciders read the field.
        self._field.decay()
        local_targets = {}
        # Deterministic decision order so the booking sequence is reproducible.
        for vehicle in sorted(vehicles, key=lambda v: str(v.vehicle_id)):
            directions = self._shortest_path_directions(vehicle)
            route_edges = self._edges_from_directions(vehicle.current_edge, directions)
            self._field.seed_route(route_edges)
            local_targets[vehicle.vehicle_id] = self.compute_local_target(directions, vehicle)
        return local_targets


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
