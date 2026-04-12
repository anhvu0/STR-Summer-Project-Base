from abc import ABC, abstractmethod
import os
import sys
from core.Util import *
import sumolib

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("No environment variable SUMO_HOME!")

STRAIGHT = "s"
TURN_AROUND = "t"
LEFT = "l"
RIGHT = "r"
SLIGHT_LEFT = "L"
SLIGHT_RIGHT = "R"


class RouteController(ABC):
    def __init__(self, connection_info: ConnectionInfo):
        self.connection_info = connection_info
        self.direction_choices = [STRAIGHT, TURN_AROUND, SLIGHT_RIGHT, RIGHT, SLIGHT_LEFT, LEFT]
        self._net = None
        self._path_cache = {}

    def _ensure_net(self):
        if self._net is None:
            self._net = sumolib.net.readNet(self.connection_info.net_filename)

    def _path_to_dest(self, from_edge: str, destination: str):
        key = (from_edge, destination)
        if key in self._path_cache:
            return self._path_cache[key]
        self._ensure_net()
        try:
            start = self._net.getEdge(from_edge)
            end = self._net.getEdge(destination)
            path, _ = self._net.getShortestPath(start, end)
            ids = [e.getID() for e in path] if path else []
        except Exception:
            ids = []
        self._path_cache[key] = ids
        return ids

    def compute_local_target(self, decision_list, vehicle):
        """
        Deterministic and destination-safe local target computation.
        - applies provided directions while valid
        - validates destination connectivity
        - extends short plans with validated shortest-path continuation
        - blocks repeated turn-around oscillation
        """
        current_edge = vehicle.current_edge
        destination = vehicle.destination
        if current_edge == destination:
            return destination

        max_hops = 8
        horizon_m = max(140.0, min(320.0, float(vehicle.current_speed) * 9.0 + 100.0))
        walked = 0.0
        route = [current_edge]

        prev_dir = None
        for direction in decision_list:
            if len(route) >= max_hops:
                break
            outgoing = self.connection_info.outgoing_edges_dict.get(route[-1], {})
            if direction not in outgoing:
                break
            if prev_dir == TURN_AROUND and direction == TURN_AROUND:
                break
            next_edge = outgoing[direction]
            if next_edge in route:
                break
            route.append(next_edge)
            walked += float(self.connection_info.edge_length_dict.get(next_edge, 30.0))
            prev_dir = direction
            if next_edge == destination or walked >= horizon_m:
                break

        tail = route[-1]
        if tail != destination:
            continuation = self._path_to_dest(tail, destination)
            # continuation includes tail itself
            if continuation:
                for edge in continuation[1:]:
                    if len(route) >= max_hops:
                        break
                    if edge in route:
                        break
                    route.append(edge)
                    walked += float(self.connection_info.edge_length_dict.get(edge, 30.0))
                    if edge == destination or walked >= horizon_m:
                        break

        # Never return an unsafe disconnected target.
        for edge in reversed(route):
            if edge == current_edge:
                return current_edge
            if self._path_to_dest(edge, destination):
                return edge
        return current_edge

    @abstractmethod
    def make_decisions(self, vehicles, connection_info):
        pass


class RandomPolicy(RouteController):
    def __init__(self, connection_info):
        super().__init__(connection_info)

    def make_decisions(self, vehicles, connection_info):
        local_targets = {}
        for vehicle in vehicles:
            start_edge = vehicle.current_edge
            decision_list = []
            for _ in range(6):
                outgoing = self.connection_info.outgoing_edges_dict.get(start_edge, {})
                if not outgoing:
                    break
                choice = next(iter(outgoing.keys()))
                decision_list.append(choice)
                start_edge = outgoing[choice]
                if start_edge == vehicle.destination:
                    break
            local_targets[vehicle.vehicle_id] = self.compute_local_target(decision_list, vehicle)
        return local_targets
