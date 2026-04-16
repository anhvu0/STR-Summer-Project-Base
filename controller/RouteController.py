from abc import ABC, abstractmethod
import os
import sys
from core.Util import *
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("No environment variable SUMO_HOME!")
import traci
import sumolib
import random

STRAIGHT = "s"
TURN_AROUND = "t"
LEFT = "l"
RIGHT = "r"
SLIGHT_LEFT = "L"
SLIGHT_RIGHT = "R"

class RouteController(ABC):
    """
    Base class for routing policy

    To implement a scheduling algorithm, implement the make_decisions() method.
    Please use the boilerplate code from the example, and implement your algorithm between
    the 'Your algo...' comments.

    make_decisions takes in a list of vehicles and network information (connection_info).
        Using this data, it should return a dictionary of {vehicle_id: decision}, where "decision"
        is one of the directions defined by SUMO (see constants above). Any scheduling algorithm
        may be injected into the simulation, as long as it is wrapped by the RouteController class
        and implements the make_decisions method.

    :param connection_info: object containing network information, including:
                            - out_going_edges_dict {edge_id: {direction: out_edge}}
                            - edge_length_dict {edge_id: edge_length}
                            - edge_index_dict {edge_index_dict} keep track of edge ids by an index
                            - edge_vehicle_count {edge_id: number of vehicles at edge}
                            - edge_list [edge_id]

    """
    def __init__(self, connection_info: ConnectionInfo):
        self.connection_info = connection_info
        self.direction_choices = [STRAIGHT, TURN_AROUND,  SLIGHT_RIGHT, RIGHT, SLIGHT_LEFT, LEFT]

    def compute_local_target(self, decision_list, vehicle):
        # Legacy helper for non-RL local-target controllers (e.g., Random/Dijkstra).
        # RL training/inference now apply contiguous routes directly via setRoute(...).
        current_target_edge = vehicle.current_edge
        try:
            if current_target_edge == vehicle.destination:
                return vehicle.destination

            # deterministic, no random fallbacks in runtime-critical routing
            path_length = 0.0
            horizon = max(float(vehicle.current_speed), 140.0)
            traversed_edges = [current_target_edge]

            for choice in decision_list:
                outgoing = self.connection_info.outgoing_edges_dict.get(current_target_edge, {})
                if choice not in outgoing:
                    break
                current_target_edge = outgoing[choice]
                traversed_edges.append(current_target_edge)
                path_length += float(self.connection_info.edge_length_dict.get(current_target_edge, 30.0))
                if current_target_edge == vehicle.destination or path_length >= horizon:
                    return current_target_edge

            # Extend deterministically via shortest path to keep fragment stable and connected.
            try:
                net = sumolib.net.readNet(self.connection_info.net_filename)
                from_edge = net.getEdge(current_target_edge)
                to_edge = net.getEdge(vehicle.destination)
                path_edges, _ = net.getShortestPath(from_edge, to_edge, vClass="passenger")
                if path_edges:
                    for edge_obj in path_edges[1:]:
                        edge_id = edge_obj.getID()
                        if not edge_obj.allows("passenger"):
                            break
                        traversed_edges.append(edge_id)
                        path_length += float(self.connection_info.edge_length_dict.get(edge_id, 30.0))
                        current_target_edge = edge_id
                        if current_target_edge == vehicle.destination or path_length >= horizon:
                            break
            except Exception:
                # Keep deterministic safe behavior; stay on last connected edge.
                pass

            return current_target_edge if traversed_edges else vehicle.current_edge

        except Exception as e:
            print("compute_local_target exception:", e)
            return vehicle.current_edge


    @abstractmethod
    def make_decisions(self, vehicles, connection_info):
        pass


class RandomPolicy(RouteController):
    """
    Example class for a custom scheduling algorithm.
    Utilizes a random decision policy until vehicle destination is within reach,
    then targets the vehicle destination.
    """
    def __init__(self, connection_info):
        super().__init__(connection_info)

    def make_decisions(self, vehicles, connection_info):
        """
        A custom scheduling algorithm can be written in between the 'Your algo...' comments.
        -For each car in the vehicle batch, your algorithm should provide a list of future decisions.
        -Sometimes short paths result in the vehicle reaching its local TRACI destination before reaching its
         true global destination. In order to counteract this, ask for a list of decisions rather than just one.
        -This list of decisions is sent to a function that returns the 'closest viable target' edge
          reachable by the decisions - it is not the case that all decisions will always be consumed.
          As soon as there is enough distance between the current edge and the target edge, the compute_target_edge
          function will return.
        -The 'closest viable edge' is a local target that is used by TRACI to control vehicles
        -The closest viable edge should always be far enough away to ensure that the vehicle is not removed
          from the simulation by TRACI before the vehicle reaches its true destination

        :param vehicles: list of vehicles to make routing decisions for
        :param connection_info: object containing network information
        :return: local_targets: {vehicle_id, target_edge}, where target_edge is a local target to send to TRACI
        """

        local_targets = {}
        for vehicle in vehicles:
            start_edge = vehicle.current_edge

            '''
            Your algo starts here
            '''
            decision_list = []

            i = 0
            while i < 10:  # choose the number of decisions to make in advanced; depends on the algorithm and network
                choice = self.direction_choices[random.randint(0, 5)]  # 6 choices available in total

                # dead end
                if len(self.connection_info.outgoing_edges_dict[start_edge].keys()) == 0:
                    break

                # make sure to check if it's a valid edge
                if choice in self.connection_info.outgoing_edges_dict[start_edge].keys():
                    decision_list.append(choice)
                    start_edge = self.connection_info.outgoing_edges_dict[start_edge][choice]

                    if i > 0:
                        if decision_list[i-1] == decision_list[i] and decision_list[i] == 't':
                            # stuck in a turnaround loop, let TRACI remove vehicle
                            break

                    i += 1

            '''
            Your algo ends here
            '''
            local_targets[vehicle.vehicle_id] = self.compute_local_target(decision_list, vehicle)

        return local_targets
