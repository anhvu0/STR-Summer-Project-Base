import os
import sys
import optparse
from xml.dom.minidom import parse, parseString
from core.Util import *
from core.target_vehicles_generation_protocols import *

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("No environment variable SUMO_HOME!")

import traci
import sumolib
import numpy as np
from controller.RouteController import *

"""
SUMO Selfless Traffic Routing (STR) Testbed
"""

MAX_SIMULATION_STEPS = 2000

# TODO: decide which file to put these in. Right now they're also defined in RouteController!!
STRAIGHT = "s"
TURN_AROUND = "t"
LEFT = "l"
RIGHT = "r"
SLIGHT_LEFT = "L"
SLIGHT_RIGHT = "R"

class StrSumo:
    def __init__(self, route_controller, connection_info, controlled_vehicles):
        """
        :param route_controller: object that implements the scheduling algorithm for controlled vehicles
        :param connection_info: object that includes the map information
        :param controlled_vehicles: a dictionary that includes the vehicles under control
        """
        self.direction_choices = [STRAIGHT, TURN_AROUND, SLIGHT_RIGHT, RIGHT, SLIGHT_LEFT, LEFT]
        self.connection_info = connection_info
        self.route_controller = route_controller
        self.controlled_vehicles =  controlled_vehicles # dictionary of Vehicles by id
        #print(self.controlled_vehicles)

    def run(self, verbose=True, return_stats=False):
        """
        Runs the SUMO simulation.

        At each time-step, cars that have moved edges make a decision based on the user-supplied scheduler algorithm.

        Args:
            verbose: If False, suppress per-vehicle arrival and timeout prints.
            return_stats: If True, append a stats dictionary to the return tuple.

        Returns:
            By default: (total_time, number_reached_destination, deadlines_missed_count)
            If return_stats=True: (..., stats_dict)
        """
        total_time = 0
        end_number = 0
        deadlines_missed = []
        completed_travel_times = []
        arrived_controlled_ids = set()
        alive_at_step_cap_ids = set()
        step_limit_reached = False

        step = 0
        vehicles_to_direct = [] #  the batch of controlled vehicles passed to make_decisions()
        vehicle_IDs_in_simulation = []

        try:
            while traci.simulation.getMinExpectedNumber() > 0:
                vehicle_ids = set(traci.vehicle.getIDList())

                # store edge vehicle counts in connection_info.edge_vehicle_count
                self.get_edge_vehicle_counts()
                #initialize vehicles to be directed
                vehicles_to_direct = []
                queued_ids = set()

                # iterate through vehicles currently in simulation
                for vehicle_id in vehicle_ids:

                    #should not be added because there is no corresponding -1, this makes edge_vehicle_count becomes the total number of vehicles that used to be on this edge.
                    #self.connection_info.edge_vehicle_count[traci.vehicle.getRoadID(vehicle_id)] += 1

                    # handle newly arrived controlled vehicles
                    if vehicle_id not in vehicle_IDs_in_simulation and vehicle_id in self.controlled_vehicles:
                        vehicle_IDs_in_simulation.append(vehicle_id)
                        traci.vehicle.setColor(vehicle_id, (255, 0, 0)) # set color so we can visually track controlled vehicles
                        self.controlled_vehicles[vehicle_id].start_time = float(step)#Use the detected release time as start time

                    if vehicle_id in self.controlled_vehicles.keys():
                        current_edge = traci.vehicle.getRoadID(vehicle_id)

                        if current_edge not in self.connection_info.edge_index_dict.keys():
                            continue
                        elif current_edge == self.controlled_vehicles[vehicle_id].destination:
                            continue

                        edge_changed = (current_edge != self.controlled_vehicles[vehicle_id].current_edge)

                        should_force_control = False
                        if hasattr(self.route_controller, "should_control_vehicle"):
                            try:
                                should_force_control = bool(
                                    self.route_controller.should_control_vehicle(
                                        vehicle_id,
                                        self.controlled_vehicles[vehicle_id],
                                        step,
                                    )
                                )
                            except Exception:
                                should_force_control = False

                        if edge_changed:
                            self.controlled_vehicles[vehicle_id].current_edge = current_edge

                        if (edge_changed or should_force_control) and vehicle_id not in queued_ids:
                            self.controlled_vehicles[vehicle_id].current_speed = traci.vehicle.getSpeed(vehicle_id)
                            vehicles_to_direct.append(self.controlled_vehicles[vehicle_id])
                            queued_ids.add(vehicle_id)
                #print(len(vehicles_to_direct))
                vehicle_decisions_by_id = self.route_controller.make_decisions(vehicles_to_direct, self.connection_info)
                for vehicle_id, route_decision in vehicle_decisions_by_id.items():
                    if vehicle_id in traci.vehicle.getIDList():
                        try:
                            if isinstance(route_decision, (list, tuple)) and len(route_decision) >= 2:
                                # Preferred path: apply one contiguous explicit route.
                                traci.vehicle.setRoute(vehicle_id, list(route_decision))
                                self.controlled_vehicles[vehicle_id].local_destination = route_decision[-1]
                            else:
                                # Backward-compatible fallback for legacy controllers.
                                destination = self.controlled_vehicles[vehicle_id].destination
                                local_target_edge = route_decision
                                if local_target_edge != destination:
                                    traci.vehicle.setVia(vehicle_id, [local_target_edge])
                                else:
                                    traci.vehicle.setVia(vehicle_id, [])
                                traci.vehicle.changeTarget(vehicle_id, destination)
                                self.controlled_vehicles[vehicle_id].local_destination = local_target_edge
                        except traci.exceptions.TraCIException:
                            # If SUMO cannot build a route to this local target from
                            # current lane/route context, keep the previous target and
                            # retry on the next control step.
                            continue

                arrived_at_destination = traci.simulation.getArrivedIDList()

                for vehicle_id in arrived_at_destination:
                    if vehicle_id in self.controlled_vehicles:
                        end_number += 1
                        arrived_controlled_ids.add(vehicle_id)
                        time_span = step - self.controlled_vehicles[vehicle_id].start_time
                        completed_travel_times.append(float(time_span))
                        total_time += time_span
                        miss = False
                        if step > self.controlled_vehicles[vehicle_id].deadline:
                            deadlines_missed.append(vehicle_id)
                            miss = True
                        if verbose:
                            print("Vehicle {} reaches the destination: {}, timespan: {}, deadline missed: {}"                                .format(vehicle_id, True, time_span, miss))
                traci.simulationStep()
                step += 1

                if step > MAX_SIMULATION_STEPS:
                    step_limit_reached = True
                    alive_at_step_cap_ids = set()
                    for vehicle_id in traci.vehicle.getIDList():
                        if vehicle_id not in self.controlled_vehicles:
                            continue
                        try:
                            current_edge = traci.vehicle.getRoadID(vehicle_id)
                        except traci.TraCIException:
                            continue
                        if current_edge != self.controlled_vehicles[vehicle_id].destination:
                            alive_at_step_cap_ids.add(vehicle_id)
                    if verbose:
                        print('Ending due to timeout.')
                    break

        except ValueError as err:
            if verbose:
                print('Exception caught.')
                print(err)

        num_deadlines_missed = len(deadlines_missed)
        total_controlled = max(len(self.controlled_vehicles), 1)
        avg_travel_time = (float(total_time) / float(end_number)) if end_number > 0 else float('inf')
        p50_travel_time = (float(np.percentile(completed_travel_times, 50)) if completed_travel_times else float('inf'))
        p90_travel_time = (float(np.percentile(completed_travel_times, 90)) if completed_travel_times else float('inf'))
        if completed_travel_times:
            tail_travel_times = [tt for tt in completed_travel_times if tt >= p90_travel_time]
        else:
            tail_travel_times = []
        unfinished_count = int(len(alive_at_step_cap_ids))
        tail_vehicles_over_p90_count = int(len(tail_travel_times) + unfinished_count)
        tail_reference = (
            float(np.mean(tail_travel_times))
            if tail_travel_times else (p90_travel_time if np.isfinite(p90_travel_time) else float(step))
        )
        if unfinished_count > 0:
            # Treat unresolved controlled vehicles as long-tail outcomes when
            # summarizing frozen inference behavior.
            tail_reference = (
                (tail_reference * len(tail_travel_times)) + (float(step) * unfinished_count)
            ) / max(len(tail_travel_times) + unfinished_count, 1)
        p50_baseline = p50_travel_time if np.isfinite(p50_travel_time) else 0.0
        tail_completion_gap_steps = float(max(tail_reference - p50_baseline, 0.0))
        p95_to_p50_travel_ratio = (
            float(np.percentile(completed_travel_times, 95) / max(p50_travel_time, 1e-6))
            if completed_travel_times else float('inf')
        )
        stats = {
            'completion_rate': float(end_number) / float(total_controlled),
            'avg_travel_time': avg_travel_time,
            'p50_travel_time': p50_travel_time,
            'p90_travel_time': p90_travel_time,
            'deadlines_missed': num_deadlines_missed,
            'vehicles_reached_destination': int(end_number),
            'controlled_vehicle_count': int(len(self.controlled_vehicles)),
            'completed_travel_times': completed_travel_times,
            'step_limit_reached': bool(step_limit_reached),
            'alive_at_step_cap': int(len(alive_at_step_cap_ids)),
            'timeout_rate': float(len(alive_at_step_cap_ids)) / float(total_controlled),
            'tail_vehicles_over_p90_count': tail_vehicles_over_p90_count,
            'tail_completion_gap_steps': tail_completion_gap_steps,
            'p95_to_p50_travel_ratio': p95_to_p50_travel_ratio,
            'max_step': int(step),
        }

        if return_stats:
            return total_time, end_number, num_deadlines_missed, stats
        return total_time, end_number, num_deadlines_missed

    def get_edge_vehicle_counts(self):
        for edge in self.connection_info.edge_list:
            self.connection_info.edge_vehicle_count[edge] = traci.edge.getLastStepVehicleNumber(edge)
