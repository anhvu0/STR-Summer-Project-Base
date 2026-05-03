import os
import sys
import optparse
import xml.etree.ElementTree as ET
from xml.dom.minidom import parse, parseString
from core.Util import *
from core.target_vehicles_generation_protocols import *

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("No environment variable SUMO_HOME!")

import traci
from traci import constants as tc
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


def build_runtime_sumocfg(sumocfg_path, fast_mode=False):
    resolved_path = os.path.abspath(sumocfg_path)
    if not fast_mode:
        return resolved_path

    runtime_path = os.path.splitext(resolved_path)[0] + ".fast.sumocfg"
    tree = ET.parse(resolved_path)
    root = tree.getroot()
    for output_node in list(root.findall("output")):
        root.remove(output_node)
    tree.write(runtime_path, encoding="utf-8", xml_declaration=False)
    return runtime_path

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
        self._vehicle_subscription_vars = (
            tc.VAR_ROAD_ID,
            tc.VAR_SPEED,
        )
        self._edge_subscription_vars = (tc.LAST_STEP_VEHICLE_NUMBER,)
        self._active_vehicle_subscriptions = set()
        #print(self.controlled_vehicles)

    def _initialize_edge_subscriptions(self):
        for edge_id in self.connection_info.edge_list:
            traci.edge.subscribe(edge_id, self._edge_subscription_vars)

    def _ensure_vehicle_subscriptions(self, vehicle_ids):
        for vehicle_id in vehicle_ids:
            if vehicle_id in self._active_vehicle_subscriptions:
                continue
            try:
                traci.vehicle.subscribe(vehicle_id, self._vehicle_subscription_vars)
                self._active_vehicle_subscriptions.add(vehicle_id)
            except traci.TraCIException:
                continue

    def run(self, verbose=True, return_stats=False, print_runtime_summary=True):
        """
        Runs the SUMO simulation.

        At each time-step, cars that have moved edges make a decision based on the user-supplied scheduler algorithm.

        Args:
            verbose: If False, suppress per-vehicle arrival and timeout prints.
            return_stats: If True, append a stats dictionary to the return tuple.
            print_runtime_summary: If True, print controller runtime summaries when available.

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
        vehicle_IDs_in_simulation = set()
        controlled_vehicle_ids = set(self.controlled_vehicles.keys())

        simulation_get_min_expected = traci.simulation.getMinExpectedNumber
        simulation_get_arrived_ids = traci.simulation.getArrivedIDList
        simulation_step = traci.simulationStep
        vehicle_get_ids = traci.vehicle.getIDList
        vehicle_get_road = traci.vehicle.getRoadID
        vehicle_get_speed = traci.vehicle.getSpeed
        vehicle_set_color = traci.vehicle.setColor
        vehicle_set_route = traci.vehicle.setRoute
        vehicle_set_via = traci.vehicle.setVia
        vehicle_change_target = traci.vehicle.changeTarget
        vehicle_get_all_subscription_results = traci.vehicle.getAllSubscriptionResults
        edge_get_all_subscription_results = traci.edge.getAllSubscriptionResults
        self._active_vehicle_subscriptions = set()
        self._initialize_edge_subscriptions()

        try:
            while simulation_get_min_expected() > 0:
                vehicle_ids = set(vehicle_get_ids())
                self._ensure_vehicle_subscriptions(vehicle_ids & controlled_vehicle_ids)
                vehicle_results = vehicle_get_all_subscription_results() or {}
                edge_results = edge_get_all_subscription_results() or {}

                # store edge vehicle counts in connection_info.edge_vehicle_count
                self.get_edge_vehicle_counts(edge_results=edge_results)
                #initialize vehicles to be directed
                vehicles_to_direct = []
                queued_ids = set()

                # iterate through vehicles currently in simulation
                for vehicle_id in vehicle_ids:

                    #should not be added because there is no corresponding -1, this makes edge_vehicle_count becomes the total number of vehicles that used to be on this edge.
                    #self.connection_info.edge_vehicle_count[traci.vehicle.getRoadID(vehicle_id)] += 1

                    # handle newly arrived controlled vehicles
                    if vehicle_id not in vehicle_IDs_in_simulation and vehicle_id in controlled_vehicle_ids:
                        vehicle_IDs_in_simulation.add(vehicle_id)
                        vehicle_set_color(vehicle_id, (255, 0, 0)) # set color so we can visually track controlled vehicles
                        self.controlled_vehicles[vehicle_id].start_time = float(step)#Use the detected release time as start time

                    if vehicle_id in controlled_vehicle_ids:
                        result = vehicle_results.get(vehicle_id) or {}
                        try:
                            current_edge = result.get(tc.VAR_ROAD_ID)
                            if current_edge is None:
                                current_edge = vehicle_get_road(vehicle_id)
                        except traci.TraCIException:
                            continue

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
                            try:
                                current_speed = result.get(tc.VAR_SPEED)
                                if current_speed is None:
                                    current_speed = vehicle_get_speed(vehicle_id)
                            except traci.TraCIException:
                                continue
                            self.controlled_vehicles[vehicle_id].current_speed = float(current_speed)
                            vehicles_to_direct.append(self.controlled_vehicles[vehicle_id])
                            queued_ids.add(vehicle_id)
                #print(len(vehicles_to_direct))
                vehicle_decisions_by_id = self.route_controller.make_decisions(vehicles_to_direct, self.connection_info)
                live_vehicle_ids = vehicle_ids
                for vehicle_id, route_decision in vehicle_decisions_by_id.items():
                    if vehicle_id in live_vehicle_ids:
                        try:
                            if isinstance(route_decision, (list, tuple)) and len(route_decision) >= 2:
                                # Preferred path: apply one contiguous explicit route.
                                vehicle_set_route(vehicle_id, list(route_decision))
                                self.controlled_vehicles[vehicle_id].local_destination = route_decision[-1]
                            else:
                                # Backward-compatible fallback for legacy controllers.
                                destination = self.controlled_vehicles[vehicle_id].destination
                                local_target_edge = route_decision
                                if local_target_edge != destination:
                                    vehicle_set_via(vehicle_id, [local_target_edge])
                                else:
                                    vehicle_set_via(vehicle_id, [])
                                vehicle_change_target(vehicle_id, destination)
                                self.controlled_vehicles[vehicle_id].local_destination = local_target_edge
                        except traci.exceptions.TraCIException:
                            # If SUMO cannot build a route to this local target from
                            # current lane/route context, keep the previous target and
                            # retry on the next control step.
                            continue

                simulation_step()
                step += 1

                arrived_at_destination = simulation_get_arrived_ids()

                for vehicle_id in arrived_at_destination:
                    if vehicle_id not in controlled_vehicle_ids:
                        continue
                    if vehicle_id in arrived_controlled_ids:
                        continue
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
                    if hasattr(self.route_controller, "cleanup_vehicle_state"):
                        try:
                            self.route_controller.cleanup_vehicle_state(vehicle_id)
                        except Exception:
                            pass

                if step > MAX_SIMULATION_STEPS:
                    step_limit_reached = True
                    alive_at_step_cap_ids = set()
                    for vehicle_id in vehicle_get_ids():
                        if vehicle_id not in controlled_vehicle_ids:
                            continue
                        try:
                            current_edge = vehicle_get_road(vehicle_id)
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

        if hasattr(self.route_controller, 'get_runtime_metrics'):
            try:
                stats['controller_runtime_metrics'] = self.route_controller.get_runtime_metrics()
            except Exception:
                pass

        if print_runtime_summary and hasattr(self.route_controller, 'format_runtime_metrics_summary'):
            try:
                for line in self.route_controller.format_runtime_metrics_summary():
                    print(line)
            except Exception:
                pass

        if return_stats:
            return total_time, end_number, num_deadlines_missed, stats
        return total_time, end_number, num_deadlines_missed

    def get_edge_vehicle_counts(self, edge_results=None):
        edge_results = edge_results or {}
        for edge in self.connection_info.edge_list:
            result = edge_results.get(edge) or {}
            count = result.get(tc.LAST_STEP_VEHICLE_NUMBER)
            if count is None:
                count = traci.edge.getLastStepVehicleNumber(edge)
            self.connection_info.edge_vehicle_count[edge] = int(count)
