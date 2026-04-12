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
from traci import constants as tc
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

    def run(self):
        """
        Runs the SUMO simulation
        At each time-step, cars that have moved edges make a decision based on user-supplied scheduler algorithm
        Decisions are enforced in SUMO by setting the destination of the vehicle to the result of the
        :returns: total time, number of cars that reached their destination, number of deadlines missed
        """
        total_time = 0
        end_number = 0
        deadlines_missed = []

        step = 0
        vehicles_to_direct = [] #  the batch of controlled vehicles passed to make_decisions()
        vehicle_IDs_in_simulation = set()
        subscribed_vehicle_ids = set()
        controlled_vehicle_ids = set(self.controlled_vehicles.keys())

        for edge in self.connection_info.edge_list:
            traci.edge.subscribe(edge, (tc.LAST_STEP_VEHICLE_NUMBER,))

        try:
            while traci.simulation.getMinExpectedNumber() > 0:
                vehicle_ids = set(traci.vehicle.getIDList())

                # store edge vehicle counts in connection_info.edge_vehicle_count
                self.get_edge_vehicle_counts()
                #initialize vehicles to be directed
                vehicles_to_direct = []

                # iterate through vehicles currently in simulation
                for vehicle_id in vehicle_ids:

                    #should not be added because there is no corresponding -1, this makes edge_vehicle_count becomes the total number of vehicles that used to be on this edge.
                    #self.connection_info.edge_vehicle_count[traci.vehicle.getRoadID(vehicle_id)] += 1
                    
                    

                    # handle newly arrived controlled vehicles
                    if vehicle_id not in vehicle_IDs_in_simulation and vehicle_id in controlled_vehicle_ids:
                        vehicle_IDs_in_simulation.add(vehicle_id)
                        traci.vehicle.setColor(vehicle_id, (255, 0, 0)) # set color so we can visually track controlled vehicles
                        # keep lane changing conflict-free: disable autonomous lane changes from SUMO
                        # while preserving safety checks/collision avoidance behavior.
                        traci.vehicle.setLaneChangeMode(vehicle_id, 512)
                        self.controlled_vehicles[vehicle_id].start_time = float(step)#Use the detected release time as start time

                    if vehicle_id in controlled_vehicle_ids:
                        if vehicle_id not in subscribed_vehicle_ids:
                            traci.vehicle.subscribe(vehicle_id, (tc.VAR_ROAD_ID, tc.VAR_SPEED))
                            subscribed_vehicle_ids.add(vehicle_id)
                vehicle_states = traci.vehicle.getAllSubscriptionResults()

                for vehicle_id in (vehicle_ids & controlled_vehicle_ids):
                    vehicle_state = vehicle_states.get(vehicle_id, {})
                    current_edge = vehicle_state.get(tc.VAR_ROAD_ID)
                    current_speed = vehicle_state.get(tc.VAR_SPEED, 0.0)

                    if current_edge not in self.connection_info.edge_index_dict.keys():
                        continue
                    elif current_edge == self.controlled_vehicles[vehicle_id].destination:
                        continue

                    #print("{} now on: {}, records on {}; {} ".format(vehicle_id, current_edge, self.controlled_vehicles[vehicle_id].current_edge, current_edge!=self.controlled_vehicles[vehicle_id].current_edge))
                    if current_edge != self.controlled_vehicles[vehicle_id].current_edge:
                        self.controlled_vehicles[vehicle_id].current_edge = current_edge
                        self.controlled_vehicles[vehicle_id].current_speed = current_speed
                        vehicles_to_direct.append(self.controlled_vehicles[vehicle_id])
                #print(len(vehicles_to_direct))
                vehicle_decisions_by_id = self.route_controller.make_decisions(vehicles_to_direct, self.connection_info)
                if getattr(self.route_controller, "apply_direct_routes", False):
                    for vehicle_id, chosen_next_edge in vehicle_decisions_by_id.items():
                        if vehicle_id in self.controlled_vehicles:
                            self.controlled_vehicles[vehicle_id].local_destination = chosen_next_edge
                else:
                    for vehicle_id, local_target_edge in vehicle_decisions_by_id.items():
                        if vehicle_id in vehicle_ids:
                            try:
                                destination = self.controlled_vehicles[vehicle_id].destination
                                if local_target_edge != destination:
                                    traci.vehicle.setVia(vehicle_id, [local_target_edge])
                                else:
                                    traci.vehicle.setVia(vehicle_id, [])
                                traci.vehicle.changeTarget(vehicle_id, destination)
                                self.controlled_vehicles[vehicle_id].local_destination = local_target_edge
                            except traci.exceptions.TraCIException:
                                continue

                arrived_at_destination = traci.simulation.getArrivedIDList()

                for vehicle_id in arrived_at_destination:
                    if vehicle_id in self.controlled_vehicles:
                        #print the raw result out to the terminal
                        arrived_at_destination = True
                        end_number += 1
                        time_span = step - self.controlled_vehicles[vehicle_id].start_time
                        total_time += time_span
                        miss = False
                        if step > self.controlled_vehicles[vehicle_id].deadline:
                            deadlines_missed.append(vehicle_id)
                            miss = True
                        print("Vehicle {} reaches the destination: {}, timespan: {}, deadline missed: {}"\
                            .format(vehicle_id, arrived_at_destination, time_span, miss))
                        #if not arrived_at_destination:
                            #print("{} - {}".format(self.controlled_vehicles[vehicle_id].local_destination, self.controlled_vehicles[vehicle_id].destination))
                #  for x  in self.edge_list:
                        #44884008#2
                        #traci.getLastStepVehicleNumber(x)
                traci.simulationStep()
                step += 1

                if step > MAX_SIMULATION_STEPS:
                    print('Ending due to timeout.')
                    break

        except ValueError as err:
            print('Exception caught.')
            print(err)

        num_deadlines_missed = len(deadlines_missed)

        return total_time, end_number, num_deadlines_missed

    def get_edge_vehicle_counts(self):
        edge_states = traci.edge.getAllSubscriptionResults()
        for edge in self.connection_info.edge_list:
            edge_state = edge_states.get(edge, {})
            self.connection_info.edge_vehicle_count[edge] = edge_state.get(tc.LAST_STEP_VEHICLE_NUMBER, 0)
