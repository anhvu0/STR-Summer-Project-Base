from controller.RouteController import RouteController
from core.Util import ConnectionInfo, Vehicle
from keras.models import load_model
import numpy as np
import traci
import sumolib
import math

from xml.dom.minidom import parse
import os

def parse_sumocfg(sumocfg_path):
    dom = parse(sumocfg_path)
    net_file = dom.getElementsByTagName('net-file')[0].attributes['value'].nodeValue
    return os.path.join(os.path.dirname(sumocfg_path), net_file)
net_path = parse_sumocfg("./configurations/myconfig.sumocfg")


class QLearningPolicy(RouteController):
    def __init__(self, vehicles, connection_info, model_file, net_xml_file = net_path):
        super().__init__(connection_info)
        self.model = load_model(model_file)
        self.vehicles = vehicles
        self.net = sumolib.net.readNet(net_xml_file)
        self._visit_count = {}
        self._best_dist = {}
        # Cache for shortest-path distances (edge_id, dest_id) -> cost
        # How many actions to plan ahead each time
        self.decision_horizon = 6


    
    #The below function is for verifying whether the shape of the first layer matches the network expected in SUMO config file.
    # def _validate_model_input_shape(self):
    #     expected_state_size = 2 + 6 + len(self.connection_info.edge_list)
    #     model_input_shape = self.model.input_shape
    #     if isinstance(model_input_shape, (list, tuple)) and model_input_shape:
    #         model_input_dim = model_input_shape[-1]
    #     else:
    #         model_input_dim = None
    #     if model_input_dim != expected_state_size:
    #         raise ValueError(
    #             "Model input shape does not match SUMO network state size. "
    #             f"Expected {expected_state_size}, got {model_input_dim}. "
    #             "Retrain the model using the current SUMO .sumocfg/.net.xml files "
    #             "or load a model trained on this network."
    #         )
    #-----------------------DEBUGGING-------------------------------------
    def _dist_to_dest(self, edge_id, dest_id):
        try:
            from_edge = self.net.getEdge(edge_id)
            to_edge = self.net.getEdge(dest_id)
            path_edges, path_cost = self.net.getShortestPath(from_edge, to_edge)
            if path_edges is None:
                return float("inf")
            return path_cost
        except Exception:
            return float("inf")
    #----------------------------------------------------------------------


    def make_decisions(self, vehicles, connection_info: ConnectionInfo):
        local_targets = {}

        if not hasattr(self, "_debug_net_checked"):
            self._debug_net_checked = True
            print("[DEBUG] has self.net:", hasattr(self, "net"))
            print("[DEBUG] self.net type:", type(self.net))

        for vehicle in vehicles:
            wrong_decision = False
            start_edge = vehicle.current_edge
            decision_list = []

            if vehicle.destination == vehicle.current_edge:
                continue

            vid = vehicle.vehicle_id
            if vid not in self._visit_count:
                self._visit_count[vid] = {}
            if vid not in self._best_dist:
                self._best_dist[vid] = float("inf")

            i = 0
            while i < self.decision_horizon:
                outgoing = connection_info.outgoing_edges_dict.get(start_edge, {})
                valid_dirs = list(outgoing.keys())

                if not valid_dirs:
                    print(f"[DEADEND] veh={vid} edge={start_edge} dest={vehicle.destination} has no outgoing")
                    wrong_decision = True
                    break

                state = self.getState(start_edge, vehicle.destination)
                action_idx = self.act(state)
                action = self.direction_choices[action_idx]

                # ---------- if model picks an impossible action, fallback ----------
                if action not in outgoing:
                    print(
                        f"[IMPOSSIBLE] veh={vid} edge={start_edge} dest={vehicle.destination} "
                        f"chosen='{action}' valid_dirs={valid_dirs} state_bits={state[0][2:8].tolist()}"
                    )
                    action = None  # trigger fallback below

                #---------- PROGRESS / LOOP GUARD ----------
                #Evaluate proposed action and alternatives using shortest-path distance
                dest_id = vehicle.destination

                def score_dir(dir_char):
                    nxt = outgoing[dir_char]
                    d = self._dist_to_dest(nxt, dest_id)
                    return d, nxt

                # best possible move from here (by distance-to-dest)
                best_dir = None
                best_d = float("inf")
                best_next = None
                for dch in valid_dirs:
                    d, nxt = score_dir(dch)
                    if d < best_d:
                        best_d = d
                        best_dir = dch
                        best_next = nxt

                # model-proposed next
                if action is not None:
                    prop_next = outgoing[action]
                    prop_d = self._dist_to_dest(prop_next, dest_id)
                else:
                    prop_next = None
                    prop_d = float("inf")

                # update visit count for proposed next (if any)
                if prop_next is not None:
                    self._visit_count[vid][prop_next] = self._visit_count[vid].get(prop_next, 0) + 1
                    visit = self._visit_count[vid][prop_next]
                else:
                    visit = 999

                # update best distance achieved so far
                if best_d < self._best_dist[vid]:
                    self._best_dist[vid] = best_d

                # when to override:
                # 1) proposed has no path
                # 2) proposed repeats too much
                # 3) proposed is much worse than best available
                override = False
                if prop_d == float("inf"):
                    override = True
                if visit >= 3:
                    override = True
                if prop_d > best_d + 50.0:   # slack threshold (tune this)
                    override = True

                if override:
                    print(
                        f"[OVERRIDE] veh={vid} edge={start_edge} dest={dest_id} "
                        f"chosen='{action}' prop_d={prop_d} best_dir='{best_dir}' best_d={best_d} visit={visit}"
                    )
                    action = best_dir
                    prop_next = best_next
                    prop_d = best_d
                #------------------------------------------

                print("Choice for " + str(start_edge) + " is: " + str(action))

                target_edge = outgoing[action]
                start_edge = target_edge
                decision_list.append(action)

                i += 1
                if start_edge == vehicle.destination:
                    break

            if wrong_decision:
                continue

            local_targets[vehicle.vehicle_id] = self.compute_local_target(decision_list, vehicle)

        return local_targets




    # this function reacheds the Neural Network trained before and let it make a decision for the situation now
    def act(self, state):
        act_values = self.model.predict(state, verbose=0)
        state_vals = state[0][2:8]
        state_vals = state_vals.reshape(act_values.shape)
        #print(state)
        mod_values = act_values - 10000 * (1 - state_vals)
        #print(mod_values)
        #print('**************************')
        return np.argmax(mod_values[0])

    # this function gives the current state of the vehicle based on the state size
    def getState(self, edge_now, destination_edge):
        en = edge_now
        state = []
        state.append(self.connection_info.edge_index_dict[en])
        state.append(self.connection_info.edge_index_dict[destination_edge])
        for c in self.direction_choices:
            if c in self.connection_info.outgoing_edges_dict[en].keys():
                state.append(1)
                # 1 stands for the edge is available for being chose
            else:
                state.append(0)
                # 0 means this action cannot be chosen.
        # put the congestion ratio of all edges into the state.
        for edge_now in self.connection_info.edge_list:
            car_num = traci.edge.getLastStepVehicleNumber(edge_now)
            density = car_num / self.connection_info.edge_length_dict[edge_now]
            state.append(density)

        state = np.reshape(state, [1, len(state)])
        return state
