from controller.RouteController import RouteController
from core.Util import ConnectionInfo, Vehicle
from keras.models import load_model
import numpy as np
import traci
import sumolib
import math
from collections import deque

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
        self._recent_edges = {}
        self._metrics = {
            "decisions": 0,
            "overrides": 0,
            "loop_overrides": 0,
            "distance_overrides": 0,
            "impossible_action_overrides": 0,
        }
        self._last_metrics_snapshot = None
        # Cache for shortest-path distances (edge_id, dest_id) -> cost
        # How many actions to plan ahead each time
        self.decision_horizon = 6
        self.loop_window = 10
        self.loop_repeat_threshold = 2
        self.distance_slack = 50.0


    


    def _compute_deadline_features(self, vehicle_id):
        vehicle_obj = self.vehicles.get(str(vehicle_id))
        if vehicle_obj is None:
            return [0.0, 0.0, 0.0]

        now = traci.simulation.getTime()
        deadline_window = max(float(vehicle_obj.deadline) - float(vehicle_obj.start_time), 1.0)
        time_left = max(float(vehicle_obj.deadline) - float(now), 0.0)
        elapsed = max(float(now) - float(vehicle_obj.start_time), 0.0)
        urgency = 1.0 - min(time_left / deadline_window, 1.0)

        return [
            min(time_left / deadline_window, 1.0),
            min(elapsed / deadline_window, 1.0),
            urgency,
        ]

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
            # print("[DEBUG] has self.net:", hasattr(self, "net"))
            # print("[DEBUG] self.net type:", type(self.net))

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
            if vid not in self._recent_edges:
                self._recent_edges[vid] = deque(maxlen=self.loop_window)

            i = 0
            while i < self.decision_horizon:
                outgoing = connection_info.outgoing_edges_dict.get(start_edge, {})
                valid_dirs = list(outgoing.keys())

                if not valid_dirs:
                    # print(f"[DEADEND] veh={vid} edge={start_edge} dest={vehicle.destination} has no outgoing")
                    wrong_decision = True
                    break

                state = self.getState(vid, start_edge, vehicle.destination)
                action_idx = self.act(state)
                action = self.direction_choices[action_idx]
                self._metrics["decisions"] += 1

                # ---------- if model picks an impossible action, fallback ----------
                if action not in outgoing:
                    self._metrics["impossible_action_overrides"] += 1
                    # print(
                    #     f"[IMPOSSIBLE] veh={vid} edge={start_edge} dest={vehicle.destination} "
                    #     f"chosen='{action}' valid_dirs={valid_dirs} state_bits={state[0][2:8].tolist()}"
                    # )
                    action = None  # trigger fallback below

                #---------- PROGRESS / LOOP GUARD ----------
                #Evaluate proposed action and alternatives using shortest-path distance
                dest_id = vehicle.destination

                def score_dir(dir_char):
                    nxt = outgoing[dir_char]
                    d = self._dist_to_dest(nxt, dest_id)
                    return d, nxt

                # best possible move from here (by distance-to-dest)
                best_dir = valid_dirs[0]
                best_next = outgoing[best_dir]
                best_d = self._dist_to_dest(best_next, dest_id)
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

                recent_repeat = 0
                if prop_next is not None:
                    recent_repeat = sum(1 for edge in self._recent_edges[vid] if edge == prop_next)

                # update best distance achieved so far
                if best_d < self._best_dist[vid]:
                    self._best_dist[vid] = best_d

                # when to override:
                # 1) proposed has no path
                # 2) proposed repeats too much
                # 3) proposed is much worse than best available
                no_path_override = prop_d == float("inf")
                repeat_override = visit >= 3 or recent_repeat >= self.loop_repeat_threshold
                distance_override = prop_d > best_d + self.distance_slack

                override = no_path_override or repeat_override or distance_override

                # If the model already chose the best available direction, avoid
                # logging/counting a no-op override. This keeps metrics meaningful
                # and reduces noisy repeated [OVERRIDE] messages.
                if action == best_dir and prop_d != float("inf"):
                    override = False

                if override:
                    self._metrics["overrides"] += 1
                    if repeat_override:
                        self._metrics["loop_overrides"] += 1
                    if distance_override:
                        self._metrics["distance_overrides"] += 1
                    # print(
                    #     f"[OVERRIDE] veh={vid} edge={start_edge} dest={dest_id} "
                    #     f"chosen='{action}' prop_d={prop_d} best_dir='{best_dir}' best_d={best_d} "
                    #     f"visit={visit} recent_repeat={recent_repeat}"
                    # )
                    action = best_dir
                    prop_next = best_next
                    prop_d = best_d
                #------------------------------------------

                if action not in outgoing:
                    self._metrics["impossible_action_overrides"] += 1
                    action = valid_dirs[0]
                    # print(
                    #     f"[FALLBACK] veh={vid} edge={start_edge} dest={dest_id} "
                    #     f"picked safe default action='{action}'"
                    # )

                # print(f"For vehicle {vid},Choice for " + str(start_edge) + " is: " + str(action))

                target_edge = outgoing[action]
                self._recent_edges[vid].append(target_edge)
                start_edge = target_edge
                decision_list.append(action)

                i += 1
                if start_edge == vehicle.destination:
                    break

            if wrong_decision:
                continue

            local_targets[vehicle.vehicle_id] = self.compute_local_target(decision_list, vehicle)

        if self._metrics["decisions"] > 0:
            snapshot = (
                self._metrics["decisions"],
                self._metrics["overrides"],
                self._metrics["loop_overrides"],
                self._metrics["distance_overrides"],
                self._metrics["impossible_action_overrides"],
            )
            if snapshot == self._last_metrics_snapshot:
                return local_targets

            self._last_metrics_snapshot = snapshot
            ratio = self._metrics["overrides"] / float(self._metrics["decisions"])
            # print(
            #     "[Q-METRICS] decisions={} overrides={} override_ratio={:.2%} "
            #     "loop_overrides={} distance_overrides={} impossible_action_overrides={}".format(
            #         self._metrics["decisions"],
            #         self._metrics["overrides"],
            #         ratio,
            #         self._metrics["loop_overrides"],
            #         self._metrics["distance_overrides"],
            #         self._metrics["impossible_action_overrides"],
            #     )
            # )

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
    def getState(self, vehicle_id, edge_now, destination_edge):
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

        lane_idx_norm = 0.0
        lane_count_norm = 0.0
        dist_to_end_norm = 0.0
        try:
            lane_id = traci.vehicle.getLaneID(vehicle_id)
            lane_idx = traci.vehicle.getLaneIndex(vehicle_id)
            lane_count = max(traci.edge.getLaneNumber(en), 1)
            lane_len = traci.lane.getLength(lane_id)
            lane_pos = traci.vehicle.getLanePosition(vehicle_id)
            dist_to_end = max(lane_len - lane_pos, 0.0)

            lane_idx_norm = lane_idx / max(lane_count - 1, 1)
            lane_count_norm = min(lane_count, 6) / 6.0
            dist_to_end_norm = min(dist_to_end, 200.0) / 200.0
        except traci.TraCIException:
            # Vehicle may have arrived/teleported between steps. Keep neutral defaults.
            pass

        state.extend([lane_idx_norm, lane_count_norm, dist_to_end_norm])
        state.extend(self._compute_deadline_features(vehicle_id))

        for edge_now in self.connection_info.edge_list:
            car_num = traci.edge.getLastStepVehicleNumber(edge_now)
            density = car_num / self.connection_info.edge_length_dict[edge_now]
            state.append(density)

        state = np.reshape(state, [1, len(state)])
        return state
