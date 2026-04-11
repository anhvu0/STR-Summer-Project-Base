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
        self.model_state_size = int(self.model.input_shape[-1])
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
        self.score_slack = 30.0
        self.deadline_deficit_override_slack = 2.0
        self.distance_tiebreak_scale = 0.05
        self.edge_embedding_dim = 8
        self.local_congestion_k = 6
        self.compact_state_size = (2 * self.edge_embedding_dim) + 6 + 3 + 3 + self.local_congestion_k
        self.legacy_state_size = 2 + 6 + 3 + 3 + len(self.connection_info.edge_list)
        self.use_compact_state = (self.model_state_size == self.compact_state_size)
        self.direction_mask_start = (2 * self.edge_embedding_dim) if self.use_compact_state else 2
        self._init_edge_embeddings(seed=1337)

    def _init_edge_embeddings(self, seed=1337):
        rng = np.random.default_rng(seed)
        self._edge_embeddings = {}
        for edge_id in self.connection_info.edge_list:
            emb = rng.normal(loc=0.0, scale=0.1, size=self.edge_embedding_dim).astype(np.float32)
            self._edge_embeddings[edge_id] = emb

    def _get_edge_embedding(self, edge_id):
        return self._edge_embeddings.get(
            edge_id,
            np.zeros(self.edge_embedding_dim, dtype=np.float32),
        )

    def _local_congestion_features(self, edge_id):
        lengths = self.connection_info.edge_length_dict
        current_density = traci.edge.getLastStepVehicleNumber(edge_id) / max(lengths.get(edge_id, 5.0), 5.0)
        outgoing = self.connection_info.outgoing_edges_dict.get(edge_id, {})
        outgoing_densities = [
            traci.edge.getLastStepVehicleNumber(next_edge) / max(lengths.get(next_edge, 5.0), 5.0)
            for next_edge in outgoing.values()
        ]

        all_densities = [
            traci.edge.getLastStepVehicleNumber(edge) / max(lengths.get(edge, 5.0), 5.0)
            for edge in self.connection_info.edge_list
        ]
        mean_global = float(np.mean(all_densities)) if len(all_densities) > 0 else 0.0
        std_global = float(np.std(all_densities)) if len(all_densities) > 0 else 0.0
        mean_out = float(np.mean(outgoing_densities)) if outgoing_densities else current_density
        max_out = float(np.max(outgoing_densities)) if outgoing_densities else current_density
        min_out = float(np.min(outgoing_densities)) if outgoing_densities else current_density

        return [
            current_density,
            mean_out,
            max_out,
            min_out,
            current_density - mean_global,
            std_global,
        ]


    


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

    def _estimate_eta(self, edge_id, dest_id):
        dist = self._dist_to_dest(edge_id, dest_id)
        if not np.isfinite(dist):
            return float("inf")
        return float(dist) / 8.0
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

                vehicle_obj = self.vehicles.get(str(vid))
                if vehicle_obj is not None:
                    now = traci.simulation.getTime()
                    deadline_window = max(float(vehicle_obj.deadline) - float(vehicle_obj.start_time), 1.0)
                    time_left = max(float(vehicle_obj.deadline) - float(now), 0.0)
                    urgency = 1.0 - min(time_left / deadline_window, 1.0)
                else:
                    urgency = 0.5
                flexibility = 1.0 - urgency

                def score_dir(dir_char):
                    nxt = outgoing[dir_char]
                    d = self._dist_to_dest(nxt, dest_id)
                    if not np.isfinite(d):
                        return float("inf"), float("inf"), float("inf"), nxt

                    now = traci.simulation.getTime()
                    time_left = max(float(vehicle_obj.deadline) - float(now), 0.0) if vehicle_obj is not None else 0.0
                    eta = self._estimate_eta(nxt, dest_id)
                    deadline_deficit = max(eta - time_left, 0.0) if np.isfinite(eta) else float("inf")

                    edge_count = traci.edge.getLastStepVehicleNumber(nxt)
                    edge_length = max(self.connection_info.edge_length_dict.get(nxt, 5.0), 5.0)
                    density = edge_count / edge_length
                    # Flexible vehicles should yield more aggressively to reduce congestion.
                    density_weight = 80.0 * (0.8 + flexibility)
                    congestion_externality = density_weight * density
                    score = (
                        10000.0 * deadline_deficit
                        + congestion_externality
                        + (self.distance_tiebreak_scale * float(d))
                    )
                    return score, deadline_deficit, congestion_externality, nxt

                # best possible move from here (distance + congestion score)
                best_dir = valid_dirs[0]
                best_next = outgoing[best_dir]
                best_d, best_deadline_deficit, best_congestion_externality, _ = score_dir(best_dir)
                for dch in valid_dirs:
                    d, ddl_deficit, cong_externality, nxt = score_dir(dch)
                    if d < best_d:
                        best_d = d
                        best_deadline_deficit = ddl_deficit
                        best_congestion_externality = cong_externality
                        best_dir = dch
                        best_next = nxt

                # model-proposed next
                if action is not None:
                    prop_next = outgoing[action]
                    prop_d, prop_deadline_deficit, prop_congestion_externality, _ = score_dir(action)
                else:
                    prop_next = None
                    prop_d = float("inf")
                    prop_deadline_deficit = float("inf")
                    prop_congestion_externality = float("inf")

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
                deadline_override = prop_deadline_deficit > (best_deadline_deficit + self.deadline_deficit_override_slack)
                congestion_override = (
                    (not deadline_override)
                    and (prop_congestion_externality > best_congestion_externality + self.score_slack)
                )
                distance_override = (
                    (not deadline_override)
                    and (not congestion_override)
                    and (prop_d > best_d + self.score_slack)
                )

                override = no_path_override or repeat_override or deadline_override or congestion_override or distance_override

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
        mask_start = self.direction_mask_start
        state_vals = state[0][mask_start:mask_start + 6]
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
        if self.use_compact_state:
            state.extend(self._get_edge_embedding(en).tolist())
            state.extend(self._get_edge_embedding(destination_edge).tolist())
        else:
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

        if self.use_compact_state:
            state.extend(self._local_congestion_features(en))
        else:
            for edge_now in self.connection_info.edge_list:
                car_num = traci.edge.getLastStepVehicleNumber(edge_now)
                density = car_num / self.connection_info.edge_length_dict[edge_now]
                state.append(density)

        state = np.reshape(state, [1, len(state)])
        return state
