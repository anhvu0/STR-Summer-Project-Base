from collections import defaultdict, deque
import os
from xml.dom.minidom import parse

import numpy as np
from keras.models import load_model
import traci

from controller.RouteController import RouteController
from core.routing_shared import SharedRoutingLogic


def parse_sumocfg(sumocfg_path):
    dom = parse(sumocfg_path)
    net_file = dom.getElementsByTagName('net-file')[0].attributes['value'].nodeValue
    return os.path.join(os.path.dirname(sumocfg_path), net_file)


class QLearningPolicy(RouteController):
    def __init__(self, vehicles, connection_info, model_file, net_xml_file=None):
        super().__init__(connection_info)
        self.model = load_model(model_file)
        self.vehicles = vehicles
        self.shared = SharedRoutingLogic(connection_info, slot_count=8)
        self.model_state_size = int(self.model.input_shape[-1])
        if self.model_state_size != self.shared.state_size:
            raise ValueError(
                f"Model input size {self.model_state_size} does not match shared state size {self.shared.state_size}."
            )

        self.recent_edges = defaultdict(lambda: deque(maxlen=14))
        self.recent_transitions = defaultdict(lambda: deque(maxlen=14))
        self.mismatch_count = defaultdict(int)
        self.expected_next_edge = {}
        self.last_decision_step = defaultdict(lambda: -9999)
        self.cooldown_steps = 4

        self.metrics = defaultdict(int)

    def _masked_argmax(self, q_values: np.ndarray, mask: np.ndarray) -> int:
        if mask.sum() <= 0:
            return 0
        masked = np.full_like(q_values, -1e9, dtype=np.float32)
        valid_idx = np.where(mask > 0.0)[0]
        masked[valid_idx] = q_values[valid_idx]
        return int(np.argmax(masked))

    def _check_mismatch(self, vehicle_id: str, current_edge: str):
        expected = self.expected_next_edge.get(vehicle_id)
        if expected and current_edge != expected:
            self.mismatch_count[vehicle_id] += 1
            self.metrics["edge_mismatch"] += 1
        self.expected_next_edge.pop(vehicle_id, None)

    def make_decisions(self, vehicles, connection_info):
        local_targets = {}
        now = int(traci.simulation.getTime())

        for vehicle in vehicles:
            vid = vehicle.vehicle_id
            edge = vehicle.current_edge
            self._check_mismatch(vid, edge)
            self.recent_edges[vid].append(edge)

            if edge == vehicle.destination:
                continue
            if now - self.last_decision_step[vid] < self.cooldown_steps:
                continue
            if not self.shared.is_decision_point(vid, edge):
                continue

            trans = self.recent_transitions[vid]
            candidates, mask = self.shared.build_candidates(
                vehicle=vehicle,
                vehicle_id=vid,
                edge_id=edge,
                destination_edge=vehicle.destination,
                recent_edges=self.recent_edges[vid],
                recent_transitions=trans,
                mismatch_count=self.mismatch_count[vid],
            )
            loop_score = float(sum(1 for e in self.recent_edges[vid] if e == edge))
            obs = self.shared.encode_observation(
                vehicle=vehicle,
                vehicle_id=vid,
                edge_id=edge,
                destination_edge=vehicle.destination,
                candidates=candidates,
                mask=mask,
                loop_score=loop_score,
                mismatch_count=self.mismatch_count[vid],
            )

            q = self.model.predict(obs, verbose=0)[0]
            selected_slot = self._masked_argmax(q, mask)
            candidate = self.shared.choose_safe_candidate(candidates, mask, selected_slot)

            aligned = self.shared.apply_lane_alignment(vid, edge, candidate)
            if not aligned and not candidate.lane_supported:
                self.metrics["lane_override"] += 1
                for c in candidates:
                    if c.valid and c.min_lane_shift <= 1:
                        candidate = c
                        break

            local_target = self.shared.plan_local_target(edge, candidate.next_edge, vehicle.destination)
            if local_target == edge:
                self.metrics["fallback_current_edge"] += 1
                continue

            self.expected_next_edge[vid] = candidate.next_edge
            self.last_decision_step[vid] = now
            self.recent_transitions[vid].append((edge, candidate.next_edge))
            self.metrics["decisions"] += 1
            local_targets[vid] = local_target

        return local_targets
