import json
import math
import os
import random
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from xml.dom.minidom import parse

import numpy as np
from keras.layers import Dense
from keras.models import Sequential, clone_model
from keras.optimizers import Adam

from controller.RouteController import RouteController
from core.Util import ConnectionInfo
from core.routing_shared import SharedRoutingLogic
from core.target_vehicles_generation_protocols import target_vehicles_generator

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("No environment variable SUMO_HOME!")

from sumolib import checkBinary
import traci

MAX_SIMULATION_STEPS = 3200


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.buffer = deque(maxlen=self.capacity)

    def add(self, state, action, reward, next_state, done, next_mask):
        self.buffer.append((state, action, reward, next_state, done, next_mask))

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in idx]

    def __len__(self):
        return len(self.buffer)


class TrainingRouteHelper(RouteController):
    def __init__(self, connection_info):
        super().__init__(connection_info)

    def make_decisions(self, vehicles, connection_info):
        return {}


class DQNTrainer:
    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.0007,
        gamma=0.98,
        epsilon=1.0,
        epsilon_decay=0.992,
        epsilon_min=0.05,
        replay_capacity=40000,
        batch_size=64,
        replay_warmup=1200,
        target_update_every=250,
    ):
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon_min = float(epsilon_min)
        self.batch_size = int(batch_size)
        self.replay_warmup = max(int(replay_warmup), self.batch_size)
        self.target_update_every = int(target_update_every)
        self.train_steps = 0

        self.memory = ReplayBuffer(replay_capacity)
        self.model = self._build_model(learning_rate)
        self.target_model = clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

    def _build_model(self, lr):
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=(self.state_size,)))
        model.add(Dense(96, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=lr))
        return model

    def select_action(self, state, mask):
        valid = np.where(mask > 0.0)[0]
        if len(valid) == 0:
            return 0
        if random.random() < self.epsilon:
            return int(random.choice(valid))
        q = self.model(state, training=False).numpy()[0]
        masked = np.full_like(q, -1e9)
        masked[valid] = q[valid]
        return int(np.argmax(masked))

    def remember(self, state, action, reward, next_state, done, next_mask):
        self.memory.add(state, int(action), float(reward), next_state, bool(done), np.array(next_mask, dtype=np.float32))

    def replay(self):
        if len(self.memory) < self.replay_warmup:
            return
        batch = self.memory.sample(self.batch_size)
        states = np.vstack([b[0] for b in batch])
        actions = np.asarray([b[1] for b in batch], dtype=np.int32)
        rewards = np.asarray([b[2] for b in batch], dtype=np.float32)
        next_states = np.vstack([b[3] for b in batch])
        dones = np.asarray([b[4] for b in batch], dtype=np.float32)
        next_masks = np.asarray([b[5] for b in batch], dtype=np.float32)

        q = self.model(states, training=False).numpy()
        q_next_online = self.model(next_states, training=False).numpy()
        q_next_target = self.target_model(next_states, training=False).numpy()

        bootstrap = np.zeros(self.batch_size, dtype=np.float32)
        for i in range(self.batch_size):
            valid = np.where(next_masks[i] > 0.0)[0]
            if dones[i] >= 1.0 or len(valid) == 0:
                continue
            masked = np.full(self.action_size, -1e9, dtype=np.float32)
            masked[valid] = q_next_online[i, valid]
            a_star = int(np.argmax(masked))
            bootstrap[i] = q_next_target[i, a_star]

        target = q.copy()
        target[np.arange(self.batch_size), actions] = rewards + (1.0 - dones) * self.gamma * bootstrap
        self.model.train_on_batch(states, target)

        self.train_steps += 1
        if self.train_steps % self.target_update_every == 0:
            self.target_model.set_weights(self.model.get_weights())


@dataclass
class DecisionTransition:
    state: np.ndarray
    action: int
    edge: str
    expected_next_edge: str
    candidate: object
    step: int
    mask: np.ndarray
    mismatch_happened: bool = False


class RLTrainingPipeline:
    def __init__(
        self,
        sumocfg_path,
        model_output_path,
        episodes=50,
        spawn_interval=4.0,
        seed_with_episode=True,
        epsilon_decay=0.992,
        epsilon_min=0.06,
        gamma=0.98,
        replay_capacity=40000,
        batch_size=64,
        replay_warmup=1200,
        train_every=12,
        grad_steps=1,
        rolling_window=40,
        target_pattern=2,
        debug_exit_diagnostics=False,
        debug_exit_diagnostics_limit=20,
    ):
        self.sumocfg_path = sumocfg_path
        self.model_output_path = model_output_path
        self.episodes = int(episodes)
        self.spawn_interval = float(spawn_interval)
        self.seed_with_episode = bool(seed_with_episode)
        self.target_pattern = target_pattern
        self.train_every = int(train_every)
        self.grad_steps = int(grad_steps)
        self.rolling_window = int(rolling_window)

        self.sumocfg_dir = os.path.dirname(sumocfg_path)
        self.net_file, self.route_file = self.parse_sumocfg(sumocfg_path)
        self.connection_info = ConnectionInfo(os.path.join(self.sumocfg_dir, self.net_file))
        self.route_helper = TrainingRouteHelper(self.connection_info)
        self.shared = SharedRoutingLogic(self.connection_info, slot_count=8)

        self.trainer = DQNTrainer(
            state_size=self.shared.state_size,
            action_size=self.shared.slot_count,
            gamma=gamma,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            replay_capacity=replay_capacity,
            batch_size=batch_size,
            replay_warmup=replay_warmup,
        )

        self.metrics_file = os.path.join(self.sumocfg_dir, "training_metrics.jsonl")

    def parse_sumocfg(self, sumocfg_path):
        dom = parse(sumocfg_path)
        net_file = dom.getElementsByTagName('net-file')[0].attributes['value'].nodeValue
        route_file = dom.getElementsByTagName('route-files')[0].attributes['value'].nodeValue
        return net_file, route_file

    def _generate_vehicles(self, episode):
        generator = target_vehicles_generator(os.path.join(self.sumocfg_dir, self.net_file))
        vehicle_count_controlled = 100
        vehicle_count_random = 180
        seed = episode if self.seed_with_episode else None
        route_path = os.path.join(self.sumocfg_dir, self.route_file)

        vehicles = generator.generate_vehicles(
            vehicle_count_controlled,
            vehicle_count_random,
            self.target_pattern,
            route_path,
            os.path.join(self.sumocfg_dir, self.net_file),
            spawn_interval=self.spawn_interval,
            seed=seed,
        )
        return {v.vehicle_id: v for v in vehicles} if vehicles else {}

    def _teleport_ids(self):
        ids = set()
        for fn in ("getStartingTeleportIDList", "getEndingTeleportIDList"):
            try:
                ids.update(getattr(traci.simulation, fn)())
            except Exception:
                pass
        try:
            ids.update(traci.vehicle.getTeleportingList())
        except Exception:
            pass
        return ids

    def _apply_route(self, vehicle_id: str, final_dest: str, local_target: str):
        try:
            if local_target != final_dest:
                traci.vehicle.setVia(vehicle_id, [local_target])
            else:
                traci.vehicle.setVia(vehicle_id, [])
            traci.vehicle.changeTarget(vehicle_id, final_dest)
            return True
        except traci.exceptions.TraCIException:
            try:
                traci.vehicle.setVia(vehicle_id, [])
                traci.vehicle.changeTarget(vehicle_id, final_dest)
                return False
            except traci.exceptions.TraCIException:
                return False

    def _close_transition(self, tid, transition, vehicle, current_edge, step, arrived, teleported, histories, ep_metrics):
        loop_hits = sum(1 for e in histories["recent_edges"][tid] if e == current_edge)
        trans_hits = sum(1 for t in histories["recent_transitions"][tid] if t == (transition.edge, current_edge))
        reward, done, _terms = self.shared.compute_transition_reward(
            vehicle=vehicle,
            prev_edge=transition.edge,
            current_edge=current_edge,
            selected_candidate=transition.candidate,
            expected_next_edge=transition.expected_next_edge,
            arrived=arrived,
            teleported=teleported,
            step=step,
            loop_hits=loop_hits,
            transition_hits=trans_hits,
            mismatch_happened=transition.mismatch_happened,
        )

        if (not arrived) and (not teleported):
            new_candidates, new_mask = self.shared.build_candidates(
                vehicle,
                tid,
                current_edge,
                vehicle.destination,
                histories["recent_edges"][tid],
                histories["recent_transitions"][tid],
                histories["mismatch_count"][tid],
            )
            next_state = self.shared.encode_observation(
                vehicle,
                tid,
                current_edge,
                vehicle.destination,
                new_candidates,
                new_mask,
                loop_score=float(loop_hits),
                mismatch_count=histories["mismatch_count"][tid],
            )
            next_mask = new_mask
        else:
            next_state = np.zeros((1, self.shared.state_size), dtype=np.float32)
            next_mask = np.zeros(self.shared.slot_count, dtype=np.float32)

        self.trainer.remember(transition.state, transition.action, reward, next_state, done, next_mask)
        ep_metrics["return"] += reward

    def run(self):
        sumo_binary = checkBinary('sumo')
        rolling = {
            "return": deque(maxlen=self.rolling_window),
            "completion": deque(maxlen=self.rolling_window),
            "on_time": deque(maxlen=self.rolling_window),
            "lateness": deque(maxlen=self.rolling_window),
        }

        with open(self.metrics_file, "w", encoding="utf-8") as _:
            pass

        for episode in range(self.episodes):
            vehicles = self._generate_vehicles(episode)
            if not vehicles:
                print(f"Episode {episode}: no vehicles generated, skipping.")
                continue

            traci.start([
                sumo_binary,
                "-c", self.sumocfg_path,
                "--quit-on-end",
            ])

            histories = {
                "recent_edges": defaultdict(lambda: deque(maxlen=16)),
                "recent_transitions": defaultdict(lambda: deque(maxlen=16)),
                "mismatch_count": defaultdict(int),
                "last_decision_step": defaultdict(lambda: -9999),
                "expected_next_edge": {},
                "open_transition": {},
            }

            ep = defaultdict(float)
            ep.update({
                "episode": episode,
                "steps": 0,
                "teleports": 0,
                "dead_end_failures": 0,
                "no_path_failures": 0,
                "loop_detections": 0,
                "loop_recoveries": 0,
                "route_lane_mismatches": 0,
                "edge_mismatches": 0,
                "forced_fallbacks": 0,
                "lane_interventions": 0,
                "decisions": 0,
                "arrived": 0,
                "arrived_ontime": 0,
                "lateness_total": 0.0,
                "return": 0.0,
            })
            controlled_ids = set(vehicles.keys())

            try:
                for step in range(MAX_SIMULATION_STEPS):
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        break

                    current_ids = set(traci.vehicle.getIDList())
                    for edge in self.connection_info.edge_list:
                        self.connection_info.edge_vehicle_count[edge] = traci.edge.getLastStepVehicleNumber(edge)

                    # detect teleports as terminal transitions
                    teleported = self._teleport_ids()
                    ep["teleports"] += len([x for x in teleported if x in controlled_ids])

                    for vid in list(current_ids):
                        if vid not in vehicles:
                            continue
                        vehicle = vehicles[vid]
                        edge = traci.vehicle.getRoadID(vid)
                        if edge not in self.connection_info.edge_index_dict:
                            continue

                        if vehicle.start_time <= 0.0:
                            vehicle.start_time = float(step)
                        vehicle.current_edge = edge
                        vehicle.current_speed = traci.vehicle.getSpeed(vid)

                        histories["recent_edges"][vid].append(edge)

                        expected = histories["expected_next_edge"].get(vid)
                        if expected and edge != expected:
                            histories["mismatch_count"][vid] += 1
                            ep["edge_mismatches"] += 1
                            if vid in histories["open_transition"]:
                                histories["open_transition"][vid].mismatch_happened = True
                        if expected and edge != histories["open_transition"].get(vid, DecisionTransition(None,0,edge,edge,None,0,np.zeros(1))).edge:
                            histories["expected_next_edge"].pop(vid, None)

                        if vid in histories["open_transition"]:
                            trn = histories["open_transition"][vid]
                            if edge != trn.edge:
                                self._close_transition(vid, trn, vehicle, edge, step, arrived=False, teleported=False, histories=histories, ep_metrics=ep)
                                histories["open_transition"].pop(vid, None)

                        if edge == vehicle.destination:
                            ep["arrived"] += 1
                            if step <= vehicle.deadline:
                                ep["arrived_ontime"] += 1
                            else:
                                ep["lateness_total"] += float(step - vehicle.deadline)
                            if vid in histories["open_transition"]:
                                trn = histories["open_transition"][vid]
                                self._close_transition(vid, trn, vehicle, edge, step, arrived=True, teleported=False, histories=histories, ep_metrics=ep)
                                histories["open_transition"].pop(vid, None)
                            continue

                        if vid in teleported and vid in histories["open_transition"]:
                            trn = histories["open_transition"][vid]
                            self._close_transition(vid, trn, vehicle, edge, step, arrived=False, teleported=True, histories=histories, ep_metrics=ep)
                            histories["open_transition"].pop(vid, None)
                            continue

                        if not self.shared.is_decision_point(vid, edge):
                            continue
                        if step - histories["last_decision_step"][vid] < 4:
                            continue

                        candidates, mask = self.shared.build_candidates(
                            vehicle,
                            vid,
                            edge,
                            vehicle.destination,
                            histories["recent_edges"][vid],
                            histories["recent_transitions"][vid],
                            histories["mismatch_count"][vid],
                        )
                        loop_score = float(sum(1 for e in histories["recent_edges"][vid] if e == edge))
                        if loop_score >= 3:
                            ep["loop_detections"] += 1
                        obs = self.shared.encode_observation(
                            vehicle,
                            vid,
                            edge,
                            vehicle.destination,
                            candidates,
                            mask,
                            loop_score=loop_score,
                            mismatch_count=histories["mismatch_count"][vid],
                        )

                        action = self.trainer.select_action(obs, mask)
                        candidate = self.shared.choose_safe_candidate(candidates, mask, action)
                        aligned = self.shared.apply_lane_alignment(vid, edge, candidate)
                        if aligned:
                            ep["lane_interventions"] += 1
                        if (not aligned) and (not candidate.lane_supported):
                            ep["route_lane_mismatches"] += 1
                            ep["forced_fallbacks"] += 1
                            replacement = None
                            for c in candidates:
                                if c.valid and c.min_lane_shift <= 1:
                                    replacement = c
                                    break
                            if replacement is not None:
                                candidate = replacement
                                ep["loop_recoveries"] += 1

                        local_target = self.shared.plan_local_target(edge, candidate.next_edge, vehicle.destination)
                        ok = self._apply_route(vid, vehicle.destination, local_target)
                        if not ok:
                            ep["forced_fallbacks"] += 1

                        histories["recent_transitions"][vid].append((edge, candidate.next_edge))
                        histories["expected_next_edge"][vid] = candidate.next_edge
                        histories["last_decision_step"][vid] = step
                        histories["open_transition"][vid] = DecisionTransition(
                            state=obs,
                            action=action,
                            edge=edge,
                            expected_next_edge=candidate.next_edge,
                            candidate=candidate,
                            step=step,
                            mask=mask,
                        )
                        ep["decisions"] += 1

                    traci.simulationStep()
                    ep["steps"] = step

                    if step % self.train_every == 0:
                        for _ in range(self.grad_steps):
                            self.trainer.replay()

                # close remaining open transitions at episode end
                for vid, trn in list(histories["open_transition"].items()):
                    vehicle = vehicles.get(vid)
                    if vehicle is None:
                        continue
                    edge = vehicle.current_edge if vehicle.current_edge else trn.edge
                    self._close_transition(vid, trn, vehicle, edge, int(ep["steps"]), arrived=False, teleported=False, histories=histories, ep_metrics=ep)

            finally:
                traci.close()

            total = float(len(controlled_ids)) if controlled_ids else 1.0
            completion = ep["arrived"] / total
            on_time = ep["arrived_ontime"] / total
            avg_lateness = ep["lateness_total"] / max(ep["arrived"] - ep["arrived_ontime"], 1.0)
            avg_return = ep["return"] / total
            avg_decisions = ep["decisions"] / total

            rolling["return"].append(avg_return)
            rolling["completion"].append(completion)
            rolling["on_time"].append(on_time)
            rolling["lateness"].append(avg_lateness)

            metrics_row = {
                "episode": episode,
                "step": int(ep["steps"]),
                "epsilon": self.trainer.epsilon,
                "replay_size": len(self.trainer.memory),
                "train_steps": self.trainer.train_steps,
                "average_return": avg_return,
                "completion_rate": completion,
                "on_time_completion_rate": on_time,
                "average_lateness": avg_lateness,
                "teleports": int(ep["teleports"]),
                "dead_end_failures": int(ep["dead_end_failures"]),
                "no_path_failures": int(ep["no_path_failures"]),
                "loop_detections": int(ep["loop_detections"]),
                "loop_recoveries": int(ep["loop_recoveries"]),
                "route_lane_mismatches": int(ep["route_lane_mismatches"]),
                "intended_actual_mismatches": int(ep["edge_mismatches"]),
                "forced_safe_fallbacks": int(ep["forced_fallbacks"]),
                "lane_change_interventions": int(ep["lane_interventions"]),
                "avg_decisions_per_vehicle": avg_decisions,
            }
            with open(self.metrics_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics_row) + "\n")

            roll = {k: sum(v) / len(v) for k, v in rolling.items() if len(v) > 0}
            print(
                f"Episode {episode:03d} | eps={self.trainer.epsilon:.3f} | "
                f"ret={avg_return:.3f} comp={completion:.3f} ontime={on_time:.3f} "
                f"late={avg_lateness:.2f} tele={int(ep['teleports'])} "
                f"mismatch={int(ep['edge_mismatches'])} fallback={int(ep['forced_fallbacks'])} "
                f"roll_comp={roll.get('completion', 0.0):.3f}"
            )

            self.trainer.epsilon = max(self.trainer.epsilon_min, self.trainer.epsilon * self.trainer.epsilon_decay)

        self.trainer.model.save(self.model_output_path)
