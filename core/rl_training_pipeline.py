import numpy as np
import os
import sys
import math
from xml.dom.minidom import parse
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from collections import defaultdict, deque
import random
from controller.RouteController import RouteController
from core.Util import ConnectionInfo
from core.target_vehicles_generation_protocols import target_vehicles_generator

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("No environment variable SUMO_HOME!")

from sumolib import checkBinary
import traci
import sumolib

"""
In this file, we build a DQN network
"""

MAX_SIMULATION_STEPS = 2000 # This is the limit for each episode. Because vehicle might be stuck in infinite loop

class ReplayBuffer:
    """
    This is the replay buffer mechanism in DQN
    """
    def __init__(self, capacity):
        """
        :param capacity: Maximum number of transitions
        """
        self.buffer = deque(maxlen=capacity) # Use deque here so you can pop the first element later easily
        self.capacity = capacity

    def add(self, state, action, reward, next_state, done):
        """      
        Store one transition into the buffer
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class TrainingRouteHelper(RouteController):
    """
    Helper class to reuse compute_local_target during RL training and use connection_info
    """

    def __init__(self, connection_info):
        super().__init__(connection_info)

    def make_decisions(self, vehicles, connection_info):
        return {}

class DQNTrainer:
    """
    Deep Q-Network trainer for routing decisions
    """

    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.99,
        epsilon_min=0.05,
        replay_capacity=10000,
        batch_size=32,
        replay_warmup=1000,
    ):
        """
        :param learning_rate: Can be adjusted for further optimization
        :param gamma: Can be adjusted for further optimization
        :param epsilon: 1.0 allows free exploration
        :param epsilon_decay: epsilon value in next episode
        :param epsilon_min: minimum epsilon to ensure that there's still some chance for free exploration later
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.replay_warmup = max(int(replay_warmup), self.batch_size)
        self.memory = ReplayBuffer(replay_capacity)
        self.model = self.build_model(learning_rate)

    def build_model(self, learning_rate):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))      #May increase Dense for bigger network
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate = learning_rate))
        return model
    
    def select_action(self, state, valid_actions):
        """
        Select an action with epsilon-greedy exploration
        :param valid_actions: List of valid actions at a specific edge
        """
        if not valid_actions:
            return None
        if np.random.rand() <= self.epsilon: # Random to see if the agent should choose a new path
            return random.choice(valid_actions)
        q_values = self.model.predict(state, verbose = 0)[0]
        masked_values = np.full_like(q_values, -1e9)    #Make all q-values -1e9, then valid actions will update their according value, invalid actions will not be updated and stay negative
        for action in valid_actions:
            masked_values[action] = q_values[action]
        return int(np.argmax(masked_values))
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store 1 transition for replay
        """
        self.memory.add(state, action, reward, next_state, done)
    
    def replay(self):
        """
        Train the Q-network from replayed experiences. Update q-values of previous state based on the most recent one.
        """
        if len(self.memory) < self.replay_warmup:
            return
        minibatch = self.memory.sample(self.batch_size)
        states      = np.vstack([s[0] for s in minibatch])
        actions     = np.array([s[1] for s in minibatch], dtype=np.int32)
        rewards     = np.array([s[2] for s in minibatch], dtype=np.float32)
        next_states = np.vstack([s[3] for s in minibatch])
        dones       = np.array([s[4] for s in minibatch], dtype=np.bool_)

        q = self.model.predict(states, verbose=0)
        q_next = self.model.predict(next_states, verbose=0)

        target = q.copy()
        target[np.arange(self.batch_size), actions] = rewards + (1.0 - dones.astype(np.float32)) * self.gamma * np.max(q_next, axis=1)

        self.model.train_on_batch(states, target)

        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay
            
class RLTrainingPipeline:
    """
    Pipeline for training a routing policy with Deep Q-Learning.
    """

    def __init__(
        self,
        sumocfg_path,
        model_output_path,
        episodes=10,
        spawn_interval=2.0,
        seed_with_episode=True,
        decision_horizon=6,
        destination_reward=120.0,       #Adjustible
        deadline_penalty=100.0,
        on_time_arrival_bonus=20.0,
        teleport_penalty=-150.0,
        epsilon_decay=0.997,
        epsilon_min=0.10,
        gamma=0.97,
        replay_capacity=20000,
        batch_size=64,
        replay_warmup=2000,
        train_every=10,
        grad_steps=2,
        rolling_window=100,
        debug_exit_diagnostics=True,
        debug_exit_diagnostics_limit=20,
    ):
        """
        Args:
            sumocfg_path: SUMO config file path.
            model_output_path: Path to save the trained model.
            episodes: Number of training episodes.
            spawn_interval: Interval between vehicle spawns.
            seed_with_episode: Whether to use the episode number as random seed.
            decision_horizon: Number of actions to pad a decision list.
            destination_reward: Reward when reaching the destination.
            deadline_penalty: Penalty when missing the deadline.
            on_time_arrival_bonus: Extra reward for arriving before deadline.
            teleport_penalty: Terminal penalty for teleport events.
        """
        self.sumocfg_path = sumocfg_path
        self.model_output_path = model_output_path
        self.episodes = episodes
        self.spawn_interval = spawn_interval
        self.seed_with_episode = seed_with_episode
        self.decision_horizon = decision_horizon
        self.destination_reward = destination_reward
        self.deadline_penalty = deadline_penalty
        self.on_time_arrival_bonus = on_time_arrival_bonus
        self.teleport_penalty = teleport_penalty
        self.train_every = train_every
        self.grad_steps = grad_steps
        self.rolling_window = rolling_window
        self.debug_exit_diagnostics = debug_exit_diagnostics
        self.debug_exit_diagnostics_limit = max(int(debug_exit_diagnostics_limit), 0)
        self._distance_cache = {}
        self.progress_reward_scale = 1.0  # or 0.0 to disable progress term cheaply
        self.system_congestion_scale = 0.01  # tune this later
        self.loop_window = 12
        self.loop_repeat_penalty = 10.0

        self.sumocfg_dir = os.path.dirname(sumocfg_path)
        self.net_file, self.route_file = self.parse_sumocfg(sumocfg_path)

        self.net = sumolib.net.readNet(os.path.join(self.sumocfg_dir, self.net_file))

        self.connection_info = ConnectionInfo(os.path.join(self.sumocfg_dir, self.net_file))
        self.route_helper = TrainingRouteHelper(self.connection_info)

        self.state_size = 2 + 6 + 3 + len(self.connection_info.edge_list)
        self.action_size = 6
        self.trainer = DQNTrainer(
            self.state_size,
            self.action_size,
            gamma=gamma,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            replay_capacity=replay_capacity,
            batch_size=batch_size,
            replay_warmup=replay_warmup,
        )

    def _deadline_window(self, vehicle):
        """
        Return a strictly positive scheduling window based on
        (deadline - start_time).
        """
        return max(float(vehicle.deadline) - float(vehicle.start_time), 1.0)

    def _deadline_urgency(self, vehicle, step):
        """
        Convert deadline flexibility into an urgency score in [0, 1].

        Vehicles with smaller (deadline - start_time) or little time left
        have higher urgency and should be prioritized.
        """
        deadline_window = self._deadline_window(vehicle)
        time_left = max(float(vehicle.deadline) - float(step), 0.0)
        # 1.0 means no slack left, 0.0 means fully relaxed.
        return 1.0 - min(time_left / deadline_window, 1.0)

    def parse_sumocfg(self, sumocfg_path):
        """
        Parse the SUMO config file and return net and route filenames.
        """
        dom = parse(sumocfg_path)
        net_file_node = dom.getElementsByTagName('net-file')
        route_file_node = dom.getElementsByTagName('route-files')
        net_file = net_file_node[0].attributes['value'].nodeValue
        route_file = route_file_node[0].attributes['value'].nodeValue
        return net_file, route_file

    def encode_state(self, vehicle_id, edge_id, destination_edge):
        """
        Build a state vector for the given edge using cached per-step densities.
        """
        state = np.zeros(self.state_size, dtype=np.float32)
        state[0] = self.connection_info.edge_index_dict[edge_id]
        state[1] = self.connection_info.edge_index_dict[destination_edge]

        outgoing = self.connection_info.outgoing_edges_dict[edge_id]
        base = 2
        for i, choice in enumerate(self.route_helper.direction_choices):
            state[base + i] = 1.0 if choice in outgoing else 0.0

        # lane features
        lane_id = traci.vehicle.getLaneID(vehicle_id)
        lane_idx = traci.vehicle.getLaneIndex(vehicle_id)
        n_lanes = max(traci.edge.getLaneNumber(edge_id), 1)
        lane_len = traci.lane.getLength(lane_id)
        lane_pos = traci.vehicle.getLanePosition(vehicle_id)
        dist_to_end = max(lane_len - lane_pos, 0.0)

        lane_base = base + 6
        state[lane_base + 0] = lane_idx / max(n_lanes - 1, 1)
        state[lane_base + 1] = min(n_lanes, 6) / 6.0
        state[lane_base + 2] = min(dist_to_end, 200.0) / 200.0

        state[lane_base + 3:] = self._density_vec
        return state.reshape(1, -1)

    def valid_actions(self, edge_id):
        """
        Return action indices that are valid from the current edge.
        """
        valid = []
        for idx, choice in enumerate(self.route_helper.direction_choices):
            if choice in self.connection_info.outgoing_edges_dict[edge_id]:
                valid.append(idx)
        return valid
    
    def valid_actions_for_vehicle(self, vehicle_id, edge_id):
        """
        Valid actions from the vehicle's CURRENT LANE (not just edge-level).
        """
        lane_id = traci.vehicle.getLaneID(vehicle_id)
        lane_map = self.connection_info.lane_outgoing_edges_dict.get(lane_id, {})
        valid = []
        for idx, choice in enumerate(self.route_helper.direction_choices):
            if choice in lane_map:
                valid.append(idx)
        return valid
    
    def dist_to_end(self, vehicle_id):
        """
        Distance (meters) from the vehicle to the end of its current lane.
        """
        lane_id = traci.vehicle.getLaneID(vehicle_id)
        lane_len = traci.lane.getLength(lane_id)
        lane_pos = traci.vehicle.getLanePosition(vehicle_id)
        return max(lane_len - lane_pos, 0.0)

    def is_decision_point(self, edge_id, vehicle_id, dist_threshold=80.0):
        """
        Make routing decisions only when it matters:
        - the edge has > 1 outgoing option (real branch), AND
        - the vehicle is close enough to the junction (within dist_threshold meters)
        """
        outgoing = self.connection_info.outgoing_edges_dict.get(edge_id, {})
        if outgoing is None or len(outgoing) <= 1:
            return False
        return self.dist_to_end(vehicle_id) <= dist_threshold
    
    def adaptive_dist_threshold(self, edge_id, max_dist=200.0, ratio=0.6, min_dist=30.0):
        """
        Adaptive threshold: decide when within min(max_dist, ratio * edge_length),
        clamped to at least min_dist.
        """
        edge_len = self.connection_info.edge_length_dict.get(edge_id, None)
        if edge_len is None:
            return max_dist
        return max(min_dist, min(max_dist, ratio * float(edge_len)))

    def is_decision_point_adaptive(self, edge_id, vehicle_id, max_dist=200.0, ratio=0.6, min_dist=30.0):
        """
        Decide near junctions, but adapt threshold based on edge length so short edges are safe.
        """
        outgoing = self.connection_info.outgoing_edges_dict.get(edge_id, {})
        if not outgoing or len(outgoing) <= 1:
            return False

        dist_th = self.adaptive_dist_threshold(edge_id, max_dist=max_dist, ratio=ratio, min_dist=min_dist)
        return self.dist_to_end(vehicle_id) <= dist_th

    # =========================
    # Teleport detection helpers
    # =========================
    def get_teleport_ids(self):
        """
        Return a set of vehicle IDs that teleported this step.

        SUMO/TraCI API differs by version, so we try multiple methods.
        """
        teleported = set()

        # Most common in many SUMO versions:
        try:
            teleported.update(traci.simulation.getStartingTeleportIDList())
        except Exception:
            pass
        try:
            teleported.update(traci.simulation.getEndingTeleportIDList())
        except Exception:
            pass

        # Some versions expose a vehicle-level list:
        try:
            teleported.update(traci.vehicle.getTeleportingList())
        except Exception:
            pass

        return teleported

    def make_terminal_next_state(self, vehicle_id, edge_id, destination_edge):
        """
        Build a next_state even if the vehicle is in a weird edge after teleport.
        If edge_id is unknown, fall back to a zero state (safe for training).
        """
        try:
            if edge_id in self.connection_info.edge_index_dict and destination_edge in self.connection_info.edge_index_dict:
                return self.encode_state(vehicle_id, edge_id, destination_edge)
        except Exception:
            pass
        return np.zeros((1, self.state_size), dtype=np.float32)

    def build_decision_list(self, edge_id, initial_action):
        """
        Build a decision list that starts with the chosen action and is padded
        with random valid actions to ensure a viable local target.
        """
        decision_list = []
        current_edge = edge_id
        for _ in range(self.decision_horizon):
            if not self.connection_info.outgoing_edges_dict[current_edge]:
                break
            if not decision_list:
                action = initial_action
            else:
                valid_actions = self.valid_actions(current_edge)
                if not valid_actions:
                    break
                action = random.choice(valid_actions) if decision_list else initial_action
                if action not in valid_actions:
                    action = random.choice(valid_actions)
            direction = self.route_helper.direction_choices[action]
            if direction not in self.connection_info.outgoing_edges_dict[current_edge]:
                break
            decision_list.append(direction)
            current_edge = self.connection_info.outgoing_edges_dict[current_edge][direction]
        return decision_list
    

    def get_distance_to_destination(self, edge_id, destination_edge):
        """
        Return shortest-path cost from edge_id to destination_edge.
        Uses a cache because this is called frequently during training.
        """
        key = (edge_id, destination_edge)
        if key in self._distance_cache:
            return self._distance_cache[key]

        try:
            from_edge = self.net.getEdge(edge_id)
            to_edge = self.net.getEdge(destination_edge)
        except Exception:
            self._distance_cache[key] = math.inf
            return math.inf

        path_edges, path_cost = self.net.getShortestPath(from_edge, to_edge)
        distance = path_cost if path_edges is not None else math.inf
        self._distance_cache[key] = distance
        return distance
    
    def ensure_lane_for_direction(self, vehicle_id, edge_id, direction, min_dist=40.0, duration=50):
        """
        If current lane cannot do 'direction', try to change into a lane that can,
        as long as we aren't too close to the junction end.
        """
        lane_id = traci.vehicle.getLaneID(vehicle_id)
        lane_pos = traci.vehicle.getLanePosition(vehicle_id)
        lane_len = traci.lane.getLength(lane_id)
        dist_to_end = lane_len - lane_pos

        if dist_to_end < min_dist:
            return  # too late

        lane_ids = self.connection_info.edge_lane_ids.get(edge_id, [])
        for target_lane_index, ln_id in enumerate(lane_ids):
            lane_map = self.connection_info.lane_outgoing_edges_dict.get(ln_id, {})
            if direction in lane_map:
                curr_idx = traci.vehicle.getLaneIndex(vehicle_id)
                if curr_idx != target_lane_index:
                    traci.vehicle.changeLane(vehicle_id, target_lane_index, duration)
                return


    def compute_reward(self, vehicle, prev_edge, current_edge, step, arrived, repeated_recent_edges=0):
        """
        Compute a reward based on travel time, congestion, progress,
        and proper dead-end handling.
        """

        deadline_window = self._deadline_window(vehicle)
        urgency = self._deadline_urgency(vehicle, step)
        flexibility = 1.0 - urgency

        # ---- Base penalties ----
        time_penalty = -5.0
        congestion = self.connection_info.edge_vehicle_count.get(current_edge, 0)
        congestion_penalty = -(congestion / max(self.connection_info.edge_length_dict[current_edge], 5.0))
        # More flexible vehicles (larger deadline - start_time) should yield,
        # so congestion penalty is stronger for them.
        congestion_penalty *= (1.0 + flexibility)

        reward = time_penalty + congestion_penalty

        # ---- Global system congestion penalty (selfless term) ----
        total_vehicles = sum(self.connection_info.edge_vehicle_count.values())
        system_penalty = -self.system_congestion_scale * total_vehicles
        reward += system_penalty

        done = False

        # ---- Progress shaping ----
        prev_distance = self.get_distance_to_destination(prev_edge, vehicle.destination)
        curr_distance = self.get_distance_to_destination(current_edge, vehicle.destination)

        if math.isfinite(prev_distance) and math.isfinite(curr_distance):
            # Prioritize strict-deadline vehicles by giving urgency-dependent
            # progress shaping. Flexible vehicles receive lower progress reward.
            progress_scale = self.progress_reward_scale * (0.5 + urgency)
            progress_reward = (prev_distance - curr_distance) * progress_scale
            reward += progress_reward

            # If a strict vehicle is spending too long without progress,
            # add extra shaping penalty so it reaches destination sooner.
            if curr_distance >= prev_distance and urgency > 0.7:
                reward -= 5.0 * urgency

        # Penalize repeatedly entering edges seen in recent history.
        if repeated_recent_edges > 0:
            reward -= self.loop_repeat_penalty * repeated_recent_edges

        # If vehicle moved into a region with no path to destination
        if math.isfinite(prev_distance) and not math.isfinite(curr_distance):
            reward -= 100.0
            done = True

        # ---- Arrival handling ----
        if arrived:
            reward += self.destination_reward
            if step <= vehicle.deadline:
                reward += self.on_time_arrival_bonus
            done = True
            return reward, done

        # ---- Dead-end handling ----
        # If no outgoing edges AND this is not the destination
        outgoing = self.connection_info.outgoing_edges_dict.get(current_edge, {})
        if (not outgoing or len(outgoing) == 0) and current_edge != vehicle.destination:
            dead_end_penalty = -50.0
            reward += dead_end_penalty
            done = True

        # ---- Deadline handling ----
        if step > vehicle.deadline:
            reward -= self.deadline_penalty
            done = True
        else:
            # Escalate penalty when approaching the deadline, normalized by
            # (deadline - start_time) so strict deadlines are emphasized.
            remaining_ratio = max(float(vehicle.deadline) - float(step), 0.0) / deadline_window
            if remaining_ratio < 0.25:
                reward -= (0.25 - remaining_ratio) * 20.0

        return reward, done

    def generate_episode_vehicles(self, episode_seed=None):
        """
        Generate controlled and uncontrolled vehicles for one training episode.
        """
        generator = target_vehicles_generator(os.path.join(self.sumocfg_dir, self.net_file))
        route_path = os.path.join(self.sumocfg_dir, self.route_file)
        vehicle_list = generator.generate_vehicles(
            num_target_vehicles=10,
            num_random_vehicles=15,
            pattern=3,
            target_xml_file=route_path,
            net_xml_file=os.path.join(self.sumocfg_dir, self.net_file),
            spawn_interval=self.spawn_interval,
            seed=episode_seed,
        )
        if vehicle_list is None:
            raise RuntimeError(
                "Failed to generate vehicles. Check randomTrips.py output for errors."
            )
        return {str(vehicle.vehicle_id): vehicle for vehicle in vehicle_list}

    def run(self):
        """
        Run the full training loop across episodes, with cleaner decision timing:
        - Decide near junctions (decision points), not on every edge change
        - Mask actions by lane feasibility
        - Optionally force lane alignment for chosen direction
        - Close transitions at the next decision point
        """
        sumo_binary = checkBinary('sumo')
        rolling_teleport_events = deque(maxlen=self.rolling_window)
        rolling_teleported_controlled = deque(maxlen=self.rolling_window)
        rolling_completion_rate = deque(maxlen=self.rolling_window)
        rolling_avg_return = deque(maxlen=self.rolling_window)

        for episode in range(self.episodes):
            episode_seed = episode if self.seed_with_episode else None
            if episode_seed is not None:
                random.seed(episode_seed)
                np.random.seed(episode_seed)

            vehicles = self.generate_episode_vehicles(episode_seed=episode_seed)

            traci.start([
                sumo_binary,
                "-c", self.sumocfg_path,
                "--tripinfo-output", os.path.join(self.sumocfg_dir, "trips.trips.xml"),
                "--quit-on-end",
            ])

            # vehicle_id -> (state, action, decision_edge)
            last_state_action = {}
            # vehicle_id -> edge_id where we last issued a decision (prevents repeat decisions)
            last_decision_edge = {}
            recent_edge_history = defaultdict(lambda: deque(maxlen=self.loop_window))

            episode_return = 0.0
            episode_teleport_events = 0
            teleported_controlled_ids = set()
            arrived_ids = set()
            arrived_before_deadline_ids = set()
            arrived_non_global_ids = set()
            exited_without_destination_ids = set()
            total_controlled = len(vehicles)
            controlled_ids = set(vehicles.keys())
            last_seen_edge_by_vehicle = {}
            last_target_by_vehicle = {}
            arrived_debug_records = []

            try:
                for step in range(MAX_SIMULATION_STEPS):
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        break

                    self.update_edge_vehicle_counts(step, every=10)  # try 10–20
                    vehicle_ids = list(traci.vehicle.getIDList())

                    for vehicle_id in vehicle_ids:
                        if vehicle_id not in vehicles:
                            continue

                        current_edge = traci.vehicle.getRoadID(vehicle_id)
                        if current_edge not in self.connection_info.edge_index_dict:
                            continue

                        vehicle = vehicles[vehicle_id]
                        vehicle.current_edge = current_edge
                        vehicle.current_speed = traci.vehicle.getSpeed(vehicle_id)
                        last_seen_edge_by_vehicle[vehicle_id] = current_edge
                        recent_edge_history[vehicle_id].append(current_edge)

                        # arrived
                        if current_edge == vehicle.destination:
                            if vehicle_id not in arrived_ids:
                                arrived_ids.add(vehicle_id)
                                if step <= vehicle.deadline:
                                    arrived_before_deadline_ids.add(vehicle_id)
                            last_state_action.pop(vehicle_id, None)
                            last_decision_edge.pop(vehicle_id, None)
                            continue

                        # only decide at decision points
                        if not self.is_decision_point_adaptive(
                            current_edge,
                            vehicle_id,
                            max_dist=200.0,   # cap for long edges
                            ratio=0.6,        # 60% of edge length
                            min_dist=30.0     # don't go too tiny
                        ):
                            continue

                        # avoid repeating decisions multiple steps on same edge
                        if last_decision_edge.get(vehicle_id) == current_edge:
                            continue

                        # ---- close previous transition at this decision point ----
                        if vehicle_id in last_state_action:
                            prev_state, prev_action, prev_edge = last_state_action[vehicle_id]

                            repeated_recent_edges = sum(
                                1 for edge in recent_edge_history[vehicle_id] if edge == current_edge
                            )

                            reward, done = self.compute_reward(
                                vehicle,
                                prev_edge,
                                current_edge,
                                step,
                                arrived=False,
                                repeated_recent_edges=repeated_recent_edges
                            )

                            next_state = self.encode_state(vehicle_id, current_edge, vehicle.destination)
                            self.trainer.remember(prev_state, prev_action, reward, next_state, done)
                            episode_return += reward

                            if done:
                                last_state_action.pop(vehicle_id, None)
                                last_decision_edge[vehicle_id] = current_edge
                                continue

                        # ---- choose action (lane-feasible) ----
                        state = self.encode_state(vehicle_id, current_edge, vehicle.destination)

                        valid = self.valid_actions_for_vehicle(vehicle_id, current_edge)
                        action = self.trainer.select_action(state, valid)

                        if action is None:
                            # no feasible action from this lane; skip decision (or penalize if you prefer)
                            last_decision_edge[vehicle_id] = current_edge
                            continue

                        direction = self.route_helper.direction_choices[action]

                        # optional lane alignment (strongly recommended)
                        align_min_dist = min(
                            150.0,
                            0.6 * self.connection_info.edge_length_dict.get(current_edge, 250.0)
                        )

                        self.ensure_lane_for_direction(
                            vehicle_id,
                            current_edge,
                            direction,
                            min_dist=align_min_dist,
                            duration=80
                        )
                        # compute local target and apply routing
                        decision_list = self.build_decision_list(current_edge, action)
                        local_target = self.route_helper.compute_local_target(decision_list, vehicle)
                        traci.vehicle.changeTarget(vehicle_id, local_target)
                        last_target_by_vehicle[vehicle_id] = local_target

                        # store new transition start
                        last_state_action[vehicle_id] = (state, action, current_edge)
                        last_decision_edge[vehicle_id] = current_edge

                    traci.simulationStep()

                    # Vehicles removed by SUMO because they reached their current route target.
                    # This is where we can tell whether they ended at the global destination
                    # or were terminated at an intermediate/local target.
                    arrived_this_step = traci.simulation.getArrivedIDList()
                    for arrived_vehicle_id in arrived_this_step:
                        if arrived_vehicle_id not in vehicles:
                            continue

                        vehicle = vehicles[arrived_vehicle_id]
                        last_seen_edge = last_seen_edge_by_vehicle.get(arrived_vehicle_id, "<unknown>")
                        last_local_target = last_target_by_vehicle.get(arrived_vehicle_id, "<unset>")
                        reached_global_destination = (
                            last_seen_edge == vehicle.destination
                            or last_local_target == vehicle.destination
                        )

                        if reached_global_destination:
                            arrived_ids.add(arrived_vehicle_id)
                            if step <= vehicle.deadline:
                                arrived_before_deadline_ids.add(arrived_vehicle_id)
                        else:
                            arrived_non_global_ids.add(arrived_vehicle_id)

                        debug_record = {
                            "vehicle_id": arrived_vehicle_id,
                            "global_destination": vehicle.destination,
                            "last_seen_edge": last_seen_edge,
                            "last_local_target": last_local_target,
                            "reached_global_destination": reached_global_destination,
                        }
                        arrived_debug_records.append(debug_record)

                        # No more transitions should be open once SUMO removes the vehicle.
                        last_state_action.pop(arrived_vehicle_id, None)
                        last_decision_edge.pop(arrived_vehicle_id, None)

                    # =========================
                    # Teleport detection + terminal penalty
                    # =========================
                    teleported_ids = self.get_teleport_ids()
                    if teleported_ids:
                        episode_teleport_events += len(teleported_ids)
                        teleported_controlled_ids.update(tid for tid in teleported_ids if tid in vehicles)
                        for tid in list(teleported_ids):
                            if tid not in vehicles:
                                continue

                            # If we have an open transition for this vehicle, close it as terminal
                            if tid in last_state_action:
                                prev_state, prev_action, prev_edge = last_state_action[tid]
                                v = vehicles[tid]

                                # Try to get where it ended up; may fail if removed, so guard
                                try:
                                    tele_edge = traci.vehicle.getRoadID(tid)
                                except Exception:
                                    tele_edge = prev_edge

                                # Big penalty so agent learns to avoid situations leading to teleports
                                next_state = self.make_terminal_next_state(tid, tele_edge, v.destination)

                                self.trainer.remember(prev_state, prev_action, self.teleport_penalty, next_state, True)
                                episode_return += self.teleport_penalty

                                # Clear open transition
                                last_state_action.pop(tid, None)

                            # Prevent repeated “decision” bookkeeping for teleported cars
                            last_decision_edge.pop(tid, None)

                    if step % self.train_every == 0:
                        for _ in range(self.grad_steps):
                            self.trainer.replay()

            finally:
                completion_rate = (
                    len(arrived_before_deadline_ids) / float(total_controlled)
                    if total_controlled > 0 else 0.0
                )
                avg_return = episode_return / float(total_controlled) if total_controlled > 0 else 0.0

                rolling_teleport_events.append(float(episode_teleport_events))
                rolling_teleported_controlled.append(float(len(teleported_controlled_ids)))
                rolling_completion_rate.append(float(completion_rate))
                rolling_avg_return.append(float(avg_return))

                roll_tele_events = sum(rolling_teleport_events) / len(rolling_teleport_events)
                roll_tele_ctrl = sum(rolling_teleported_controlled) / len(rolling_teleported_controlled)
                roll_completion = sum(rolling_completion_rate) / len(rolling_completion_rate)
                roll_return = sum(rolling_avg_return) / len(rolling_avg_return)

                self.trainer.epsilon = max(
                    self.trainer.epsilon_min,
                    self.trainer.epsilon * self.trainer.epsilon_decay
                )
                print(
                    f"\n\nDone with episode {episode} | "
                    f"teleport_events={episode_teleport_events}, "
                    f"teleported_controlled={len(teleported_controlled_ids)}, "
                    f"completion_before_deadline={completion_rate:.3f}, "
                    f"avg_return={avg_return:.3f}"
                )
                print(
                    f"Rolling({len(rolling_teleport_events)}) | "
                    f"teleport_events/ep={roll_tele_events:.3f}, "
                    f"teleported_controlled/ep={roll_tele_ctrl:.3f}, "
                    f"completion_before_deadline={roll_completion:.3f}, "
                    f"avg_return={roll_return:.3f}\n"
                )

                # Controlled vehicles that left simulation without being marked
                # as arrived (global destination) or teleported.
                exited_without_destination_ids = (
                    controlled_ids - arrived_ids - teleported_controlled_ids - arrived_non_global_ids
                )
                print(
                    f"Controlled exit diagnostics | "
                    f"arrived={len(arrived_ids)}/{total_controlled}, "
                    f"arrived_before_deadline={len(arrived_before_deadline_ids)}/{total_controlled}, "
                    f"teleported_controlled={len(teleported_controlled_ids)}/{total_controlled}, "
                    f"arrived_non_global={len(arrived_non_global_ids)}/{total_controlled}, "
                    f"exited_without_destination={len(exited_without_destination_ids)}/{total_controlled}"
                )

                if self.debug_exit_diagnostics:
                    mismatched_arrivals = [
                        record for record in arrived_debug_records
                        if not record["reached_global_destination"]
                    ]
                    print(
                        f"Arrival debug | total_arrived={len(arrived_debug_records)}, "
                        f"arrived_at_non_global_target={len(mismatched_arrivals)}"
                    )

                    for record in mismatched_arrivals[:self.debug_exit_diagnostics_limit]:
                        print(
                            "  ARRIVED_NON_GLOBAL "
                            f"vehicle={record['vehicle_id']} "
                            f"last_seen_edge={record['last_seen_edge']} "
                            f"last_local_target={record['last_local_target']} "
                            f"global_destination={record['global_destination']}"
                        )

                    if len(mismatched_arrivals) > self.debug_exit_diagnostics_limit:
                        print(
                            "  ARRIVED_NON_GLOBAL ... "
                            f"{len(mismatched_arrivals) - self.debug_exit_diagnostics_limit} more vehicles"
                        )

                    for vehicle_id in sorted(exited_without_destination_ids)[:self.debug_exit_diagnostics_limit]:
                        vehicle = vehicles[vehicle_id]
                        print(
                            "  EXITED_WITHOUT_DEST "
                            f"vehicle={vehicle_id} "
                            f"last_seen_edge={last_seen_edge_by_vehicle.get(vehicle_id, '<unknown>')} "
                            f"last_local_target={last_target_by_vehicle.get(vehicle_id, '<unset>')} "
                            f"global_destination={vehicle.destination}"
                        )

                    if len(exited_without_destination_ids) > self.debug_exit_diagnostics_limit:
                        print(
                            "  EXITED_WITHOUT_DEST ... "
                            f"{len(exited_without_destination_ids) - self.debug_exit_diagnostics_limit} more vehicles"
                        )
                traci.close()

        self.trainer.model.save(self.model_output_path)

    def update_edge_vehicle_counts(self, step, every=10):
        if hasattr(self, "_last_density_step") and (step - self._last_density_step) < every:
            return  # reuse cached self._density_vec

        counts = self.connection_info.edge_vehicle_count
        edge_list = self.connection_info.edge_list
        lengths = self.connection_info.edge_length_dict

        for edge in edge_list:
            counts[edge] = traci.edge.getLastStepVehicleNumber(edge)

        self._density_vec = np.array(
            [counts[e] / max(lengths[e], 1e-6) for e in edge_list],
            dtype=np.float32
        )
        self._last_density_step = step
