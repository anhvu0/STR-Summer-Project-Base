import numpy as np
import os
import sys
from xml.dom.minidom import parse
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from collections import deque
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

    def __init__(self, state_size, action_size, learning_rate = 0.001, gamma = 0.95, epsilon = 1.0, epsilon_decay = 0.995, epsilon_min = 0.05, replay_capacity=10000, batch_size = 64):
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
        self.memory = ReplayBuffer(replay_capacity)
        self.model = self.build_model(learning_rate)

    def build_model(self, learning_rate):
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate = learning_rate))
        return model
    
    def select_action(self, state, valid_actions):
        """
        Select an action with epsilon-greedy exploration
        :param valid_actions: List of valid actions at a specific edge
        """
        if not valid_actions:
            return random.randint(0, self.action_size - 1) # Something has to be returned otherwise it crashes -> no good
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
        if len(self.memory) < self.batch_size:
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

        self.model.fit(states, target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
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
        decision_horizon=6,
        destination_reward=200.0,
        deadline_penalty=100.0,
    ):
        """
        Args:
            sumocfg_path: SUMO config file path.
            model_output_path: Path to save the trained model.
            episodes: Number of training episodes.
            spawn_interval: Interval between vehicle spawns.
            decision_horizon: Number of actions to pad a decision list.
            destination_reward: Reward when reaching the destination.
            deadline_penalty: Penalty when missing the deadline.
        """
        self.sumocfg_path = sumocfg_path
        self.model_output_path = model_output_path
        self.episodes = episodes
        self.spawn_interval = spawn_interval
        self.decision_horizon = decision_horizon
        self.destination_reward = destination_reward
        self.deadline_penalty = deadline_penalty

        self.sumocfg_dir = os.path.dirname(sumocfg_path)
        self.net_file, self.route_file = self._parse_sumocfg(sumocfg_path)
        self.connection_info = ConnectionInfo(os.path.join(self.sumocfg_dir, self.net_file))
        self.route_helper = TrainingRouteHelper(self.connection_info)

        self.state_size = 1 + 6 + len(self.connection_info.edge_list)
        self.action_size = 6
        self.trainer = DQNTrainer(self.state_size, self.action_size)

    def _parse_sumocfg(self, sumocfg_path):
        """
        Parse the SUMO config file and return net and route filenames.
        """
        dom = parse(sumocfg_path)
        net_file_node = dom.getElementsByTagName('net-file')
        route_file_node = dom.getElementsByTagName('route-files')
        net_file = net_file_node[0].attributes['value'].nodeValue
        route_file = route_file_node[0].attributes['value'].nodeValue
        return net_file, route_file

    def _encode_state(self, edge_id):
        """
        Build a state vector for the given edge.
        """
        state = []
        state.append(self.connection_info.edge_index_dict[edge_id])
        for choice in self.route_helper.direction_choices:
            if choice in self.connection_info.outgoing_edges_dict[edge_id]:
                state.append(1)
            else:
                state.append(0)
        for edge_now in self.connection_info.edge_list:
            car_num = traci.edge.getLastStepVehicleNumber(edge_now)
            density = car_num / self.connection_info.edge_length_dict[edge_now]
            state.append(density)
        return np.reshape(state, [1, len(state)])

    def _valid_actions(self, edge_id):
        """
        Return action indices that are valid from the current edge.
        """
        valid = []
        for idx, choice in enumerate(self.route_helper.direction_choices):
            if choice in self.connection_info.outgoing_edges_dict[edge_id]:
                valid.append(idx)
        return valid

    def _build_decision_list(self, edge_id, initial_action):
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
                valid_actions = self._valid_actions(current_edge)
                action = random.choice(valid_actions) if valid_actions else initial_action
            direction = self.route_helper.direction_choices[action]
            if direction not in self.connection_info.outgoing_edges_dict[current_edge]:
                break
            decision_list.append(direction)
            current_edge = self.connection_info.outgoing_edges_dict[current_edge][direction]
        return decision_list

    def _compute_reward(self, vehicle, step, arrived):
        """
        Compute a reward based on travel time, congestion, and deadlines.
        """
        time_penalty = -1.0
        congestion = self.connection_info.edge_vehicle_count.get(vehicle.current_edge, 0)
        congestion_penalty = -(congestion / max(self.connection_info.edge_length_dict[vehicle.current_edge], 1.0))
        reward = time_penalty + congestion_penalty
        done = False
        if arrived:
            reward += self.destination_reward
            done = True
        if step > vehicle.deadline:
            reward -= self.deadline_penalty
            done = True
        return reward, done

    def _generate_episode_vehicles(self):
        """
        Generate controlled and uncontrolled vehicles for one training episode.
        """
        generator = target_vehicles_generator(os.path.join(self.sumocfg_dir, self.net_file))
        route_path = os.path.join(self.sumocfg_dir, self.route_file)
        vehicle_list = generator.generate_vehicles(
            num_target_vehicles=10,
            num_random_vehicles=30,
            pattern=3,
            target_xml_file=route_path,
            net_xml_file=os.path.join(self.sumocfg_dir, self.net_file),
            spawn_interval=self.spawn_interval,
        )
        if vehicle_list is None:
            raise RuntimeError(
                "Failed to generate vehicles. Check randomTrips.py output for errors."
            )
        return {str(vehicle.vehicle_id): vehicle for vehicle in vehicle_list}

    def run(self):
        """
        Run the full training loop across episodes.
        """
        sumo_binary = checkBinary('sumo')
        for episode in range(self.episodes):
            vehicles = self._generate_episode_vehicles()
            traci.start([
                sumo_binary,
                "-c",
                self.sumocfg_path,
                "--tripinfo-output",
                os.path.join(self.sumocfg_dir, "trips.trips.xml"),
                "--quit-on-end",
            ])
            last_state_action = {}
            try:
                for step in range(MAX_SIMULATION_STEPS):
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        break
                    self._update_edge_vehicle_counts()
                    vehicle_ids = set(traci.vehicle.getIDList())
                    for vehicle_id in vehicle_ids:
                        if vehicle_id not in vehicles:
                            continue
                        current_edge = traci.vehicle.getRoadID(vehicle_id)
                        if current_edge not in self.connection_info.edge_index_dict:
                            continue
                        vehicle = vehicles[vehicle_id]
                        if vehicle.current_edge != current_edge:
                            if vehicle.current_edge:
                                prev_state, prev_action = last_state_action.get(
                                    vehicle_id, (None, None)
                                )
                                if prev_state is not None:
                                    reward, done = self._compute_reward(
                                        vehicle, step, current_edge == vehicle.destination
                                    )
                                    next_state = self._encode_state(current_edge)
                                    self.trainer.remember(
                                        prev_state, prev_action, reward, next_state, done
                                    )
                            vehicle.current_edge = current_edge
                            vehicle.current_speed = traci.vehicle.getSpeed(vehicle_id)
                            if current_edge == vehicle.destination:
                                last_state_action.pop(vehicle_id, None)
                                continue
                            state = self._encode_state(current_edge)
                            action = self.trainer.select_action(
                                state, self._valid_actions(current_edge)
                            )
                            decision_list = self._build_decision_list(current_edge, action)
                            local_target = self.route_helper.compute_local_target(
                                decision_list, vehicle
                            )
                            traci.vehicle.changeTarget(vehicle_id, local_target)
                            last_state_action[vehicle_id] = (state, action)
                    traci.simulationStep()
                    self.trainer.replay()
            finally:
                traci.close()
            self.trainer.model.save(self.model_output_path)

    def _update_edge_vehicle_counts(self):
        """
        Update edge vehicle counts in connection_info.
        """
        for edge in self.connection_info.edge_list:
            self.connection_info.edge_vehicle_count[edge] = traci.edge.getLastStepVehicleNumber(edge)