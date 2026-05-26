from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import math

import numpy as np

from core.junction_decision_engine import DecisionContext, JunctionDecisionEngine, PendingDecision
from core.routing_graph import (
    RoutingGraphFeatureLayout,
    RoutingGraphObservation,
    RoutingGraphSpec,
)


@dataclass(frozen=True)
class DecisionMode:
    mode: str
    action: Optional[int] = None
    skip_reason: Optional[str] = None


@dataclass(frozen=True)
class PendingReleaseEvaluation:
    should_release: bool
    release_reason: Optional[str]
    release_as_timeout: bool
    grace_keep: bool
    progress_view: Dict[str, object]
    stall_age: int
    total_age: int
    avoid_reopen_action: Optional[int] = None
    preferred_replan_action: Optional[int] = None


@dataclass(frozen=True)
class ActionCorridorStats:
    next_edge: Optional[str]
    edges: Tuple[str, ...]
    mean_density: float
    max_density: float
    congested_edges: int
    eta: float
    next_distance: float
    best_distance: float
    trap_score: float
    revisit_hits: int
    score: float


@dataclass
class CoordinationReservationState:
    reserved_agents: float = 0.0
    next_edge_loads: Dict[str, float] = field(default_factory=dict)
    corridor_edge_loads: Dict[str, float] = field(default_factory=dict)
    destination_loads: Dict[str, float] = field(default_factory=dict)


class SharedDecisionPolicy:
    """
    Shared owner for policy-facing decision logic that wraps the junction executor.

    The engine remains the source of truth for route continuity and raw feasibility.
    This helper owns:
    - graph observation spec
    - decision-mode classification
    - policy candidate filtering
    - fallback selection mode
    - observe <-> route_pending pending construction
    - route_pending release evaluation
    """

    def __init__(
        self,
        connection_info,
        decision_engine: JunctionDecisionEngine,
        direction_choices: Sequence[str],
        *,
        density_scale_m: float = 100.0,
        max_simulation_steps: int = 2000,
    ):
        self.connection_info = connection_info
        self.decision_engine = decision_engine
        self.direction_choices = list(direction_choices)
        self.action_count = len(self.direction_choices)
        self.density_scale_m = float(density_scale_m)
        self.max_simulation_steps = max(int(max_simulation_steps), 1)
        self.proactive_brake_risk_speed_floor = 4.5
        self.proactive_brake_risk_density_threshold = 0.30
        self.proactive_brake_risk_high_density_threshold = 0.45
        self.lane_now_congestion_density_threshold = 0.34
        self.lane_now_congestion_relief_threshold = 0.18
        self.lane_now_near_junction_density_tighten = 0.10
        self.lane_now_near_junction_relief_tighten = 0.12
        self.lane_now_near_junction_distance_keep_extra = 12.0
        self.lane_now_replan_min_age_steps = 8
        self.lane_now_replan_low_speed_mps = 1.75
        route_pending_stall_steps = int(getattr(self.decision_engine, "route_pending_stall_steps", 8))
        self.lane_now_replan_deadlock_min_age_steps = max(
            int(self.decision_engine.pending_progress_timeout_steps),
            route_pending_stall_steps * 3,
        )
        self.lane_now_stale_timeout_min_stall_steps = max(
            int(self.decision_engine.route_pending_hard_timeout_steps),
            int(self.decision_engine.pending_progress_timeout_steps) * 2,
        )
        self.lane_now_stale_timeout_max_age_steps = max(
            int(self.decision_engine.route_pending_hard_timeout_steps) * 3,
            int(self.decision_engine.pending_progress_timeout_steps) * 5,
        )
        self.policy_detour_base_slack_m = 12.0
        self.policy_detour_relief_slack_m = 24.0
        self.policy_detour_score_relief = 0.28
        self.policy_detour_density_relief = 0.12
        self.lane_now_replan_distance_slack_m = 14.0
        self.lane_now_deadlock_distance_slack_m = 10.0
        self.coordination_global_feature_count = 4
        self.coordination_action_feature_count = 2
        self.coordination_next_edge_load_cap = 4.0
        self.coordination_corridor_load_cap = 8.0
        self.coordination_destination_load_cap = 8.0
        self.coordination_reserved_agents_cap = 16.0
        self.coordination_rank_penalty_scale = 0.45
        self.coordination_filter_pressure_margin = 0.85
        self.coordination_filter_distance_base_slack_m = 12.0
        self.coordination_filter_distance_cap_m = 32.0
        self.corridor_horizon_m = max(
            float(getattr(self.decision_engine, "default_fragment_horizon_m", 180.0)),
            120.0,
        )
        self.corridor_congestion_density_threshold = 0.32
        self.corridor_recent_revisit_window = 8
        self.vehicle_wait_time_clip_s = 120.0
        self.graph_node_static_feature_count = 5
        self.graph_node_dynamic_feature_count = 7
        self.graph_scalar_feature_count = 16
        self.graph_action_feature_count = 13
        self.graph_spec = self._build_graph_spec()

    def empty_coordination_state(self) -> CoordinationReservationState:
        return CoordinationReservationState()

    def _build_graph_spec(self) -> RoutingGraphSpec:
        edge_ids = tuple(str(edge_id) for edge_id in self.connection_info.edge_list)
        edge_id_to_index = {
            edge_id: idx for idx, edge_id in enumerate(edge_ids)
        }

        in_degree = {edge_id: 0 for edge_id in edge_ids}
        for edge_id in edge_ids:
            outgoing = self.connection_info.outgoing_edges_dict.get(edge_id, {}) or {}
            for next_edge in outgoing.values():
                next_edge = str(next_edge)
                if next_edge in in_degree:
                    in_degree[next_edge] += 1

        edge_lengths = np.array(
            [
                max(float(self.connection_info.edge_length_dict.get(edge_id, 5.0)), 5.0)
                for edge_id in edge_ids
            ],
            dtype=np.float32,
        )
        lane_counts = np.array(
            [
                max(len(self.connection_info.edge_lane_ids.get(edge_id, [])), 1)
                for edge_id in edge_ids
            ],
            dtype=np.float32,
        )
        out_degrees = np.array(
            [
                len(self.connection_info.outgoing_edges_dict.get(edge_id, {}) or {})
                for edge_id in edge_ids
            ],
            dtype=np.float32,
        )
        in_degrees = np.array(
            [float(in_degree.get(edge_id, 0)) for edge_id in edge_ids],
            dtype=np.float32,
        )

        max_length = max(float(np.percentile(edge_lengths, 95)), 5.0) if edge_lengths.size else 5.0
        static_features = np.zeros(
            (len(edge_ids), self.graph_node_static_feature_count),
            dtype=np.float32,
        )
        if edge_ids:
            static_features[:, 0] = np.clip(edge_lengths / max_length, 0.0, 1.0)
            static_features[:, 1] = np.clip(lane_counts / 6.0, 0.0, 1.0)
            static_features[:, 2] = np.clip(out_degrees / max(float(self.action_count), 1.0), 0.0, 1.0)
            static_features[:, 3] = np.clip(in_degrees / max(float(self.action_count), 1.0), 0.0, 1.0)
            static_features[:, 4] = (out_degrees <= 0.0).astype(np.float32)

        edge_pairs = set()
        for edge_id in edge_ids:
            src = edge_id_to_index[edge_id]
            edge_pairs.add((src, src))
            outgoing = self.connection_info.outgoing_edges_dict.get(edge_id, {}) or {}
            for next_edge in outgoing.values():
                next_edge = str(next_edge)
                dst = edge_id_to_index.get(next_edge)
                if dst is None:
                    continue
                edge_pairs.add((src, dst))
                edge_pairs.add((dst, src))

        sorted_pairs = sorted(edge_pairs)
        if sorted_pairs:
            edge_index = np.asarray(sorted_pairs, dtype=np.int64).T
        else:
            edge_index = np.empty((2, 0), dtype=np.int64)
        feature_layout = RoutingGraphFeatureLayout(
            node_static_dim=self.graph_node_static_feature_count,
            node_dynamic_dim=self.graph_node_dynamic_feature_count,
            scalar_dim=self.graph_scalar_feature_count,
            action_feature_dim=self.graph_action_feature_count,
            action_count=self.action_count,
        )
        return RoutingGraphSpec(
            edge_ids=edge_ids,
            edge_id_to_index=edge_id_to_index,
            edge_index=edge_index,
            node_static_features=static_features,
            feature_layout=feature_layout,
        )

    def coordination_priority(
        self,
        context: DecisionContext,
        *,
        destination: str,
        edge_density_fn: Optional[Callable[[str], float]] = None,
    ) -> Tuple[object, ...]:
        density = float(edge_density_fn(context.edge_id)) if edge_density_fn is not None else 0.0
        return (
            0.0 if context.commit_window else 1.0,
            float(context.dist_to_end),
            float(max(len(context.available_actions), 1)),
            -float(density),
            -float(context.speed),
            float(len(context.lane_feasible_now_actions)),
            str(destination),
            str(context.vehicle_id),
        )

    def _normalize_coordination_value(self, value: float, cap: float) -> float:
        cap = max(float(cap), 1.0)
        return float(np.clip(float(value) / cap, 0.0, 1.0))

    def coordination_features(
        self,
        *,
        context: DecisionContext,
        destination: str,
        coordination_state: Optional[CoordinationReservationState],
    ) -> Tuple[np.ndarray, np.ndarray]:
        global_features = np.zeros(self.coordination_global_feature_count, dtype=np.float32)
        action_features = np.zeros(
            self.coordination_action_feature_count * self.action_count,
            dtype=np.float32,
        )
        if coordination_state is None:
            return global_features, action_features

        outgoing_edges = tuple(
            str(edge_id)
            for edge_id in self.connection_info.outgoing_edges_dict.get(context.edge_id, {}).values()
            if edge_id
        )
        outgoing_next_edge_load = sum(
            float(coordination_state.next_edge_loads.get(edge_id, 0.0))
            for edge_id in outgoing_edges
        )
        max_outgoing_next_edge_load = max(
            [float(coordination_state.next_edge_loads.get(edge_id, 0.0)) for edge_id in outgoing_edges] or [0.0]
        )
        destination_load = float(coordination_state.destination_loads.get(destination, 0.0))
        current_edge_corridor_load = float(coordination_state.corridor_edge_loads.get(context.edge_id, 0.0))

        global_features[0] = self._normalize_coordination_value(
            coordination_state.reserved_agents,
            self.coordination_reserved_agents_cap,
        )
        global_features[1] = self._normalize_coordination_value(
            destination_load,
            self.coordination_destination_load_cap,
        )
        global_features[2] = self._normalize_coordination_value(
            max_outgoing_next_edge_load,
            self.coordination_next_edge_load_cap,
        )
        global_features[3] = self._normalize_coordination_value(
            current_edge_corridor_load + outgoing_next_edge_load,
            self.coordination_corridor_load_cap,
        )

        for action_idx in range(self.action_count):
            next_edge, corridor_edges = self._action_corridor_edges(
                edge_id=context.edge_id,
                action_idx=action_idx,
                destination=destination,
            )
            if next_edge is None:
                continue
            base = action_idx * self.coordination_action_feature_count
            next_edge_load = float(coordination_state.next_edge_loads.get(next_edge, 0.0))
            corridor_overlap = sum(
                float(coordination_state.corridor_edge_loads.get(edge_id, 0.0))
                for edge_id in corridor_edges
            )
            action_features[base + 0] = self._normalize_coordination_value(
                next_edge_load,
                self.coordination_next_edge_load_cap,
            )
            action_features[base + 1] = self._normalize_coordination_value(
                corridor_overlap,
                self.coordination_corridor_load_cap,
            )
        return global_features, action_features

    def coordination_pressure_score(
        self,
        *,
        context: DecisionContext,
        destination: str,
        action_idx: int,
        coordination_state: Optional[CoordinationReservationState],
    ) -> float:
        if coordination_state is None:
            return 0.0
        next_edge, corridor_edges = self._action_corridor_edges(
            edge_id=context.edge_id,
            action_idx=action_idx,
            destination=destination,
        )
        if next_edge is None:
            return 0.0

        next_edge_load = float(coordination_state.next_edge_loads.get(next_edge, 0.0))
        corridor_overlap = sum(
            float(coordination_state.corridor_edge_loads.get(edge_id, 0.0))
            for edge_id in corridor_edges
        ) / max(len(corridor_edges), 1)
        destination_load = float(coordination_state.destination_loads.get(destination, 0.0))
        required_shift = float(max(context.required_lane_shift.get(action_idx, 0), 0))
        return float(
            (1.0 * next_edge_load)
            + (0.35 * corridor_overlap)
            + (0.10 * destination_load)
            + (0.08 * required_shift)
        )

    def reserve_action(
        self,
        coordination_state: Optional[CoordinationReservationState],
        *,
        context: DecisionContext,
        destination: str,
        action_idx: int,
        weight: float = 1.0,
    ) -> None:
        if coordination_state is None:
            return
        weight = max(float(weight), 0.0)
        if weight <= 0.0:
            return
        next_edge, corridor_edges = self._action_corridor_edges(
            edge_id=context.edge_id,
            action_idx=action_idx,
            destination=destination,
        )
        if next_edge is None:
            return
        coordination_state.reserved_agents += weight
        coordination_state.destination_loads[destination] = (
            float(coordination_state.destination_loads.get(destination, 0.0)) + weight
        )
        coordination_state.next_edge_loads[next_edge] = (
            float(coordination_state.next_edge_loads.get(next_edge, 0.0)) + weight
        )
        for edge_id in corridor_edges:
            coordination_state.corridor_edge_loads[edge_id] = (
                float(coordination_state.corridor_edge_loads.get(edge_id, 0.0)) + weight
            )

    def seed_coordination_from_pending(
        self,
        coordination_state: Optional[CoordinationReservationState],
        pending_decisions: Dict[str, PendingDecision],
        *,
        current_step: Optional[int] = None,
        max_age_steps: Optional[int] = None,
    ) -> int:
        """
        Carry active cross-step commitments into this step's coordination state.

        Same-step reservations alone miss vehicles that already committed a route
        fragment in a previous step but have not reached the next strategic edge
        yet. Seeding these pendings lets the policy see near-future corridor load,
        not just vehicles already physically counted on an edge.
        """
        if coordination_state is None or not pending_decisions:
            return 0

        seeded = 0
        for pending in pending_decisions.values():
            if pending is None or getattr(pending, "context", None) is None:
                continue
            metadata = pending.metadata if isinstance(pending.metadata, dict) else {}
            if bool(metadata.get("decision_finalized", False)):
                continue
            if current_step is not None and max_age_steps is not None:
                age = max(int(current_step) - int(pending.decision_step), 0)
                if age > int(max_age_steps):
                    continue
            phase = self.pending_phase(pending)
            if phase == "route_pending":
                weight = 1.0
            elif phase == "observe_lane_change":
                weight = 0.5
            else:
                continue
            self.reserve_action(
                coordination_state,
                context=pending.context,
                destination=pending.destination,
                action_idx=int(pending.intended_action),
                weight=weight,
            )
            seeded += 1
        return int(seeded)

    def classify_decision(self, context: DecisionContext) -> DecisionMode:
        if context.forced_action is not None:
            return DecisionMode("forced", action=int(context.forced_action), skip_reason=context.skip_reason)
        if context.skip_reason:
            return DecisionMode("skip", skip_reason=context.skip_reason)
        if self.decision_engine.is_decision_open(context):
            return DecisionMode("open")
        return DecisionMode("hold")

    def _action_corridor_edges(
        self,
        *,
        edge_id: str,
        action_idx: int,
        destination: str,
    ) -> Tuple[Optional[str], Tuple[str, ...]]:
        next_edge = self.decision_engine.get_next_edge(edge_id, action_idx)
        if next_edge is None:
            return None, ()

        build_route_fragment = getattr(self.decision_engine, "build_route_fragment", None)
        if callable(build_route_fragment):
            try:
                fragment, _, error = build_route_fragment(
                    edge_id,
                    action_idx,
                    destination,
                    horizon_m=self.corridor_horizon_m,
                )
            except TypeError:
                fragment, _, error = build_route_fragment(edge_id, action_idx, destination)
            except Exception:
                fragment, error = None, "fragment_error"
            if fragment and not error:
                return next_edge, tuple(str(edge) for edge in fragment if edge)

        return next_edge, (str(next_edge),)

    def action_corridor_stats(
        self,
        *,
        edge_id: str,
        action_idx: int,
        destination: str,
        edge_density_fn: Callable[[str], float],
        distance_fn: Optional[Callable[[str, str], float]] = None,
        eta_fn: Optional[Callable[[str, str], float]] = None,
        recent_history: Optional[Sequence[str]] = None,
    ) -> Optional[ActionCorridorStats]:
        next_edge, corridor_edges = self._action_corridor_edges(
            edge_id=edge_id,
            action_idx=action_idx,
            destination=destination,
        )
        if next_edge is None:
            return None

        if not corridor_edges:
            corridor_edges = (str(next_edge),)

        edge_length_lookup = getattr(self.connection_info, "edge_length_dict", {}) or {}
        outgoing_lookup = getattr(self.connection_info, "outgoing_edges_dict", {}) or {}

        densities = []
        weights = []
        finite_distances = []
        for edge in corridor_edges:
            densities.append(max(float(edge_density_fn(edge)), 0.0))
            weights.append(max(float(edge_length_lookup.get(edge, 40.0)), 5.0))
            if distance_fn is not None:
                distance = float(distance_fn(edge, destination))
                if math.isfinite(distance):
                    finite_distances.append(distance)

        if not densities:
            return None

        mean_density = float(
            np.average(
                np.array(densities, dtype=np.float32),
                weights=np.array(weights, dtype=np.float32),
            )
        )
        max_density = float(np.max(densities))
        congested_edges = int(
            sum(1 for density in densities if density >= self.corridor_congestion_density_threshold)
        )

        next_distance = float("inf")
        if distance_fn is not None:
            next_distance = float(distance_fn(next_edge, destination))

        eta = float("inf")
        if eta_fn is not None:
            eta = float(eta_fn(next_edge, destination))
        if not math.isfinite(eta):
            eta = (next_distance / 8.0) if math.isfinite(next_distance) else float("inf")

        best_distance = min(finite_distances) if finite_distances else next_distance

        recent_window = tuple(
            str(edge)
            for edge in list(recent_history or [])[-self.corridor_recent_revisit_window:]
        )
        recent_set = set(recent_window)
        revisit_hits = int(sum(1 for edge in corridor_edges[:3] if edge in recent_set))

        trap_score = 0.0
        for depth, edge in enumerate(corridor_edges[:3], start=1):
            out_degree = len(outgoing_lookup.get(edge, {}) or {})
            if edge != destination and out_degree == 0:
                trap_score += 6.0 / float(depth)
                break
            if edge != destination and out_degree == 1:
                trap_score += 1.4 / float(depth)
            if edge in recent_set:
                trap_score += 1.6 / float(depth)

        score = (
            0.95 * mean_density
            + 0.75 * max_density
            + 0.18 * float(congested_edges)
            + 0.22 * trap_score
            + 0.12 * float(revisit_hits)
        )
        if math.isfinite(eta):
            score += 0.01 * float(eta)
        else:
            score += 25.0

        return ActionCorridorStats(
            next_edge=str(next_edge),
            edges=tuple(corridor_edges),
            mean_density=mean_density,
            max_density=max_density,
            congested_edges=congested_edges,
            eta=eta,
            next_distance=next_distance,
            best_distance=best_distance,
            trap_score=float(trap_score),
            revisit_hits=revisit_hits,
            score=float(score),
        )

    def _normalize_wait_time(self, wait_time_s: float) -> float:
        return float(
            np.clip(
                float(max(wait_time_s, 0.0)) / max(float(self.vehicle_wait_time_clip_s), 1.0),
                0.0,
                1.0,
            )
        )

    def encode_graph_observation(
        self,
        *,
        edge_id: str,
        destination_edge: str,
        context: DecisionContext,
        node_density_vector: np.ndarray,
        destination_eta_vector: np.ndarray,
        edge_density_fn: Callable[[str], float],
        eta_fn: Callable[[str, str], float],
        social_cost_fn: Callable[[str, int, str], float],
        distance_fn: Optional[Callable[[str, str], float]] = None,
        global_density_stats: Optional[Tuple[float, float, float]] = None,
        step: Optional[int] = None,
        vehicle_start_time: Optional[float] = None,
        vehicle_wait_time_fn: Optional[Callable[[str], float]] = None,
        coordination_state: Optional[CoordinationReservationState] = None,
        recent_history: Optional[Sequence[str]] = None,
    ) -> RoutingGraphObservation:
        edge_lookup = self.graph_spec.edge_id_to_index
        current_idx = edge_lookup.get(str(edge_id))
        destination_idx = edge_lookup.get(str(destination_edge))
        if current_idx is None or destination_idx is None:
            return self.graph_spec.zero_observation()

        density_vector = np.asarray(node_density_vector, dtype=np.float32).reshape(-1)
        eta_vector = np.asarray(destination_eta_vector, dtype=np.float32).reshape(-1)
        if density_vector.shape[0] != self.graph_spec.node_count:
            raise ValueError(
                "node_density_vector has length {} but graph has {} nodes.".format(
                    density_vector.shape[0],
                    self.graph_spec.node_count,
                )
            )
        if eta_vector.shape[0] != self.graph_spec.node_count:
            raise ValueError(
                "destination_eta_vector has length {} but graph has {} nodes.".format(
                    eta_vector.shape[0],
                    self.graph_spec.node_count,
                )
            )

        if global_density_stats is None:
            mean_global = float(np.mean(density_vector)) if density_vector.size else 0.0
            std_global = float(np.std(density_vector)) if density_vector.size else 0.0
            occupied = density_vector[density_vector > 0.0]
            p95_global = float(np.percentile(occupied, 95)) if occupied.size else 0.0
        else:
            if len(global_density_stats) == 2:
                mean_global = float(global_density_stats[0])
                std_global = float(global_density_stats[1])
                occupied = density_vector[density_vector > 0.0]
                p95_global = float(np.percentile(occupied, 95)) if occupied.size else 0.0
            else:
                mean_global = float(global_density_stats[0])
                std_global = float(global_density_stats[1])
                p95_global = float(global_density_stats[2])

        node_dynamic_features = np.zeros(
            (self.graph_spec.node_count, self.graph_node_dynamic_feature_count),
            dtype=np.float32,
        )
        node_dynamic_features[:, 0] = np.clip(density_vector, 0.0, 1.0)
        node_dynamic_features[:, 1] = np.clip(density_vector - mean_global, -1.0, 1.0)
        finite_eta_mask = np.isfinite(eta_vector)
        node_dynamic_features[:, 2] = 1.0
        node_dynamic_features[finite_eta_mask, 2] = np.clip(
            eta_vector[finite_eta_mask] / float(self.max_simulation_steps),
            0.0,
            1.0,
        )
        node_dynamic_features[current_idx, 3] = 1.0
        node_dynamic_features[destination_idx, 4] = 1.0

        if coordination_state is not None:
            for load_edge, load_value in coordination_state.corridor_edge_loads.items():
                load_idx = edge_lookup.get(str(load_edge))
                if load_idx is not None:
                    node_dynamic_features[load_idx, 5] = self._normalize_coordination_value(
                        load_value,
                        self.coordination_corridor_load_cap,
                    )
            for load_edge, load_value in coordination_state.next_edge_loads.items():
                load_idx = edge_lookup.get(str(load_edge))
                if load_idx is not None:
                    node_dynamic_features[load_idx, 6] = self._normalize_coordination_value(
                        load_value,
                        self.coordination_next_edge_load_cap,
                    )

        scalar_features = np.zeros(self.graph_scalar_feature_count, dtype=np.float32)
        scalar_features[0] = 1.0 if context.commit_window else 0.0
        scalar_features[1] = context.lane_index / max(context.lane_count - 1, 1)
        scalar_features[2] = min(context.lane_count, 6) / 6.0
        scalar_features[3] = min(max(context.dist_to_end, 0.0), 200.0) / 200.0
        scalar_features[4] = min(max(float(context.speed), 0.0) / 20.0, 1.0)
        if vehicle_wait_time_fn is not None:
            scalar_features[5] = self._normalize_wait_time(vehicle_wait_time_fn(context.vehicle_id))
        if step is not None and vehicle_start_time is not None:
            elapsed = max(float(step) - float(vehicle_start_time), 0.0)
            scalar_features[6] = min(elapsed / float(self.max_simulation_steps), 1.0)

        remaining_eta = float(eta_fn(edge_id, destination_edge))
        scalar_features[7] = (
            min(remaining_eta / float(self.max_simulation_steps), 1.0)
            if math.isfinite(remaining_eta) else 1.0
        )
        scalar_features[8] = float(np.clip(density_vector[current_idx], 0.0, 1.0))
        scalar_features[9] = float(np.clip(mean_global, 0.0, 1.0))
        scalar_features[10] = float(np.clip(std_global, 0.0, 1.0))
        scalar_features[11] = float(np.clip(p95_global, 0.0, 1.0))

        coordination_global_features, coordination_action_features = self.coordination_features(
            context=context,
            destination=destination_edge,
            coordination_state=coordination_state,
        )
        scalar_features[12:16] = coordination_global_features
        coordination_action_features = coordination_action_features.reshape(
            self.action_count,
            self.coordination_action_feature_count,
        )

        action_features = np.zeros(
            (self.action_count, self.graph_action_feature_count),
            dtype=np.float32,
        )
        action_node_indices = np.full(self.action_count, -1, dtype=np.int64)
        edge_valid = set(context.edge_valid_actions)
        lane_now = set(context.lane_feasible_now_actions)
        reachable = set(context.reachable_with_lane_change_actions)
        available = set(context.available_actions)
        recent_history = list(recent_history or [])

        for action_idx in range(self.action_count):
            next_edge = self.decision_engine.get_next_edge(edge_id, action_idx)
            next_idx = edge_lookup.get(str(next_edge)) if next_edge is not None else None
            if next_idx is not None:
                action_node_indices[action_idx] = int(next_idx)

            action_features[action_idx, 0] = 1.0 if action_idx in edge_valid else 0.0
            action_features[action_idx, 1] = 1.0 if action_idx in lane_now else 0.0
            action_features[action_idx, 2] = 1.0 if action_idx in reachable else 0.0
            action_features[action_idx, 3] = 1.0 if action_idx in available else 0.0
            action_features[action_idx, 4] = min(
                float(max(context.required_lane_shift.get(action_idx, 0), 0)) / 3.0,
                1.0,
            )

            stats = self.action_corridor_stats(
                edge_id=edge_id,
                action_idx=action_idx,
                destination=destination_edge,
                edge_density_fn=edge_density_fn,
                distance_fn=distance_fn,
                eta_fn=eta_fn,
                recent_history=recent_history,
            )
            if stats is None:
                continue

            action_features[action_idx, 5] = float(np.clip(stats.mean_density, 0.0, 1.0))
            action_features[action_idx, 6] = float(np.clip(stats.max_density, 0.0, 1.0))
            action_features[action_idx, 7] = (
                min(float(stats.eta) / float(self.max_simulation_steps), 1.0)
                if math.isfinite(stats.eta) else 1.0
            )
            social_cost = float(social_cost_fn(edge_id, action_idx, destination_edge))
            action_features[action_idx, 8] = (
                min(max(social_cost, 0.0), 10.0) / 10.0
                if math.isfinite(social_cost) else 1.0
            )
            action_features[action_idx, 9] = min(max(float(stats.trap_score), 0.0), 10.0) / 10.0
            action_features[action_idx, 10] = min(float(max(stats.revisit_hits, 0)), 3.0) / 3.0
            action_features[action_idx, 11:13] = coordination_action_features[action_idx]

        return RoutingGraphObservation(
            node_dynamic_features=node_dynamic_features,
            scalar_features=scalar_features,
            action_features=action_features,
            action_node_indices=action_node_indices,
            current_node_index=int(current_idx),
            destination_node_index=int(destination_idx),
        )

    def _increment_metric(self, metrics: Optional[Dict[str, float]], key: str, amount: float = 1.0) -> None:
        if metrics is None:
            return
        metrics[key] = float(metrics.get(key, 0.0)) + float(amount)

    def policy_action_candidates(
        self,
        context: DecisionContext,
        *,
        recent_history: Sequence[str],
        cooldown_active: bool,
        destination: str,
        distance_fn: Callable[[str, str], float],
        edge_density_fn: Optional[Callable[[str], float]] = None,
        metrics: Optional[Dict[str, float]] = None,
        distance_slack: Optional[float] = None,
        coordination_state: Optional[CoordinationReservationState] = None,
    ) -> List[int]:
        available_actions = list(context.available_actions)
        if not available_actions:
            return []

        lane_now = set(context.lane_feasible_now_actions)
        recent_history = list(recent_history or [])

        commit_distance = max(
            float(self.decision_engine.commit_min_distance),
            float(context.speed) * float(self.decision_engine.commit_time_s),
        )
        extra_buffer = max(
            float(self.decision_engine.proactive_extra_buffer_m),
            0.35 * float(self.decision_engine.lane_change_margin_m),
        )
        comfortable_dist_threshold = (
            commit_distance
            + extra_buffer
            + float(self.decision_engine.proactive_safety_margin_m)
        )

        safe_lane_now_actions = []
        proactive_actions = []
        proactive_risk_scored = []
        filtered_available_actions = []

        for action in available_actions:
            safe_ok, _ = self.decision_engine.prefilter_action_for_loops(
                context=context,
                action_idx=action,
                destination=destination,
                recent_history=recent_history,
                distance_fn=distance_fn,
                distance_slack=distance_slack,
            )
            if not safe_ok:
                continue
            filtered_available_actions.append(action)
            if action in lane_now:
                safe_lane_now_actions.append(action)
                continue

            if cooldown_active or float(context.speed) < 0.5:
                continue

            required_shift = int(context.required_lane_shift.get(action, 99))
            if context.commit_window:
                self._increment_metric(metrics, 'commit_window_non_lane_candidates_seen')
                self._increment_metric(metrics, 'commit_window_candidates_rejected')
                continue

            if required_shift == 2:
                self._increment_metric(metrics, 'proactive_shift2_candidates_seen')
            if required_shift not in (1, 2):
                if required_shift == 2:
                    self._increment_metric(metrics, 'proactive_shift2_candidates_rejected')
                continue

            dist_threshold = comfortable_dist_threshold
            if required_shift == 2:
                dist_threshold = comfortable_dist_threshold + (0.9 * float(self.decision_engine.lane_change_margin_m))
            if float(context.dist_to_end) <= dist_threshold:
                if required_shift == 2:
                    self._increment_metric(metrics, 'proactive_shift2_candidates_rejected')
                continue

            brake_risk_score = self._proactive_brake_risk_score(
                context=context,
                action_idx=action,
                comfortable_dist_threshold=comfortable_dist_threshold,
                edge_density_fn=edge_density_fn,
            )
            if brake_risk_score > 0.0:
                self._increment_metric(metrics, 'proactive_brake_risk_candidates_seen')
                self._increment_metric(metrics, 'proactive_brake_risk_candidates_rejected')
                proactive_risk_scored.append((float(brake_risk_score), action))
                continue

            proactive_actions.append(action)

        safe_lane_now_actions = self._filter_lane_now_congestion_traps(
            context=context,
            lane_now_actions=safe_lane_now_actions,
            destination=destination,
            distance_fn=distance_fn,
            edge_density_fn=edge_density_fn,
            metrics=metrics,
            distance_slack=distance_slack,
            recent_history=recent_history,
        )
        policy_actions = sorted(set(safe_lane_now_actions) | set(proactive_actions))
        if not policy_actions and proactive_risk_scored:
            proactive_risk_scored.sort(key=lambda item: (item[0], item[1]))
            policy_actions = [int(proactive_risk_scored[0][1])]
            self._increment_metric(metrics, 'proactive_brake_risk_fallback_kept')
        if not policy_actions:
            policy_actions = sorted(set(filtered_available_actions))
        if not policy_actions:
            return available_actions

        policy_actions = self._filter_policy_detour_actions(
            context=context,
            actions=policy_actions,
            destination=destination,
            distance_fn=distance_fn,
            edge_density_fn=edge_density_fn,
            recent_history=recent_history,
        )
        policy_actions = self._filter_coordination_pressure_actions(
            context=context,
            actions=policy_actions,
            destination=destination,
            distance_fn=distance_fn,
            edge_density_fn=edge_density_fn,
            recent_history=recent_history,
            coordination_state=coordination_state,
            metrics=metrics,
        )

        broader_available_set = set(filtered_available_actions)
        lane_now_set = set(safe_lane_now_actions)
        policy_set = set(policy_actions)
        if len(broader_available_set) > len(lane_now_set):
            self._increment_metric(metrics, 'policy_candidates_with_broader_available')
            if policy_set == lane_now_set and len(policy_set) < len(broader_available_set):
                self._increment_metric(metrics, 'policy_candidates_collapsed_to_lane_now_only')
        return policy_actions

    def _filter_coordination_pressure_actions(
        self,
        *,
        context: DecisionContext,
        actions: Sequence[int],
        destination: str,
        distance_fn: Callable[[str, str], float],
        edge_density_fn: Optional[Callable[[str], float]],
        recent_history: Optional[Sequence[str]],
        coordination_state: Optional[CoordinationReservationState],
        metrics: Optional[Dict[str, float]],
    ) -> List[int]:
        unique_actions = sorted(set(int(action) for action in actions))
        if coordination_state is None or len(unique_actions) <= 1:
            return unique_actions

        density_lookup = edge_density_fn if edge_density_fn is not None else (lambda edge_id: 0.0)
        current_distance = float(distance_fn(context.edge_id, destination))
        distance_slack = self._distance_detour_slack(
            current_distance,
            base=float(self.coordination_filter_distance_base_slack_m),
            cap=float(self.coordination_filter_distance_cap_m),
            fraction=0.08,
        )

        scored = []
        for action in unique_actions:
            stats = self.action_corridor_stats(
                edge_id=context.edge_id,
                action_idx=action,
                destination=destination,
                edge_density_fn=density_lookup,
                distance_fn=distance_fn,
                recent_history=recent_history,
            )
            if stats is None or stats.next_edge is None:
                continue
            pressure = self.coordination_pressure_score(
                context=context,
                destination=destination,
                action_idx=action,
                coordination_state=coordination_state,
            )
            scored.append((int(action), stats, float(pressure)))

        if len(scored) <= 1:
            return unique_actions

        min_pressure = min(pressure for _, _, pressure in scored)
        best_distance = min(
            float(stats.next_distance)
            for _, stats, _ in scored
            if math.isfinite(float(stats.next_distance))
        ) if any(math.isfinite(float(stats.next_distance)) for _, stats, _ in scored) else float("inf")
        kept = []
        for action, stats, pressure in scored:
            action_distance = float(stats.next_distance)
            high_pressure = pressure >= (
                min_pressure + float(self.coordination_filter_pressure_margin)
            )
            if not high_pressure:
                kept.append(action)
                continue

            lower_pressure_alternative = False
            for alt_action, alt_stats, alt_pressure in scored:
                if alt_action == action:
                    continue
                if alt_pressure > pressure - float(self.coordination_filter_pressure_margin):
                    continue
                alt_distance = float(alt_stats.next_distance)
                if not math.isfinite(action_distance) or not math.isfinite(alt_distance):
                    lower_pressure_alternative = True
                    break
                if alt_distance <= action_distance + distance_slack:
                    lower_pressure_alternative = True
                    break

            if not lower_pressure_alternative:
                kept.append(action)
                continue
            self._increment_metric(metrics, "coordination_pressure_candidates_seen")
            if (
                math.isfinite(action_distance)
                and math.isfinite(best_distance)
                and action_distance <= best_distance - distance_slack
            ):
                kept.append(action)
                continue

            self._increment_metric(metrics, "coordination_pressure_candidates_rejected")

        if kept:
            return sorted(set(kept))
        return [min(scored, key=lambda item: (item[2], float(item[1].next_distance), item[0]))[0]]

    def rank_policy_actions(
        self,
        *,
        context: DecisionContext,
        actions: Sequence[int],
        destination: str,
        distance_fn: Callable[[str, str], float],
        edge_density_fn: Optional[Callable[[str], float]],
        recent_history: Optional[Sequence[str]] = None,
        coordination_state: Optional[CoordinationReservationState] = None,
    ) -> List[int]:
        """
        Return policy candidates in best-first heuristic order.

        The network still decides greedily from Q-values, but epsilon exploration
        uses this order to stay close to Dijkstra in light traffic and to prefer
        lower-pressure alternatives when congestion makes the shortest branch bad.
        """
        unique_actions = sorted(set(int(action) for action in actions))
        if len(unique_actions) <= 1:
            return unique_actions

        density_lookup = edge_density_fn if edge_density_fn is not None else (lambda edge_id: 0.0)
        current_distance = distance_fn(context.edge_id, destination)
        current_density = max(float(density_lookup(context.edge_id)), 0.0)
        lane_now = set(context.lane_feasible_now_actions)
        action_stats = []
        for action in unique_actions:
            stats = self.action_corridor_stats(
                edge_id=context.edge_id,
                action_idx=action,
                destination=destination,
                edge_density_fn=density_lookup,
                distance_fn=distance_fn,
                recent_history=recent_history,
            )
            if stats is None or stats.next_edge is None:
                continue
            action_stats.append((action, stats))

        if not action_stats:
            return unique_actions

        finite_next_distances = [
            float(stats.next_distance)
            for _, stats in action_stats
            if math.isfinite(stats.next_distance)
        ]
        best_next_distance = (
            min(finite_next_distances)
            if finite_next_distances else float("inf")
        )

        scored = []
        for action, stats in action_stats:
            required_shift = int(context.required_lane_shift.get(action, 0))

            score = float(stats.score)
            score += 0.28 * max(stats.max_density - stats.mean_density, 0.0)
            score += 0.08 * float(max(required_shift, 0))
            if action not in lane_now:
                score += 0.18 * float(max(required_shift, 1))
            if math.isfinite(best_next_distance) and math.isfinite(stats.next_distance):
                score += min(max(stats.next_distance - best_next_distance, 0.0) * 0.006, 0.45)
            if math.isfinite(current_distance) and math.isfinite(stats.next_distance):
                if stats.next_distance >= (current_distance - 1.0):
                    score += 0.45
                loop_distance_slack = float(getattr(self.decision_engine, "loop_distance_slack", 30.0))
                if stats.best_distance <= (current_distance - max(8.0, 0.25 * loop_distance_slack)):
                    score -= 0.10
            if stats.mean_density <= max(current_density - 0.10, 0.0):
                score -= 0.18
            if stats.max_density <= max(current_density - 0.14, 0.0):
                score -= 0.10
            if recent_history and stats.next_edge in set(recent_history[-4:]):
                score += 0.80
            if action in lane_now:
                score -= 0.05
            score += self.coordination_rank_penalty_scale * self.coordination_pressure_score(
                context=context,
                destination=destination,
                action_idx=action,
                coordination_state=coordination_state,
            )
            scored.append((float(score), action))

        if not scored:
            return unique_actions
        scored.sort(key=lambda item: (item[0], item[1]))
        return [action for _, action in scored]

    def _distance_detour_slack(self, current_distance: float, *, base: float, cap: float, fraction: float) -> float:
        if math.isfinite(current_distance):
            return float(max(base, min(cap, fraction * max(float(current_distance), 0.0))))
        return float(base)

    def _filter_policy_detour_actions(
        self,
        *,
        context: DecisionContext,
        actions: Sequence[int],
        destination: str,
        distance_fn: Callable[[str, str], float],
        edge_density_fn: Optional[Callable[[str], float]],
        recent_history: Optional[Sequence[str]] = None,
    ) -> List[int]:
        """
        Keep the model's action mask aligned with the travel-time objective.

        The network can still choose among viable branches, but materially
        longer congestion-relief detours must be route-safe and clearly useful.
        This is a mask rather than a ranking tweak because greedy inference
        ignores candidate order.
        """
        unique_actions = sorted(set(int(action) for action in actions))
        if len(unique_actions) <= 1:
            return unique_actions

        density_lookup = edge_density_fn if edge_density_fn is not None else (lambda edge_id: 0.0)
        action_stats = []
        for action in unique_actions:
            stats = self.action_corridor_stats(
                edge_id=context.edge_id,
                action_idx=action,
                destination=destination,
                edge_density_fn=density_lookup,
                distance_fn=distance_fn,
                recent_history=recent_history,
            )
            if stats is None or stats.next_edge is None:
                continue
            action_stats.append((int(action), stats))

        finite_stats = [
            (action, stats)
            for action, stats in action_stats
            if math.isfinite(float(stats.next_distance))
        ]
        if len(finite_stats) <= 1:
            return unique_actions

        best_action, best_stats = min(
            finite_stats,
            key=lambda item: (float(item[1].next_distance), float(item[1].score), item[0]),
        )
        best_next_distance = float(best_stats.next_distance)
        best_pressure = max(
            float(best_stats.score),
            (0.75 * float(best_stats.max_density)) + (0.45 * float(best_stats.mean_density)),
        )
        current_distance = float(distance_fn(context.edge_id, destination))
        close_slack = self._distance_detour_slack(
            current_distance,
            base=float(self.policy_detour_base_slack_m),
            cap=22.0,
            fraction=0.06,
        )
        relief_slack = self._distance_detour_slack(
            current_distance,
            base=max(float(self.policy_detour_relief_slack_m), close_slack),
            cap=34.0,
            fraction=0.11,
        )
        density_threshold, relief_threshold, _ = self._lane_now_congestion_thresholds(
            context=context,
            distance_slack=close_slack,
        )

        kept = {int(best_action)}
        for action, stats in finite_stats:
            action = int(action)
            if action == int(best_action):
                continue

            next_distance = float(stats.next_distance)
            detour = next_distance - best_next_distance
            corridor_revisit = int(stats.revisit_hits) > 0
            corridor_trap = float(stats.trap_score) >= 5.0
            route_safe = not corridor_revisit and not corridor_trap
            action_pressure = max(
                float(stats.score),
                (0.75 * float(stats.max_density)) + (0.45 * float(stats.mean_density)),
            )
            high_best_pressure = (
                float(best_stats.max_density) >= density_threshold
                or float(best_stats.mean_density) >= max(density_threshold - 0.06, 0.20)
            )
            strong_relief = (
                high_best_pressure
                and (
                    (best_pressure - action_pressure) >= float(self.policy_detour_score_relief)
                    or (float(best_stats.max_density) - float(stats.max_density)) >= max(
                        float(self.policy_detour_density_relief),
                        relief_threshold,
                    )
                    or (float(best_stats.mean_density) - float(stats.mean_density)) >= max(
                        0.08,
                        0.5 * relief_threshold,
                    )
                )
            )
            corridor_recovers = (
                math.isfinite(float(stats.best_distance))
                and math.isfinite(current_distance)
                and float(stats.best_distance) <= (
                    current_distance - max(8.0, 0.25 * float(self.decision_engine.loop_distance_slack))
                )
                and float(stats.best_distance) <= (best_next_distance + close_slack)
            )

            if detour <= close_slack and (route_safe or corridor_recovers):
                kept.add(action)
                continue
            if strong_relief and route_safe and detour <= relief_slack:
                kept.add(action)
                continue
            if corridor_recovers and detour <= relief_slack:
                kept.add(action)

        return sorted(action for action in unique_actions if int(action) in kept) or [int(best_action)]

    def _filter_lane_now_congestion_traps(
        self,
        *,
        context: DecisionContext,
        lane_now_actions: Sequence[int],
        destination: str,
        distance_fn: Callable[[str, str], float],
        edge_density_fn: Optional[Callable[[str], float]],
        metrics: Optional[Dict[str, float]],
        distance_slack: Optional[float],
        recent_history: Optional[Sequence[str]] = None,
    ) -> List[int]:
        actions = sorted(set(int(action) for action in lane_now_actions))
        if edge_density_fn is None or len(actions) <= 1:
            return actions

        scored = []
        for action in actions:
            stats = self.action_corridor_stats(
                edge_id=context.edge_id,
                action_idx=action,
                destination=destination,
                edge_density_fn=edge_density_fn,
                distance_fn=distance_fn,
                recent_history=recent_history,
            )
            if stats is None:
                continue
            scored.append((
                float(stats.score),
                float(stats.max_density),
                float(stats.mean_density),
                float(stats.next_distance),
                action,
            ))

        if len(scored) <= 1:
            return actions

        best_score, best_max_density, best_mean_density, best_distance, _ = min(
            scored,
            key=lambda item: (item[0], item[3], item[4]),
        )
        density_threshold, relief_threshold, distance_keep_slack = self._lane_now_congestion_thresholds(
            context=context,
            distance_slack=distance_slack,
        )
        kept = []
        for score, max_density, mean_density, distance, action in scored:
            high_pressure = (
                max_density >= density_threshold
                or mean_density >= max(density_threshold - 0.06, 0.20)
            )
            relief_available = (
                (score - best_score) >= 0.22
                or (max_density - best_max_density) >= relief_threshold
                or (mean_density - best_mean_density) >= max(0.08, 0.5 * relief_threshold)
            )
            meaningfully_shorter = math.isfinite(distance) and math.isfinite(best_distance) and (
                distance <= best_distance - distance_keep_slack
            )
            if high_pressure and relief_available and not meaningfully_shorter:
                self._increment_metric(metrics, "lane_now_congestion_candidates_seen")
                self._increment_metric(metrics, "lane_now_congestion_candidates_rejected")
                continue
            if high_pressure and relief_available:
                self._increment_metric(metrics, "lane_now_congestion_candidates_seen")
            kept.append(action)

        return sorted(set(kept)) or actions

    def _lane_now_congestion_thresholds(
        self,
        *,
        context: DecisionContext,
        distance_slack: Optional[float],
    ) -> Tuple[float, float, float]:
        base_distance_keep_slack = max(15.0, 0.5 * float(distance_slack if distance_slack is not None else 30.0))
        commit_distance = max(
            float(self.decision_engine.commit_min_distance),
            float(context.speed) * float(self.decision_engine.commit_time_s),
        )
        reaction_time_s = float(getattr(self.decision_engine, "reaction_time_s", self.decision_engine.commit_time_s))
        reaction_distance = max(
            float(getattr(self.decision_engine, "base_reaction_distance", commit_distance)),
            float(context.speed) * reaction_time_s,
        )
        junction_guard_distance = max(
            reaction_distance + (0.5 * float(self.decision_engine.lane_change_margin_m)),
            commit_distance
            + float(self.decision_engine.lane_change_margin_m)
            + float(self.decision_engine.proactive_extra_buffer_m)
            + float(self.decision_engine.proactive_safety_margin_m),
            base_distance_keep_slack,
        )
        junction_proximity = 0.0
        if junction_guard_distance > 1e-6:
            junction_proximity = max(
                0.0,
                min((junction_guard_distance - float(context.dist_to_end)) / junction_guard_distance, 1.0),
            )
        if context.commit_window:
            junction_proximity = max(junction_proximity, 0.75)

        density_threshold = max(
            0.22,
            self.lane_now_congestion_density_threshold
            - (self.lane_now_near_junction_density_tighten * junction_proximity),
        )
        relief_threshold = max(
            0.06,
            self.lane_now_congestion_relief_threshold
            - (self.lane_now_near_junction_relief_tighten * junction_proximity),
        )
        distance_keep_slack = (
            base_distance_keep_slack
            + (self.lane_now_near_junction_distance_keep_extra * junction_proximity)
        )
        return float(density_threshold), float(relief_threshold), float(distance_keep_slack)


    def _proactive_brake_risk_score(
        self,
        *,
        context: DecisionContext,
        action_idx: int,
        comfortable_dist_threshold: float,
        edge_density_fn: Optional[Callable[[str], float]],
    ) -> float:
        if edge_density_fn is None:
            return 0.0

        required_shift = int(context.required_lane_shift.get(action_idx, 99))
        if required_shift not in (1, 2):
            return 0.0

        speed = float(context.speed)
        if speed < self.proactive_brake_risk_speed_floor:
            return 0.0

        next_edge = self.decision_engine.get_next_edge(context.edge_id, action_idx)
        if next_edge is None:
            return 0.0

        current_density = max(float(edge_density_fn(context.edge_id)), 0.0)
        next_density = max(float(edge_density_fn(next_edge)), 0.0)
        density_pressure = max(current_density, next_density)
        if density_pressure < self.proactive_brake_risk_density_threshold:
            return 0.0

        extra_distance = max(float(context.dist_to_end) - float(comfortable_dist_threshold), 0.0)
        extra_time_headroom = extra_distance / max(speed, 1.0)

        required_extra_time = 0.45 if required_shift == 1 else 0.90
        if density_pressure >= self.proactive_brake_risk_density_threshold:
            required_extra_time += 0.35
        if density_pressure >= self.proactive_brake_risk_high_density_threshold:
            required_extra_time += 0.25
        if next_density >= (current_density + 0.08):
            required_extra_time += 0.15

        risk_score = required_extra_time - extra_time_headroom
        if required_shift == 2 and density_pressure >= self.proactive_brake_risk_density_threshold:
            risk_score += 0.10
        return float(risk_score)

    def select_fallback_action(
        self,
        context: DecisionContext,
        *,
        blocked_action: int,
        destination: str,
        recent_history: Sequence[str],
        distance_fn: Callable[[str, str], float],
        edge_density_fn: Optional[Callable[[str], float]] = None,
        lane_now_only: bool = False,
        coordination_state: Optional[CoordinationReservationState] = None,
    ) -> Optional[int]:
        lane_now_candidates = self.decision_engine.lane_feasible_fallback_actions(
            context,
            blocked_action=blocked_action,
        )
        candidate_actions = None
        if edge_density_fn is not None and lane_now_candidates:
            filtered_lane_now = self._filter_lane_now_congestion_traps(
                context=context,
                lane_now_actions=lane_now_candidates,
                destination=destination,
                distance_fn=distance_fn,
                edge_density_fn=edge_density_fn,
                metrics=None,
                distance_slack=None,
                recent_history=recent_history,
            )
            if filtered_lane_now:
                if lane_now_only:
                    candidate_actions = filtered_lane_now
                else:
                    safe_connected = self.decision_engine.safe_connected_fallback_actions(
                        context,
                        blocked_action=blocked_action,
                    )
                    lane_now_set = set(lane_now_candidates)
                    candidate_actions = sorted(
                        set(filtered_lane_now)
                        | {action for action in safe_connected if action not in lane_now_set}
                    )

        if lane_now_only:
            ranked = self.decision_engine.ranked_fallback_actions(
                context=context,
                destination=destination,
                recent_history=list(recent_history or []),
                blocked_action=blocked_action,
                distance_fn=distance_fn,
                candidate_actions=candidate_actions if candidate_actions is not None else lane_now_candidates,
            )
        else:
            ranked = self.decision_engine.ranked_fallback_actions(
                context=context,
                destination=destination,
                recent_history=list(recent_history or []),
                blocked_action=blocked_action,
                distance_fn=distance_fn,
                candidate_actions=candidate_actions,
            )
        if ranked and edge_density_fn is not None:
            ranked = self.rank_policy_actions(
                context=context,
                actions=ranked,
                destination=destination,
                distance_fn=distance_fn,
                edge_density_fn=edge_density_fn,
                recent_history=recent_history,
                coordination_state=coordination_state,
            )
        if ranked:
            return int(ranked[0])
        return None

    def build_observe_pending(
        self,
        *,
        state,
        action_idx: int,
        intended_next_edge: str,
        decision_edge: str,
        step: int,
        destination: str,
        context: DecisionContext,
        lane_change_requested: bool,
        decision_id: str,
        origin_mode: str,
        action_source: str,
        observe_metadata: Dict[str, object],
        decision_open_recorded: bool,
        extra_metadata: Optional[Dict[str, object]] = None,
    ) -> PendingDecision:
        metadata = {
            "phase": "observe_lane_change",
            "action_source": action_source,
            "decision_id": decision_id,
            "decision_origin_mode": origin_mode,
            "decision_resolution_mode": "",
            "decision_current_phase": "observe_lane_change",
            "decision_open_recorded": bool(decision_open_recorded),
            "decision_finalized": False,
        }
        metadata.update(dict(observe_metadata or {}))
        metadata.update(dict(extra_metadata or {}))
        return PendingDecision(
            state=state,
            intended_action=int(action_idx),
            intended_next_edge=intended_next_edge,
            decision_edge=decision_edge,
            decision_step=int(step),
            last_credit_edge=decision_edge,
            last_credit_step=int(step),
            destination=destination,
            context=context,
            lane_change_requested=bool(lane_change_requested),
            decision_id=decision_id,
            decision_origin_mode=origin_mode,
            decision_current_phase="observe_lane_change",
            decision_open_recorded=bool(decision_open_recorded),
            route_fragment=[],
            metadata=metadata,
        )

    def build_route_pending(
        self,
        *,
        state,
        action_idx: int,
        committed_next_edge: str,
        decision_edge: str,
        step: int,
        destination: str,
        context: DecisionContext,
        lane_change_requested: bool,
        decision_id: str,
        origin_mode: str,
        action_source: str,
        full_route: Optional[Sequence[str]],
        decision_open_recorded: bool,
        extra_metadata: Optional[Dict[str, object]] = None,
    ) -> PendingDecision:
        resolution_mode = "lane_now" if action_idx in context.lane_feasible_now_actions else "proactive"
        metadata = dict(extra_metadata or {})
        metadata.update({
            "phase": "route_pending",
            "action_source": action_source,
            "decision_id": decision_id,
            "decision_origin_mode": origin_mode,
            "decision_resolution_mode": resolution_mode,
            "decision_current_phase": "route_pending",
            "decision_open_recorded": bool(decision_open_recorded),
            "best_dist_to_end": float(context.dist_to_end),
            "last_progress_step": int(step),
            "decision_open": True,
            "available_count": int(len(context.available_actions)),
            "lane_now_count": int(len(context.lane_feasible_now_actions)),
            "forced_action": bool(context.forced_action is not None),
            "decision_finalized": False,
        })
        return PendingDecision(
            state=state,
            intended_action=int(action_idx),
            intended_next_edge=committed_next_edge,
            decision_edge=decision_edge,
            decision_step=int(step),
            last_credit_edge=decision_edge,
            last_credit_step=int(step),
            destination=destination,
            context=context,
            lane_change_requested=bool(lane_change_requested),
            decision_id=decision_id,
            decision_origin_mode=origin_mode,
            decision_current_phase="route_pending",
            decision_open_recorded=bool(decision_open_recorded),
            route_fragment=list(full_route[1:]) if full_route else [],
            metadata=metadata,
        )

    def pending_phase(self, pending: PendingDecision) -> str:
        metadata = pending.metadata if isinstance(pending.metadata, dict) else {}
        phase = metadata.get("phase", pending.decision_current_phase or "route_pending")
        return str(phase or "route_pending")

    def pending_resolution_mode(self, pending: PendingDecision) -> str:
        metadata = pending.metadata if isinstance(pending.metadata, dict) else {}
        resolution_mode = metadata.get("decision_resolution_mode", pending.decision_origin_mode or "lane_now")
        return str(resolution_mode or "lane_now")

    def pending_requires_active_same_edge_monitoring(self, pending: PendingDecision) -> bool:
        phase = self.pending_phase(pending)
        if phase == "observe_lane_change":
            return True
        if phase != "route_pending":
            return True
        return self.pending_resolution_mode(pending) != "lane_now"

    def promote_observe_success(
        self,
        pending: PendingDecision,
        *,
        context: DecisionContext,
        step: int,
        committed_next_edge: str,
        full_route: Optional[Sequence[str]],
        state=None,
        action_source: Optional[str] = None,
        extra_metadata: Optional[Dict[str, object]] = None,
    ) -> PendingDecision:
        prior_metadata = dict(pending.metadata or {})
        if extra_metadata:
            prior_metadata.update(dict(extra_metadata))
        return self.build_route_pending(
            state=pending.state if state is None else state,
            action_idx=pending.intended_action,
            committed_next_edge=committed_next_edge,
            decision_edge=context.edge_id,
            step=step,
            destination=pending.destination,
            context=context,
            lane_change_requested=True,
            decision_id=str(prior_metadata.get("decision_id", pending.decision_id)),
            origin_mode=str(prior_metadata.get("decision_origin_mode", pending.decision_origin_mode or "proactive")),
            action_source=action_source or str(prior_metadata.get("action_source", "policy")),
            full_route=full_route,
            decision_open_recorded=True,
            extra_metadata=prior_metadata,
        )

    def evaluate_route_pending_release(
        self,
        pending: PendingDecision,
        *,
        context: DecisionContext,
        step: int,
        lane_position_now: float,
        edge_density_fn: Optional[Callable[[str], float]] = None,
        distance_fn: Optional[Callable[[str, str], float]] = None,
        recent_history: Optional[Sequence[str]] = None,
    ) -> PendingReleaseEvaluation:
        metadata = pending.metadata if isinstance(pending.metadata, dict) else {}
        metadata["lane_position_now"] = float(max(lane_position_now, 0.0))
        pending.metadata = metadata
        metadata, progress_view = self.decision_engine.pending_progress_update(
            pending=pending,
            context=context,
            step=step,
        )
        active_same_edge_monitoring = self.pending_requires_active_same_edge_monitoring(pending)

        current_shift = int(context.required_lane_shift.get(pending.intended_action, 99))
        grace_keep = (
            current_shift <= 1
            and context.dist_to_end >= max(self.decision_engine.commit_min_distance + 2.0, 6.0)
        )
        same_edge = context.edge_id == pending.decision_edge
        wrong_lane_commit = (
            same_edge
            and context.commit_window
            and pending.intended_action not in context.lane_feasible_now_actions
            and not grace_keep
        )
        last_progress_step = int(progress_view["last_progress_step"])
        stall_age = max(int(step) - last_progress_step, 0)
        total_age = max(int(step) - int(pending.decision_step), 0)
        no_progress_window = int(self.decision_engine.route_pending_no_progress_window_steps)
        lane_now_pending = self.pending_resolution_mode(pending) == "lane_now"
        no_progress_stall = (
            active_same_edge_monitoring
            and same_edge
            and stall_age >= max(no_progress_window, 1)
            and not bool(progress_view["made_progress"])
        )
        replan_stall = (
            active_same_edge_monitoring
            and same_edge
            and stall_age >= int(self.decision_engine.route_pending_stall_steps)
        )
        stalled_timeout = (
            replan_stall
            and total_age >= int(self.decision_engine.pending_progress_timeout_steps)
        )
        hard_timeout = (
            active_same_edge_monitoring
            and same_edge
            and total_age >= int(self.decision_engine.route_pending_hard_timeout_steps)
        )
        lane_now_replan_action = self._stalled_lane_now_replan_action(
            pending,
            context=context,
            total_age=total_age,
            stall_age=stall_age,
            edge_density_fn=edge_density_fn,
            distance_fn=distance_fn,
            recent_history=recent_history,
        )
        lane_now_replan = lane_now_replan_action is not None
        lane_now_stale_timeout = (
            lane_now_pending
            and same_edge
            and total_age >= int(self.decision_engine.route_pending_hard_timeout_steps)
            and stall_age >= int(self.lane_now_stale_timeout_min_stall_steps)
            and (
                float(context.speed) <= float(self.lane_now_replan_low_speed_mps)
                or total_age >= int(self.lane_now_stale_timeout_max_age_steps)
                or current_shift >= 90
            )
        )

        release_reason = None
        release_as_timeout = False
        if hard_timeout:
            release_reason = "route_hard_timeout"
            release_as_timeout = True
        elif stalled_timeout:
            release_reason = "route_stall_timeout"
            release_as_timeout = True
        elif replan_stall:
            release_reason = "route_no_progress_abort"
        elif no_progress_stall:
            release_reason = "route_no_progress_abort"
        elif lane_now_replan:
            # Reopen only when a stalled lane-now route has a cleaner comparable branch.
            release_reason = "route_no_progress_abort"
        elif lane_now_stale_timeout:
            # Passive lane-now pendings should survive normal queues, but not
            # hundreds of no-progress steps with no viable relief branch.
            release_reason = "route_hard_timeout"
            release_as_timeout = True
        elif wrong_lane_commit:
            release_reason = "wrong_lane_commit"

        return PendingReleaseEvaluation(
            should_release=release_reason is not None,
            release_reason=release_reason,
            release_as_timeout=release_as_timeout,
            grace_keep=grace_keep,
            progress_view=progress_view,
            stall_age=stall_age,
            total_age=total_age,
            avoid_reopen_action=int(pending.intended_action) if lane_now_replan else None,
            preferred_replan_action=int(lane_now_replan_action) if lane_now_replan_action is not None else None,
        )

    def _stalled_lane_now_replan_action(
        self,
        pending: PendingDecision,
        *,
        context: DecisionContext,
        total_age: int,
        stall_age: int,
        edge_density_fn: Optional[Callable[[str], float]],
        distance_fn: Optional[Callable[[str, str], float]],
        recent_history: Optional[Sequence[str]] = None,
    ) -> Optional[int]:
        if edge_density_fn is None or distance_fn is None:
            return None
        if self.pending_resolution_mode(pending) != "lane_now":
            return None
        if context.edge_id != pending.decision_edge:
            return None
        if total_age < int(self.lane_now_replan_min_age_steps):
            return None
        if float(context.speed) > float(self.lane_now_replan_low_speed_mps):
            return None
        route_pending_stall_steps = int(getattr(self.decision_engine, "route_pending_stall_steps", 8))
        if stall_age < route_pending_stall_steps:
            return None

        current_next_edge = pending.intended_next_edge
        if not current_next_edge:
            return None
        current_stats = self.action_corridor_stats(
            edge_id=context.edge_id,
            action_idx=int(pending.intended_action),
            destination=pending.destination,
            edge_density_fn=edge_density_fn,
            distance_fn=distance_fn,
            recent_history=recent_history,
        )
        if current_stats is None:
            return None
        current_distance = (
            float(current_stats.next_distance)
            if math.isfinite(current_stats.next_distance) else float("inf")
        )
        current_pressure = max(
            float(current_stats.score),
            (0.75 * float(current_stats.max_density)) + (0.45 * float(current_stats.mean_density)),
        )

        density_threshold, relief_threshold, distance_slack = self._lane_now_congestion_thresholds(
            context=context,
            distance_slack=None,
        )
        route_slack = self._distance_detour_slack(
            current_distance,
            base=float(self.lane_now_replan_distance_slack_m),
            cap=max(float(distance_slack), float(self.lane_now_replan_distance_slack_m)),
            fraction=0.06,
        )
        deadlock_slack = self._distance_detour_slack(
            current_distance,
            base=float(self.lane_now_deadlock_distance_slack_m),
            cap=18.0,
            fraction=0.04,
        )

        alternatives = []
        for action in sorted(set(context.lane_feasible_now_actions)):
            if int(action) == int(pending.intended_action):
                continue
            stats = self.action_corridor_stats(
                edge_id=context.edge_id,
                action_idx=int(action),
                destination=pending.destination,
                edge_density_fn=edge_density_fn,
                distance_fn=distance_fn,
                recent_history=recent_history,
            )
            if stats is None or stats.next_edge is None or stats.next_edge == current_next_edge:
                continue
            alt_distance = float(stats.next_distance)
            if not math.isfinite(alt_distance):
                continue
            alt_pressure = max(
                float(stats.score),
                (0.75 * float(stats.max_density)) + (0.45 * float(stats.mean_density)),
            )
            alternatives.append((float(alt_pressure), stats, float(alt_distance), int(action)))
        if not alternatives:
            return None

        severe_stall = stall_age >= int(self.decision_engine.pending_progress_timeout_steps)
        deadlock_stall = (
            total_age >= int(self.lane_now_replan_deadlock_min_age_steps)
            and stall_age >= route_pending_stall_steps
        )

        relief_candidates = []
        shorter_candidates = []
        deadlock_candidates = []
        for alt_pressure, alt_stats, alt_distance, action in alternatives:
            route_revisit = int(alt_stats.revisit_hits) > 0
            route_trap = float(alt_stats.trap_score) >= 5.0
            route_safe = not route_revisit and not route_trap
            enough_relief = (
                (current_pressure - alt_pressure) >= 0.22
                or (current_stats.max_density - alt_stats.max_density) >= relief_threshold
                or (current_stats.mean_density - alt_stats.mean_density) >= max(0.08, 0.5 * relief_threshold)
            )
            not_much_longer = alt_distance <= (current_distance + route_slack)
            clearly_shorter = alt_distance <= (current_distance - max(8.0, 0.35 * distance_slack))
            pressure_not_worse = (
                alt_pressure <= (current_pressure + 0.15)
                and alt_stats.max_density <= (current_stats.max_density + max(0.08, 0.5 * relief_threshold))
            )
            comparable_or_better = alt_distance <= (current_distance + deadlock_slack)
            if (
                route_safe
                and max(current_stats.max_density, current_stats.mean_density) >= density_threshold
                and enough_relief
                and not_much_longer
            ):
                relief_candidates.append((alt_pressure, alt_distance, action))
            if severe_stall and clearly_shorter and (route_safe or alt_distance < current_distance):
                shorter_candidates.append((alt_distance, alt_pressure, action))
            if deadlock_stall and route_safe and comparable_or_better and pressure_not_worse:
                deadlock_candidates.append((alt_pressure, alt_distance, action))

        # Congested lane-now choices still need relief. Low-density yield/deadlock
        # cases often do not cross the congestion threshold, so let sustained
        # low-speed stalls escape to a comparable non-worse branch.
        if relief_candidates:
            return int(min(relief_candidates, key=lambda item: (item[0], item[1], item[2]))[2])
        if shorter_candidates:
            return int(min(shorter_candidates, key=lambda item: (item[0], item[1], item[2]))[2])
        if deadlock_candidates:
            return int(min(deadlock_candidates, key=lambda item: (item[0], item[1], item[2]))[2])
        return None
