from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import math

import numpy as np

from core.junction_decision_engine import DecisionContext, JunctionDecisionEngine, PendingDecision


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


class SharedDecisionPolicy:
    """
    Shared owner for policy-facing decision logic that wraps the junction executor.

    The engine remains the source of truth for route continuity and raw feasibility.
    This helper owns:
    - state feature spec
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
        edge_embedding_dim: int = 8,
        local_congestion_k: int = 6,
        density_scale_m: float = 100.0,
        max_simulation_steps: int = 2000,
    ):
        self.connection_info = connection_info
        self.decision_engine = decision_engine
        self.direction_choices = list(direction_choices)
        self.action_count = len(self.direction_choices)
        self.edge_embedding_dim = int(edge_embedding_dim)
        self.local_congestion_k = int(local_congestion_k)
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
        self.lane_now_replan_min_age_steps = 10
        self.lane_now_replan_low_speed_mps = 1.25
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
        self.corridor_horizon_m = max(
            float(getattr(self.decision_engine, "default_fragment_horizon_m", 180.0)),
            120.0,
        )
        self.corridor_congestion_density_threshold = 0.32
        self.corridor_recent_revisit_window = 8

        self.compact_state_size = (
            (2 * self.edge_embedding_dim)
            + (4 * self.action_count)
            + 1
            + 3
            + 3
            + self.local_congestion_k
            + (5 * self.action_count)
        )

    def legacy_state_size(self, edge_count: int) -> int:
        return 2 + self.action_count + 3 + 3 + int(edge_count)

    def classify_decision(self, context: DecisionContext) -> DecisionMode:
        if context.forced_action is not None:
            return DecisionMode("forced", action=int(context.forced_action), skip_reason=context.skip_reason)
        if context.skip_reason:
            return DecisionMode("skip", skip_reason=context.skip_reason)
        if self.decision_engine.is_decision_open(context):
            return DecisionMode("open")
        return DecisionMode("hold")

    def global_density_stats(
        self,
        edge_density_fn: Callable[[str], float],
        edge_lane_meters_fn: Callable[[str], float],
    ) -> Tuple[float, float]:
        edge_list = list(self.connection_info.edge_list)
        if not edge_list:
            return 0.0, 0.0

        lane_meters = np.array([edge_lane_meters_fn(edge) for edge in edge_list], dtype=np.float32)
        densities = np.array([edge_density_fn(edge) for edge in edge_list], dtype=np.float32)
        total_lane_meters = float(np.sum(lane_meters))
        if total_lane_meters <= 0.0:
            return 0.0, 0.0

        mean_global = float(np.average(densities, weights=lane_meters))
        std_global = float(np.sqrt(np.average((densities - mean_global) ** 2, weights=lane_meters)))
        return mean_global, std_global

    def local_congestion_features(
        self,
        edge_id: str,
        *,
        edge_density_fn: Callable[[str], float],
        global_density_stats: Tuple[float, float],
    ) -> np.ndarray:
        current_density = float(edge_density_fn(edge_id))
        outgoing = self.connection_info.outgoing_edges_dict.get(edge_id, {})
        outgoing_densities = [float(edge_density_fn(next_edge)) for next_edge in outgoing.values()]
        mean_out = float(np.mean(outgoing_densities)) if outgoing_densities else current_density
        max_out = float(np.max(outgoing_densities)) if outgoing_densities else current_density
        min_out = float(np.min(outgoing_densities)) if outgoing_densities else current_density
        mean_global, std_global = global_density_stats
        return np.array(
            [
                current_density,
                mean_out,
                max_out,
                min_out,
                current_density - float(mean_global),
                float(std_global),
            ],
            dtype=np.float32,
        )

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

    def per_action_branch_features(
        self,
        context: DecisionContext,
        destination: str,
        *,
        edge_density_fn: Callable[[str], float],
        eta_fn: Callable[[str, str], float],
        social_cost_fn: Callable[[str, int, str], float],
    ) -> np.ndarray:
        features = np.zeros(5 * self.action_count, dtype=np.float32)
        lane_now = set(context.lane_feasible_now_actions)
        for action_idx in range(self.action_count):
            base = action_idx * 5
            if action_idx not in context.edge_valid_actions:
                continue
            stats = self.action_corridor_stats(
                edge_id=context.edge_id,
                action_idx=action_idx,
                destination=destination,
                edge_density_fn=edge_density_fn,
                eta_fn=eta_fn,
            )
            if stats is None or stats.next_edge is None:
                continue
            features[base + 0] = float(context.required_lane_shift.get(action_idx, 0)) / 3.0
            features[base + 1] = 1.0 if action_idx in lane_now else 0.0
            features[base + 2] = float(stats.mean_density)
            eta = float(stats.eta)
            features[base + 3] = (
                min(float(eta) / float(self.max_simulation_steps), 1.0)
                if math.isfinite(eta) else 1.0
            )
            social = social_cost_fn(context.edge_id, action_idx, destination)
            features[base + 4] = min(float(social), 10.0) if math.isfinite(social) else 10.0
        return features

    def encode_state(
        self,
        *,
        edge_id: str,
        destination_edge: str,
        context: DecisionContext,
        use_compact_state: bool,
        edge_embedding_fn: Callable[[str], np.ndarray],
        edge_density_fn: Callable[[str], float],
        eta_fn: Callable[[str, str], float],
        social_cost_fn: Callable[[str, int, str], float],
        global_density_stats: Optional[Tuple[float, float]] = None,
        edge_lane_meters_fn: Optional[Callable[[str], float]] = None,
        step: Optional[int] = None,
        vehicle_start_time: Optional[float] = None,
        edge_index_lookup: Optional[Dict[str, int]] = None,
        legacy_aux_features: Optional[Sequence[float]] = None,
        legacy_density_values: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        if use_compact_state:
            state = np.zeros(self.compact_state_size, dtype=np.float32)
            state[0:self.edge_embedding_dim] = edge_embedding_fn(edge_id)
            state[self.edge_embedding_dim:(2 * self.edge_embedding_dim)] = edge_embedding_fn(destination_edge)

            edge_mask, lane_mask, reach_mask, avail_mask = self.decision_engine.direction_masks(context)
            base = 2 * self.edge_embedding_dim
            state[base:base + self.action_count] = np.array(edge_mask, dtype=np.float32)
            state[base + self.action_count:base + (2 * self.action_count)] = np.array(lane_mask, dtype=np.float32)
            state[base + (2 * self.action_count):base + (3 * self.action_count)] = np.array(reach_mask, dtype=np.float32)
            state[base + (3 * self.action_count):base + (4 * self.action_count)] = np.array(avail_mask, dtype=np.float32)
            state[base + (4 * self.action_count)] = 1.0 if context.commit_window else 0.0

            lane_base = base + (4 * self.action_count) + 1
            state[lane_base + 0] = context.lane_index / max(context.lane_count - 1, 1)
            state[lane_base + 1] = min(context.lane_count, 6) / 6.0
            state[lane_base + 2] = min(max(context.dist_to_end, 0.0), 200.0) / 200.0

            objective_base = lane_base + 3
            current_density = float(edge_density_fn(edge_id))
            if step is not None and vehicle_start_time is not None:
                elapsed = max(float(step) - float(vehicle_start_time), 0.0)
                remaining_eta = eta_fn(edge_id, destination_edge)
                state[objective_base + 0] = min(elapsed / float(self.max_simulation_steps), 1.0)
                state[objective_base + 1] = (
                    min(float(remaining_eta) / float(self.max_simulation_steps), 1.0)
                    if math.isfinite(remaining_eta) else 1.0
                )
                state[objective_base + 2] = min(current_density, 1.0)

            if global_density_stats is None:
                if edge_lane_meters_fn is None:
                    raise ValueError("edge_lane_meters_fn is required when global_density_stats is omitted")
                global_density_stats = self.global_density_stats(edge_density_fn, edge_lane_meters_fn)

            congestion_base = objective_base + 3
            congestion_features = self.local_congestion_features(
                edge_id,
                edge_density_fn=edge_density_fn,
                global_density_stats=global_density_stats,
            )
            state[congestion_base:congestion_base + self.local_congestion_k] = congestion_features
            state[congestion_base + self.local_congestion_k:] = self.per_action_branch_features(
                context,
                destination_edge,
                edge_density_fn=edge_density_fn,
                eta_fn=eta_fn,
                social_cost_fn=social_cost_fn,
            )
            return state.reshape(1, -1)

        if edge_index_lookup is None:
            raise ValueError("edge_index_lookup is required for legacy state encoding")

        state_values = [
            float(edge_index_lookup[edge_id]),
            float(edge_index_lookup[destination_edge]),
        ]
        outgoing = self.connection_info.outgoing_edges_dict.get(edge_id, {})
        state_values.extend(
            1.0 if direction in outgoing.keys() else 0.0
            for direction in self.direction_choices
        )
        state_values.extend(
            [
                context.lane_index / max(context.lane_count - 1, 1),
                min(context.lane_count, 6) / 6.0,
                min(max(context.dist_to_end, 0.0), 200.0) / 200.0,
            ]
        )
        state_values.extend(float(value) for value in (legacy_aux_features or (0.0, 0.0, 0.0)))
        if legacy_density_values is None:
            legacy_density_values = [edge_density_fn(edge) for edge in self.connection_info.edge_list]
        state_values.extend(float(value) for value in legacy_density_values)
        return np.asarray(state_values, dtype=np.float32).reshape(1, -1)

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

        broader_available_set = set(filtered_available_actions)
        lane_now_set = set(safe_lane_now_actions)
        policy_set = set(policy_actions)
        if len(broader_available_set) > len(lane_now_set):
            self._increment_metric(metrics, 'policy_candidates_with_broader_available')
            if policy_set == lane_now_set and len(policy_set) < len(broader_available_set):
                self._increment_metric(metrics, 'policy_candidates_collapsed_to_lane_now_only')
        return policy_actions

    def rank_policy_actions(
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
            scored.append((float(score), action))

        if not scored:
            return unique_actions
        scored.sort(key=lambda item: (item[0], item[1]))
        return [action for _, action in scored]

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
            enough_relief = (
                (current_pressure - alt_pressure) >= 0.22
                or (current_stats.max_density - alt_stats.max_density) >= relief_threshold
                or (current_stats.mean_density - alt_stats.mean_density) >= max(0.08, 0.5 * relief_threshold)
            )
            not_much_longer = alt_distance <= (current_distance + distance_slack)
            clearly_shorter = alt_distance <= (current_distance - max(8.0, 0.35 * distance_slack))
            pressure_not_worse = (
                alt_pressure <= (current_pressure + 0.15)
                and alt_stats.max_density <= (current_stats.max_density + max(0.08, 0.5 * relief_threshold))
            )
            comparable_or_better = alt_distance <= (current_distance + max(distance_slack, 20.0))
            if (
                max(current_stats.max_density, current_stats.mean_density) >= density_threshold
                and enough_relief
                and not_much_longer
            ):
                relief_candidates.append((alt_pressure, alt_distance, action))
            if severe_stall and clearly_shorter:
                shorter_candidates.append((alt_distance, alt_pressure, action))
            if deadlock_stall and comparable_or_better and pressure_not_worse:
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
