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
            next_edge = self.decision_engine.get_next_edge(context.edge_id, action_idx)
            if next_edge is None:
                continue
            features[base + 0] = float(context.required_lane_shift.get(action_idx, 0)) / 3.0
            features[base + 1] = 1.0 if action_idx in lane_now else 0.0
            features[base + 2] = float(edge_density_fn(next_edge))
            eta = eta_fn(next_edge, destination)
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
    ) -> List[int]:
        actions = sorted(set(int(action) for action in lane_now_actions))
        if edge_density_fn is None or len(actions) <= 1:
            return actions

        scored = []
        for action in actions:
            next_edge = self.decision_engine.get_next_edge(context.edge_id, action)
            if next_edge is None:
                continue
            density = max(float(edge_density_fn(next_edge)), 0.0)
            distance = distance_fn(next_edge, destination)
            if not math.isfinite(distance):
                distance = float("inf")
            scored.append((density, float(distance), action))

        if len(scored) <= 1:
            return actions

        best_density, best_density_distance, _ = min(scored, key=lambda item: (item[0], item[1], item[2]))
        density_threshold, relief_threshold, distance_keep_slack = self._lane_now_congestion_thresholds(
            context=context,
            distance_slack=distance_slack,
        )
        kept = []
        for density, distance, action in scored:
            high_pressure = density >= density_threshold
            relief_available = (density - best_density) >= relief_threshold
            meaningfully_shorter = math.isfinite(distance) and math.isfinite(best_density_distance) and (
                distance <= best_density_distance - distance_keep_slack
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
        lane_now_only: bool = False,
    ) -> Optional[int]:
        if lane_now_only:
            lane_now_candidates = self.decision_engine.lane_feasible_fallback_actions(
                context,
                blocked_action=blocked_action,
            )
            ranked = self.decision_engine.ranked_fallback_actions(
                context=context,
                destination=destination,
                recent_history=list(recent_history or []),
                blocked_action=blocked_action,
                distance_fn=distance_fn,
                candidate_actions=lane_now_candidates,
            )
        else:
            ranked = self.decision_engine.ranked_fallback_actions(
                context=context,
                destination=destination,
                recent_history=list(recent_history or []),
                blocked_action=blocked_action,
                distance_fn=distance_fn,
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
        no_progress_stall = (
            active_same_edge_monitoring
            and same_edge
            and stall_age >= max(no_progress_window, 1)
            and not bool(progress_view["made_progress"])
        )
        stalled_timeout = (
            active_same_edge_monitoring
            and same_edge
            and stall_age >= int(self.decision_engine.route_pending_stall_steps)
        )
        hard_timeout = (
            active_same_edge_monitoring
            and same_edge
            and total_age >= int(self.decision_engine.route_pending_hard_timeout_steps)
        )
        lane_now_replan = self._should_replan_stalled_lane_now_pending(
            pending,
            context=context,
            total_age=total_age,
            edge_density_fn=edge_density_fn,
            distance_fn=distance_fn,
        )

        release_reason = None
        release_as_timeout = False
        if hard_timeout:
            release_reason = "route_hard_timeout"
            release_as_timeout = True
        elif stalled_timeout:
            release_reason = "route_stall_timeout"
            release_as_timeout = True
        elif no_progress_stall:
            release_reason = "route_no_progress_abort"
        elif lane_now_replan:
            # Reopen only when a stalled lane-now route has a cleaner comparable branch.
            release_reason = "route_no_progress_abort"
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
        )

    def _should_replan_stalled_lane_now_pending(
        self,
        pending: PendingDecision,
        *,
        context: DecisionContext,
        total_age: int,
        edge_density_fn: Optional[Callable[[str], float]],
        distance_fn: Optional[Callable[[str, str], float]],
    ) -> bool:
        if edge_density_fn is None or distance_fn is None:
            return False
        if self.pending_resolution_mode(pending) != "lane_now":
            return False
        if context.edge_id != pending.decision_edge:
            return False
        if total_age < int(self.lane_now_replan_min_age_steps):
            return False
        if float(context.speed) > float(self.lane_now_replan_low_speed_mps):
            return False

        current_next_edge = pending.intended_next_edge
        if not current_next_edge:
            return False
        current_density = max(float(edge_density_fn(current_next_edge)), 0.0)
        current_distance = distance_fn(current_next_edge, pending.destination)
        if not math.isfinite(current_distance):
            current_distance = float("inf")

        density_threshold, relief_threshold, distance_slack = self._lane_now_congestion_thresholds(
            context=context,
            distance_slack=None,
        )
        if current_density < density_threshold:
            return False

        alternatives = []
        for action in sorted(set(context.lane_feasible_now_actions)):
            if int(action) == int(pending.intended_action):
                continue
            next_edge = self.decision_engine.get_next_edge(context.edge_id, int(action))
            if next_edge is None or next_edge == current_next_edge:
                continue
            alt_distance = distance_fn(next_edge, pending.destination)
            if not math.isfinite(alt_distance):
                continue
            alt_density = max(float(edge_density_fn(next_edge)), 0.0)
            alternatives.append((alt_density, float(alt_distance), int(action)))
        if not alternatives:
            return False

        best_alt_density, best_alt_distance, _ = min(alternatives, key=lambda item: (item[0], item[1], item[2]))
        enough_relief = (current_density - best_alt_density) >= relief_threshold
        not_much_longer = best_alt_distance <= (current_distance + distance_slack)
        return bool(enough_relief and not_much_longer)
