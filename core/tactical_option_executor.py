from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import traci

from core.strategic_options import BranchOption, StrategicDecisionContext


TACTICAL_OUTCOMES = {
    "success",
    "commit_window_miss",
    "stalled_lane_change",
    "forced_by_lane_commit",
    "teleport",
    "collision",
    "timeout",
    "apply_route_failed",
    "became_impossible_after_selection",
}


class TacticalOptionExecutor:
    def __init__(self, decision_engine):
        self.decision_engine = decision_engine

    def build_options(self, context, shared_state: np.ndarray, queue_length_fn=None) -> StrategicDecisionContext:
        options: List[BranchOption] = []
        lane_now = set(context.lane_feasible_now_actions)
        for action_idx in context.edge_valid_actions:
            next_edge = self.decision_engine.get_next_edge(context.edge_id, action_idx)
            if not next_edge:
                continue
            required_shift = int(context.required_lane_shift.get(action_idx, 0))
            currently_executable = action_idx in context.available_actions
            queue_length = float(queue_length_fn(next_edge)) if queue_length_fn else 0.0
            dist_margin = float(context.dist_to_end - self.decision_engine.commit_min_distance)
            time_margin = dist_margin / max(float(context.speed), 0.1)
            feature_vec = np.array([
                float(required_shift),
                1.0,
                0.0,
                0.0,
                20.0,
                20.0,
                0.0,
                dist_margin,
                time_margin,
                queue_length,
                1.0 if context.commit_window else 0.0,
                queue_length + max(0.0, -dist_margin),
                queue_length + 1.0,
                0.0,
                0.0,
                1.0 if currently_executable else 0.0,
                1.0 if action_idx not in lane_now else 0.0,
                1.0 if context.commit_window else 0.0,
            ], dtype=np.float32)
            options.append(
                BranchOption(
                    option_id=int(action_idx),
                    outgoing_edge=str(next_edge),
                    movement_label=str(self.decision_engine.direction_choices[action_idx]),
                    target_lane_set=tuple(range(max(context.lane_count, 1))),
                    required_lane_shift=required_shift,
                    min_execution_horizon_steps=max(1, required_shift + 1),
                    commit_start_distance_m=float(self.decision_engine.commit_min_distance),
                    success_condition="edge_transition_matches_outgoing_edge",
                    failure_conditions=(
                        "commit_window_miss",
                        "stalled_lane_change",
                        "timeout",
                        "apply_route_failed",
                        "became_impossible_after_selection",
                    ),
                    currently_executable=bool(currently_executable),
                    tactical_risk_flags={
                        "needs_lane_change": bool(action_idx not in lane_now),
                        "commit_window": bool(context.commit_window),
                    },
                    option_features=feature_vec,
                )
            )

        return StrategicDecisionContext(
            vehicle_id=context.vehicle_id,
            step=int(context.step),
            current_edge=context.edge_id,
            lane_index=int(context.lane_index),
            lane_count=int(context.lane_count),
            dist_to_end=float(context.dist_to_end),
            speed=float(context.speed),
            shared_state=shared_state,
            options=options,
        )

    def filter_executable_options(self, context: StrategicDecisionContext) -> List[BranchOption]:
        return [opt for opt in context.options if opt.currently_executable]

    def begin_execution(self, vehicle_id: str, option: BranchOption, step: int) -> dict:
        return {
            "vehicle_id": vehicle_id,
            "option_id": option.option_id,
            "outgoing_edge": option.outgoing_edge,
            "start_step": int(step),
            "lane_change_attempts": 0,
            "lane_change_successes": 0,
            "lane_change_failures": 0,
            "commit_started": False,
            "resolved": False,
            "resolution": None,
        }

    def step_execution(self, tactical_state: dict, context, selected_option: BranchOption, step: int) -> dict:
        if tactical_state.get("resolved"):
            return tactical_state
        tactical_state["last_step"] = int(step)
        if context.edge_id != tactical_state["outgoing_edge"]:
            tactical_state["commit_started"] = True
        return tactical_state

    def resolve_execution(self, tactical_state: dict, context, selected_option: BranchOption, done: bool = False) -> str:
        if context.edge_id == selected_option.outgoing_edge:
            tactical_state["resolved"] = True
            tactical_state["resolution"] = "success"
            return "success"

        if done:
            tactical_state["resolved"] = True
            tactical_state["resolution"] = "timeout"
            return "timeout"

        if context.commit_window and not selected_option.currently_executable:
            tactical_state["resolved"] = True
            tactical_state["resolution"] = "commit_window_miss"
            return "commit_window_miss"

        return "in_progress"

    def abort_execution(self, tactical_state: dict, reason: str) -> str:
        normalized = reason if reason in TACTICAL_OUTCOMES else "became_impossible_after_selection"
        tactical_state["resolved"] = True
        tactical_state["resolution"] = normalized
        return normalized
