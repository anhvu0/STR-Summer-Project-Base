import unittest
from collections import defaultdict
from pathlib import Path
import sys
import types

sys.modules.setdefault('traci', types.SimpleNamespace(TraCIException=Exception))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.junction_decision_engine import DecisionContext
from core.shared_decision_policy import SharedDecisionPolicy


class DummyConnectionInfo:
    edge_list = []


class DummyDecisionEngine:
    commit_min_distance = 8.0
    commit_time_s = 0.45
    proactive_extra_buffer_m = 6.0
    lane_change_margin_m = 10.0
    proactive_safety_margin_m = 8.0

    def __init__(self):
        self.next_edges = {
            ('edgeA', 0): 'edgeB',
            ('edgeA', 1): 'edgeC',
            ('edgeA', 2): 'edgeD',
        }

    def prefilter_action_for_loops(self, context, action_idx, destination, recent_history, distance_fn=None, distance_slack=None):
        return True, {}

    def get_next_edge(self, edge_id, action_idx):
        return self.next_edges.get((edge_id, action_idx))

    def is_decision_open(self, context):
        return len(context.available_actions) > 1


class SharedDecisionPolicyBrakeRiskTests(unittest.TestCase):
    def setUp(self):
        self.engine = DummyDecisionEngine()
        self.policy = SharedDecisionPolicy(DummyConnectionInfo(), self.engine, ['L', 'S', 'R'])

    def _context(self, *, lane_now=None, available=None, shifts=None, dist_to_end=28.0, speed=12.0):
        return DecisionContext(
            vehicle_id='veh0',
            edge_id='edgeA',
            destination='destX',
            step=10,
            speed=speed,
            lane_id='edgeA_0',
            lane_index=0,
            lane_count=2,
            dist_to_end=dist_to_end,
            edge_valid_actions=[0, 1, 2],
            lane_feasible_now_actions=list(lane_now or []),
            reachable_with_lane_change_actions=[0, 1, 2],
            available_actions=list(available or [0, 1, 2]),
            required_lane_shift=dict(shifts or {0: 0, 1: 1, 2: 2}),
            commit_window=False,
            forced_action=None,
            branch_with_choice=True,
            skip_reason=None,
        )

    def test_prefers_lane_now_when_dense_proactive_is_brake_risky(self):
        context = self._context(lane_now=[0], available=[0, 1], shifts={0: 0, 1: 1}, dist_to_end=29.0, speed=14.0)
        metrics = defaultdict(float)
        density = {'edgeA': 0.34, 'edgeC': 0.42}

        actions = self.policy.policy_action_candidates(
            context,
            recent_history=['edgeZ'],
            cooldown_active=False,
            destination='destX',
            distance_fn=lambda edge, dest: 10.0,
            edge_density_fn=lambda edge: density.get(edge, 0.0),
            metrics=metrics,
        )

        self.assertEqual(actions, [0])
        self.assertEqual(metrics['proactive_brake_risk_candidates_seen'], 1.0)
        self.assertEqual(metrics['proactive_brake_risk_candidates_rejected'], 1.0)
        self.assertEqual(metrics['policy_candidates_collapsed_to_lane_now_only'], 1.0)

    def test_keeps_least_risky_proactive_when_no_lane_now_option_exists(self):
        context = self._context(lane_now=[], available=[1, 2], shifts={1: 1, 2: 2}, dist_to_end=32.0, speed=13.0)
        metrics = defaultdict(float)
        density = {'edgeA': 0.33, 'edgeC': 0.36, 'edgeD': 0.50}

        actions = self.policy.policy_action_candidates(
            context,
            recent_history=['edgeZ'],
            cooldown_active=False,
            destination='destX',
            distance_fn=lambda edge, dest: 10.0,
            edge_density_fn=lambda edge: density.get(edge, 0.0),
            metrics=metrics,
        )

        self.assertEqual(actions, [1])
        self.assertEqual(metrics['proactive_brake_risk_candidates_seen'], 2.0)
        self.assertEqual(metrics['proactive_brake_risk_fallback_kept'], 1.0)

    def test_keeps_proactive_option_when_density_is_low(self):
        context = self._context(lane_now=[0], available=[0, 1], shifts={0: 0, 1: 1}, dist_to_end=27.0, speed=12.0)
        metrics = defaultdict(float)
        density = {'edgeA': 0.12, 'edgeC': 0.14}

        actions = self.policy.policy_action_candidates(
            context,
            recent_history=['edgeZ'],
            cooldown_active=False,
            destination='destX',
            distance_fn=lambda edge, dest: 10.0,
            edge_density_fn=lambda edge: density.get(edge, 0.0),
            metrics=metrics,
        )

        self.assertEqual(actions, [0, 1])
        self.assertEqual(metrics['proactive_brake_risk_candidates_seen'], 0.0)


if __name__ == '__main__':
    unittest.main()
