import unittest
from collections import defaultdict
from pathlib import Path
import sys
import types

sys.modules.setdefault('traci', types.SimpleNamespace(TraCIException=Exception))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.junction_decision_engine import DecisionContext
from core.junction_decision_engine import JunctionDecisionEngine
from core.shared_decision_policy import SharedDecisionPolicy


class DummyConnectionInfo:
    edge_list = []
    outgoing_edges_dict = {
        'edgeA': {'L': 'edgeB', 'S': 'edgeC', 'R': 'edgeD'},
    }
    lane_outgoing_edges_dict = {
        'edgeA_0': {'L': 'edgeB', 'S': 'edgeC', 'R': 'edgeD'},
        'edgeA_1': {'S': 'edgeC', 'R': 'edgeD'},
    }
    edge_lane_ids = {
        'edgeA': ['edgeA_0', 'edgeA_1'],
    }


class DummyDecisionEngine:
    commit_min_distance = 8.0
    commit_time_s = 0.45
    proactive_extra_buffer_m = 6.0
    lane_change_margin_m = 10.0
    proactive_safety_margin_m = 8.0
    route_pending_hard_timeout_steps = 60
    pending_progress_timeout_steps = 32

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

    def test_prunes_congested_lane_now_branch_when_cleaner_branch_is_comparable(self):
        context = self._context(lane_now=[0, 1], available=[0, 1], shifts={0: 0, 1: 0}, dist_to_end=24.0, speed=10.0)
        metrics = defaultdict(float)
        density = {'edgeB': 0.56, 'edgeC': 0.14}
        distance = {'edgeB': 105.0, 'edgeC': 100.0, 'edgeA': 120.0}

        actions = self.policy.policy_action_candidates(
            context,
            recent_history=['edgeZ'],
            cooldown_active=False,
            destination='destX',
            distance_fn=lambda edge, dest: distance.get(edge, 100.0),
            edge_density_fn=lambda edge: density.get(edge, 0.0),
            metrics=metrics,
        )

        self.assertEqual(actions, [1])
        self.assertEqual(metrics['lane_now_congestion_candidates_seen'], 1.0)
        self.assertEqual(metrics['lane_now_congestion_candidates_rejected'], 1.0)

    def test_keeps_congested_lane_now_branch_when_it_is_much_shorter(self):
        context = self._context(lane_now=[0, 1], available=[0, 1], shifts={0: 0, 1: 0}, dist_to_end=24.0, speed=10.0)
        metrics = defaultdict(float)
        density = {'edgeB': 0.56, 'edgeC': 0.14}
        distance = {'edgeB': 60.0, 'edgeC': 100.0, 'edgeA': 120.0}

        actions = self.policy.policy_action_candidates(
            context,
            recent_history=['edgeZ'],
            cooldown_active=False,
            destination='destX',
            distance_fn=lambda edge, dest: distance.get(edge, 100.0),
            edge_density_fn=lambda edge: density.get(edge, 0.0),
            metrics=metrics,
        )

        self.assertEqual(actions, [0, 1])
        self.assertEqual(metrics['lane_now_congestion_candidates_seen'], 1.0)
        self.assertEqual(metrics['lane_now_congestion_candidates_rejected'], 0.0)

    def test_prunes_moderately_congested_lane_now_branch_when_close_to_junction(self):
        context = self._context(lane_now=[0, 1], available=[0, 1], shifts={0: 0, 1: 0}, dist_to_end=12.0, speed=12.0)
        metrics = defaultdict(float)
        density = {'edgeB': 0.31, 'edgeC': 0.18}
        distance = {'edgeB': 88.0, 'edgeC': 100.0, 'edgeA': 120.0}

        actions = self.policy.policy_action_candidates(
            context,
            recent_history=['edgeZ'],
            cooldown_active=False,
            destination='destX',
            distance_fn=lambda edge, dest: distance.get(edge, 100.0),
            edge_density_fn=lambda edge: density.get(edge, 0.0),
            metrics=metrics,
        )

        self.assertEqual(actions, [1])
        self.assertEqual(metrics['lane_now_congestion_candidates_seen'], 1.0)
        self.assertEqual(metrics['lane_now_congestion_candidates_rejected'], 1.0)

    def test_keeps_same_moderately_congested_lane_now_branch_when_far_from_junction(self):
        context = self._context(lane_now=[0, 1], available=[0, 1], shifts={0: 0, 1: 0}, dist_to_end=80.0, speed=12.0)
        metrics = defaultdict(float)
        density = {'edgeB': 0.31, 'edgeC': 0.18}
        distance = {'edgeB': 88.0, 'edgeC': 100.0, 'edgeA': 120.0}

        actions = self.policy.policy_action_candidates(
            context,
            recent_history=['edgeZ'],
            cooldown_active=False,
            destination='destX',
            distance_fn=lambda edge, dest: distance.get(edge, 100.0),
            edge_density_fn=lambda edge: density.get(edge, 0.0),
            metrics=metrics,
        )

        self.assertEqual(actions, [0, 1])
        self.assertEqual(metrics['lane_now_congestion_candidates_seen'], 0.0)
        self.assertEqual(metrics['lane_now_congestion_candidates_rejected'], 0.0)


class SharedDecisionPolicyPendingReleaseTests(unittest.TestCase):
    def setUp(self):
        self.engine = JunctionDecisionEngine(DummyConnectionInfo(), None, ['L', 'S', 'R'])
        self.policy = SharedDecisionPolicy(DummyConnectionInfo(), self.engine, ['L', 'S', 'R'])

    def _lane_now_context(
        self,
        *,
        step=0,
        speed=0.0,
        lane_position=0.0,
        dist_to_end=100.0,
        lane_now=None,
        available=None,
        shifts=None,
        forced_action=0,
        branch_with_choice=False,
        skip_reason='forced_single_path',
    ):
        lane_now = [0] if lane_now is None else list(lane_now)
        available = [0] if available is None else list(available)
        shifts = {0: 0} if shifts is None else dict(shifts)
        return DecisionContext(
            vehicle_id='veh0',
            edge_id='edgeA',
            destination='destX',
            step=step,
            speed=speed,
            lane_id='edgeA_0',
            lane_index=0,
            lane_count=1,
            dist_to_end=dist_to_end,
            edge_valid_actions=[0, 1],
            lane_feasible_now_actions=lane_now,
            reachable_with_lane_change_actions=sorted(set(available)),
            available_actions=available,
            required_lane_shift=shifts,
            commit_window=False,
            forced_action=forced_action,
            branch_with_choice=branch_with_choice,
            skip_reason=skip_reason,
        )

    def _proactive_context(self, *, step=0, speed=6.0, lane_position=0.0, dist_to_end=100.0):
        return DecisionContext(
            vehicle_id='veh0',
            edge_id='edgeA',
            destination='destX',
            step=step,
            speed=speed,
            lane_id='edgeA_0',
            lane_index=0,
            lane_count=2,
            dist_to_end=dist_to_end,
            edge_valid_actions=[0, 1],
            lane_feasible_now_actions=[0],
            reachable_with_lane_change_actions=[0, 1],
            available_actions=[0, 1],
            required_lane_shift={0: 0, 1: 1},
            commit_window=False,
            forced_action=None,
            branch_with_choice=True,
            skip_reason=None,
        )

    def test_stalled_lane_now_pending_is_kept(self):
        context = self._lane_now_context(step=0)
        pending = self.policy.build_route_pending(
            state=None,
            action_idx=0,
            committed_next_edge='edgeB',
            decision_edge='edgeA',
            step=0,
            destination='destX',
            context=context,
            lane_change_requested=False,
            decision_id='d0',
            origin_mode='lane_now',
            action_source='policy',
            full_route=['edgeA', 'edgeB'],
            decision_open_recorded=True,
        )

        release = self.policy.evaluate_route_pending_release(
            pending,
            context=self._lane_now_context(step=5, speed=0.0),
            step=5,
            lane_position_now=0.0,
        )

        self.assertFalse(release.should_release)
        self.assertEqual(release.release_reason, None)
        self.assertFalse(release.progress_view['made_progress'])

    def test_moving_lane_now_pending_is_kept(self):
        context = self._lane_now_context(step=0)
        pending = self.policy.build_route_pending(
            state=None,
            action_idx=0,
            committed_next_edge='edgeB',
            decision_edge='edgeA',
            step=0,
            destination='destX',
            context=context,
            lane_change_requested=False,
            decision_id='d0',
            origin_mode='lane_now',
            action_source='policy',
            full_route=['edgeA', 'edgeB'],
            decision_open_recorded=True,
        )

        release = self.policy.evaluate_route_pending_release(
            pending,
            context=self._lane_now_context(step=10, speed=6.0, lane_position=20.0, dist_to_end=80.0),
            step=10,
            lane_position_now=20.0,
        )

        self.assertFalse(release.should_release)
        self.assertTrue(release.progress_view['made_progress'])

    def test_fallback_filters_congested_lane_now_branch_when_cleaner_branch_is_comparable(self):
        context = self._lane_now_context(
            step=12,
            speed=12.0,
            dist_to_end=12.0,
            lane_now=[0, 1],
            available=[0, 1, 2],
            shifts={0: 0, 1: 0, 2: 1},
            forced_action=None,
            branch_with_choice=True,
            skip_reason=None,
        )
        density = {'edgeB': 0.31, 'edgeC': 0.18}
        distance = {'edgeB': 88.0, 'edgeC': 100.0, 'edgeD': 140.0}

        action = self.policy.select_fallback_action(
            context,
            blocked_action=2,
            destination='destX',
            recent_history=['edgeZ'],
            distance_fn=lambda edge, dest: distance.get(edge, float('inf')),
            edge_density_fn=lambda edge: density.get(edge, 0.0),
            lane_now_only=True,
        )

        self.assertEqual(action, 1)

    def test_fallback_keeps_congested_lane_now_branch_when_it_is_much_shorter(self):
        context = self._lane_now_context(
            step=12,
            speed=12.0,
            dist_to_end=12.0,
            lane_now=[0, 1],
            available=[0, 1, 2],
            shifts={0: 0, 1: 0, 2: 1},
            forced_action=None,
            branch_with_choice=True,
            skip_reason=None,
        )
        density = {'edgeB': 0.31, 'edgeC': 0.18}
        distance = {'edgeB': 60.0, 'edgeC': 100.0, 'edgeD': 140.0}

        action = self.policy.select_fallback_action(
            context,
            blocked_action=2,
            destination='destX',
            recent_history=['edgeZ'],
            distance_fn=lambda edge, dest: distance.get(edge, float('inf')),
            edge_density_fn=lambda edge: density.get(edge, 0.0),
            lane_now_only=True,
        )

        self.assertEqual(action, 0)

    def test_stalled_proactive_pending_still_aborts(self):
        context = self._proactive_context(step=0, speed=8.0)
        pending = self.policy.build_route_pending(
            state=None,
            action_idx=1,
            committed_next_edge='edgeC',
            decision_edge='edgeA',
            step=0,
            destination='destX',
            context=context,
            lane_change_requested=True,
            decision_id='d1',
            origin_mode='proactive',
            action_source='policy',
            full_route=['edgeA', 'edgeC'],
            decision_open_recorded=True,
        )

        release = self.policy.evaluate_route_pending_release(
            pending,
            context=self._proactive_context(step=5, speed=0.0),
            step=5,
            lane_position_now=0.0,
        )

        self.assertTrue(release.should_release)
        self.assertEqual(release.release_reason, 'route_no_progress_abort')
        self.assertFalse(release.release_as_timeout)

    def test_stalled_lane_now_pending_replans_to_cleaner_comparable_branch(self):
        context = self._lane_now_context(
            lane_now=[0, 1],
            available=[0, 1],
            shifts={0: 0, 1: 0},
            forced_action=None,
            branch_with_choice=True,
            skip_reason=None,
        )
        pending = self.policy.build_route_pending(
            state=None,
            action_idx=0,
            committed_next_edge='edgeB',
            decision_edge='edgeA',
            step=0,
            destination='destX',
            context=context,
            lane_change_requested=False,
            decision_id='d0',
            origin_mode='lane_now',
            action_source='policy',
            full_route=['edgeA', 'edgeB'],
            decision_open_recorded=True,
        )
        density = {'edgeB': 0.44, 'edgeC': 0.18}
        distance = {'edgeB': 100.0, 'edgeC': 112.0}

        release = self.policy.evaluate_route_pending_release(
            pending,
            context=self._lane_now_context(
                step=12,
                speed=0.4,
                lane_now=[0, 1],
                available=[0, 1],
                shifts={0: 0, 1: 0},
                forced_action=None,
                branch_with_choice=True,
                skip_reason=None,
            ),
            step=12,
            lane_position_now=0.0,
            edge_density_fn=lambda edge: density.get(edge, 0.0),
            distance_fn=lambda edge, dest: distance.get(edge, float('inf')),
        )

        self.assertTrue(release.should_release)
        self.assertEqual(release.release_reason, 'route_no_progress_abort')
        self.assertFalse(release.release_as_timeout)

    def test_stalled_lane_now_pending_kept_when_cleaner_branch_is_much_longer(self):
        context = self._lane_now_context(
            lane_now=[0, 1],
            available=[0, 1],
            shifts={0: 0, 1: 0},
            forced_action=None,
            branch_with_choice=True,
            skip_reason=None,
        )
        pending = self.policy.build_route_pending(
            state=None,
            action_idx=0,
            committed_next_edge='edgeB',
            decision_edge='edgeA',
            step=0,
            destination='destX',
            context=context,
            lane_change_requested=False,
            decision_id='d0',
            origin_mode='lane_now',
            action_source='policy',
            full_route=['edgeA', 'edgeB'],
            decision_open_recorded=True,
        )
        density = {'edgeB': 0.44, 'edgeC': 0.18}
        distance = {'edgeB': 100.0, 'edgeC': 160.0}

        release = self.policy.evaluate_route_pending_release(
            pending,
            context=self._lane_now_context(
                step=12,
                speed=0.4,
                lane_now=[0, 1],
                available=[0, 1],
                shifts={0: 0, 1: 0},
                forced_action=None,
                branch_with_choice=True,
                skip_reason=None,
            ),
            step=12,
            lane_position_now=0.0,
            edge_density_fn=lambda edge: density.get(edge, 0.0),
            distance_fn=lambda edge, dest: distance.get(edge, float('inf')),
        )

        self.assertFalse(release.should_release)
        self.assertEqual(release.release_reason, None)

    def test_old_moving_lane_now_pending_is_kept_before_stale_max_age(self):
        context = self._lane_now_context()
        pending = self.policy.build_route_pending(
            state=None,
            action_idx=0,
            committed_next_edge='edgeB',
            decision_edge='edgeA',
            step=0,
            destination='destX',
            context=context,
            lane_change_requested=False,
            decision_id='d0',
            origin_mode='lane_now',
            action_source='policy',
            full_route=['edgeA', 'edgeB'],
            decision_open_recorded=True,
        )
        old_step = int(self.policy.lane_now_stale_timeout_max_age_steps) - 1

        release = self.policy.evaluate_route_pending_release(
            pending,
            context=self._lane_now_context(step=old_step, speed=6.0),
            step=old_step,
            lane_position_now=0.0,
            edge_density_fn=lambda edge: 0.0,
            distance_fn=lambda edge, dest: 100.0,
        )

        self.assertFalse(release.should_release)
        self.assertEqual(release.release_reason, None)

    def test_very_old_lane_now_pending_times_out_without_relief_branch(self):
        context = self._lane_now_context()
        pending = self.policy.build_route_pending(
            state=None,
            action_idx=0,
            committed_next_edge='edgeB',
            decision_edge='edgeA',
            step=0,
            destination='destX',
            context=context,
            lane_change_requested=False,
            decision_id='d0',
            origin_mode='lane_now',
            action_source='policy',
            full_route=['edgeA', 'edgeB'],
            decision_open_recorded=True,
        )
        old_step = int(self.policy.lane_now_stale_timeout_max_age_steps) + 1

        release = self.policy.evaluate_route_pending_release(
            pending,
            context=self._lane_now_context(step=old_step, speed=6.0),
            step=old_step,
            lane_position_now=0.0,
            edge_density_fn=lambda edge: 0.0,
            distance_fn=lambda edge, dest: 100.0,
        )

        self.assertTrue(release.should_release)
        self.assertEqual(release.release_reason, 'route_hard_timeout')
        self.assertTrue(release.release_as_timeout)

    def test_invalid_lane_now_shift_times_out_after_stall_threshold(self):
        context = self._lane_now_context(shifts={0: 0})
        pending = self.policy.build_route_pending(
            state=None,
            action_idx=0,
            committed_next_edge='edgeB',
            decision_edge='edgeA',
            step=0,
            destination='destX',
            context=context,
            lane_change_requested=False,
            decision_id='d0',
            origin_mode='lane_now',
            action_source='policy',
            full_route=['edgeA', 'edgeB'],
            decision_open_recorded=True,
        )
        stale_step = int(self.policy.lane_now_stale_timeout_min_stall_steps)

        release = self.policy.evaluate_route_pending_release(
            pending,
            context=self._lane_now_context(
                step=stale_step,
                speed=6.0,
                lane_now=[],
                available=[],
                shifts={},
            ),
            step=stale_step,
            lane_position_now=0.0,
            edge_density_fn=lambda edge: 0.0,
            distance_fn=lambda edge, dest: 100.0,
        )

        self.assertTrue(release.should_release)
        self.assertEqual(release.release_reason, 'route_hard_timeout')
        self.assertTrue(release.release_as_timeout)


if __name__ == '__main__':
    unittest.main()
