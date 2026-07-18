"""Train MAPPO on the two-route-yield benchmark of Psarou et al. (arXiv:2502.13188).

All 22 vehicles of the paper's demand are controlled (100% penetration), so the
fleet objective coincides with system welfare; the trained policy is then also
deployed on the paper's 10-AV / 12-human split by eval_two_route.py.

Usage:
  SUMO_HOME=.venv/lib/python3.14/site-packages/sumo PYTHONPATH=.:scratch_two_route \
    .venv/bin/python scratch_two_route/train_two_route.py [--episodes N] [--torch-seed S]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import libsumo as _libsumo
    sys.modules["traci"] = _libsumo
    sys.modules["traci.constants"] = _libsumo.constants
except ImportError:
    pass

from core.mappo import MAPPOConfig
from core.rl_training_pipeline import RLTrainingPipeline
from core import Util

from two_route_common import DEPARTS, ROUTE0, ROUTE1

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = "E2"


def write_training_routes(path, controlled_ids):
    """Controlled vehicles get a start-edge-only route (the controller assigns
    the real route live); any others get a fixed route 0."""
    with open(path, "w") as f:
        f.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
                'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')
        for vid, dep in sorted(DEPARTS, key=lambda x: x[1]):
            if vid in controlled_ids:
                f.write(f'\t<vehicle depart="{dep:.1f}" id="{vid}">\n'
                        f'\t\t<route edges="E0"/>\n\t</vehicle>\n')
            else:
                f.write(f'\t<vehicle depart="{dep:.1f}" id="{vid}">\n'
                        f'\t\t<route edges="{ROUTE0}"/>\n\t</vehicle>\n')
        f.write("</routes>\n")


# Merge-conflict externality (extends the same-edge follower rule of the
# marginal-cost reward). At this network's single priority merge, E5->E2 has
# right of way and E7->E2 yields, so a vehicle on the major approach delays the
# queue on the yielding approach rather than any same-edge follower; the
# same-edge proxy prices that externality at zero. Charge a vehicle on the
# major approach for the vehicles currently queued on the yielding approach,
# with the same scale and per-step cap as the follower rule.
MAJOR_APPROACH = {"E3", "E4", "E5"}
YIELD_APPROACH = {"E1", "E7"}
QUEUE_SLOWNESS = 0.3


class TwoRoutePipeline(RLTrainingPipeline):
    """Fixed 22-vehicle demand from the benchmark; ignores demand seeds."""

    controlled_ids = {v for v, _ in DEPARTS}  # 100% penetration for training
    merge_conflict_pricing = True
    # Calibrated to the measured externality: one defection to the priority
    # route costs the fleet ~64 vehicle-seconds (~4.5 reward units at the
    # 0.07/s step cost), so the charge over one ~8 s major-approach pass with a
    # typical 6-10 vehicle yield queue must reach that order, not the ~1 unit
    # the follower scale (0.015) yields.
    merge_cost_scale = 0.10

    def _compute_var_edge_behind_counts(self, step_snapshots):
        counts = super()._compute_var_edge_behind_counts(step_snapshots)
        self._merge_conflict_count = {}
        if (not self.merge_conflict_pricing or self.team_reward_mode != "marginal"
                or not step_snapshots):
            return counts
        queue = sum(
            1 for vid, snap in step_snapshots.items()
            if str(snap.edge_id) in YIELD_APPROACH
            and self._fleet_slowness_by_vehicle.get(vid, 0.0) >= QUEUE_SLOWNESS
        )
        if queue > 0:
            self._merge_conflict_count = {
                vid: queue for vid, snap in step_snapshots.items()
                if str(snap.edge_id) in MAJOR_APPROACH
            }
        return counts

    def _team_congestion_cost(self, elapsed, vehicle=None):
        cost = super()._team_congestion_cost(elapsed, vehicle)
        if self.team_reward_mode != "marginal" or self.team_reward_alpha <= 0.0:
            return cost
        vid = getattr(vehicle, "vehicle_id", None)
        blocked = int(getattr(self, "_merge_conflict_count", {}).get(vid, 0))
        if blocked <= 0:
            return cost
        # Unlike the follower rule this is NOT gated on own slowness: the
        # blocker moves at free flow precisely because the others yield to it.
        blocked = min(blocked, int(self.marginal_cost_behind_cap))
        extra = (float(self.team_reward_alpha) * float(self.merge_cost_scale)
                 * float(blocked) * float(max(elapsed, 0.0)))
        extra = min(extra, float(self.marginal_cost_per_step_cap) * float(max(elapsed, 0.0)))
        return float(cost + extra)

    def generate_episode_vehicles(self, episode_seed=None, spawn_interval_override=None):
        route_path = os.path.join(self.sumocfg_dir, self.route_file)
        write_training_routes(route_path, self.controlled_ids)
        vehicles = {}
        for vid, dep in DEPARTS:
            if vid in self.controlled_ids:
                vehicles[str(vid)] = Util.Vehicle(str(vid), DEST, dep, dep + 300.0)
        return vehicles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--torch-seed", type=int, default=101)
    ap.add_argument("--team-reward-mode", default="marginal",
                    choices=["difference", "shared", "marginal"])
    ap.add_argument("--model-tag", default="two_route_marginal")
    ap.add_argument("--disable-route-balance", action="store_true")
    ap.add_argument("--disable-merge-pricing", action="store_true")
    ap.add_argument("--merge-cost-scale", type=float, default=0.10)
    ap.add_argument("--eval-every", type=int, default=5,
                    help="Frozen held-out eval cadence. Raise for long (6000-ep) "
                         "matched-budget runs so eval overhead stays bounded.")
    args = ap.parse_args()

    import random
    import numpy as np
    import torch
    random.seed(args.torch_seed)
    np.random.seed(args.torch_seed)
    torch.manual_seed(args.torch_seed)

    cfg = MAPPOConfig(
        actor_learning_rate=1.0e-3,
        critic_learning_rate=1.0e-3,
        gamma=0.995,
        gae_lambda=0.95,
        clip_epsilon=0.05,
        entropy_coef=0.15,
        entropy_coef_end=0.02,
        value_coef=0.5,
        update_epochs=8,
        minibatch_size=64,
        # 22 vehicles produce ~22-60 decisions per episode; update every episode or two
        min_transitions_per_update=48,
        target_kl=0.015,
        normalize_value_targets=True,
    )
    model_dir = os.path.join(REPO, "configurations", "model")
    pipe = TwoRoutePipeline(
        sumocfg_path=os.path.join(REPO, "configurations", "two_route.sumocfg"),
        model_output_path=os.path.join(model_dir, f"mappo_{args.model_tag}.pt"),
        best_model_output_path=os.path.join(model_dir, f"mappo_{args.model_tag}.best.pt"),
        episodes=args.episodes,
        spawn_interval=1.0,
        mappo_config=cfg,
        eval_every=args.eval_every,
        frozen_eval_seeds=[6000, 6001, 6002],
        eval_spawn_interval=1.0,
        fast_training_profile=False,
        target_pattern=4,
        num_target_vehicles=22,
        num_random_vehicles=0,
        team_reward_alpha=1.0,
        team_reward_scale=0.3,
        team_reward_mode=args.team_reward_mode,
        marginal_cost_scale=0.015,
        skip_forced_route_epochs=False,
        eval_deterministic=True,
        route_reservations=True,
        eval_stochastic_samples=0,
        reroute_epoch_edges=1,
    )
    # The default bottleneck rule (single-lane edges >= 150 m) matches nothing on
    # this ~300 m network; on it the congestible edges are the single-lane
    # approaches E1 (70 m) and E4 (100 m) and the shared exit E2 (93 m).
    # _congestible_edges is derived in __init__, so recompute it here.
    pipe.marginal_cost_min_edge_length = 60.0
    pipe._congestible_edges = frozenset(
        edge_id
        for edge_id in pipe._edge_list
        if max(len(pipe.connection_info.edge_lane_ids.get(edge_id, [])), 1) == 1
        and float(pipe.connection_info.edge_length_dict.get(edge_id, 0.0))
        >= pipe.marginal_cost_min_edge_length
    )
    print("[two_route] congestible edges:", sorted(pipe._congestible_edges), flush=True)
    # On this network the system optimum concentrates everyone on route 0, so the
    # route-balance shaping (which pays for spreading across routes) fights the
    # objective; the merge externality is cross-route, which the same-edge
    # follower marginal proxy cannot see (use --team-reward-mode shared).
    if args.disable_route_balance:
        pipe.route_balance_reward_scale = 0.0
    pipe.merge_conflict_pricing = not args.disable_merge_pricing
    pipe.merge_cost_scale = float(args.merge_cost_scale)
    pipe.run()


if __name__ == "__main__":
    main()
