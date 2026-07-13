"""Head-to-head: marginal-no-skip vs difference-mode MAPPO, plus both Dijkstra
baselines, on shared held-out seeds (7000+). One pass per seed; the two Dijkstra
arms are model-independent so we run them once. Paired one-sided Wilcoxon."""
import sys
try:
    import libsumo as _l
    sys.modules["traci"] = _l
    sys.modules["traci.constants"] = _l.constants
except ImportError:
    pass
import copy, os, traci
from sumolib import checkBinary
from scipy.stats import wilcoxon
from core.mappo import MAPPOConfig
from core.rl_training_pipeline import RLTrainingPipeline
from core.STR_SUMO import StrSumo
from controller.MAPPOController import MAPPOPolicy
from controller.DijkstraController import DijkstraPolicy

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
SEEDS = [7000 + i for i in range(N)]
SPAWN = 1.5
T_DUE, T_SO = 406.0, 303.0
MARGINAL = "configurations/model/mappo_policy_braess_marginal_noskip.best.pt"
DIFFERENCE = "configurations/model/mappo_policy_braess.best.pt.bak_difference"

pipe = RLTrainingPipeline(
    sumocfg_path="./configurations/braess.sumocfg", model_output_path="scratch_braess/_d.pt",
    episodes=1, spawn_interval=SPAWN, mappo_config=MAPPOConfig(), target_pattern=4,
    num_target_vehicles=240, num_random_vehicles=5, reroute_epoch_edges=1, eval_every=0,
    fast_training_profile=True, team_reward_alpha=1.0, team_reward_mode="marginal")
SUMO = checkBinary("sumo")
NET = os.path.join(pipe.sumocfg_dir, pipe.net_file)


def run(ctrl, veh, seed):
    # Pin SUMO RNG per run (libsumo leaks global RNG state across start/close cycles).
    sim = StrSumo(ctrl, pipe.connection_info, veh)
    traci.start([SUMO, "-c", pipe.runtime_sumocfg_path, "--quit-on-end", "--no-step-log",
                 "--no-warnings", "--seed", str(int(seed))])
    try:
        *_, st = sim.run(verbose=False, return_stats=True, print_runtime_summary=False)
    finally:
        try: traci.close()
        except Exception: pass
    return st["avg_travel_time"]


def mp(model, veh, det):
    return MAPPOPolicy(copy.deepcopy(veh), pipe.connection_info, model, net_xml_file=NET,
                       deterministic=det, reroute_epoch_edges=1)


arms = {"MARGINAL": [], "DIFFERENCE": [], "Dij-static": [], "Dij-dynamic": []}
print(f"seeds {SEEDS[0]}..{SEEDS[-1]} (n={N}), spawn {SPAWN}\n")
print(f"{'seed':>6} {'MARGINAL':>9} {'DIFFERENCE':>11} {'Dij-static':>11} {'Dij-dyn':>8}")
for s in SEEDS:
    veh = pipe.generate_episode_vehicles(episode_seed=int(s), spawn_interval_override=SPAWN)
    a = run(mp(MARGINAL, veh, True), copy.deepcopy(veh), s)
    b = run(mp(DIFFERENCE, veh, True), copy.deepcopy(veh), s)
    c = run(DijkstraPolicy(pipe.connection_info, weight_mode="distance"), copy.deepcopy(veh), s)
    d = run(DijkstraPolicy(pipe.connection_info, weight_mode="traveltime"), copy.deepcopy(veh), s)
    arms["MARGINAL"].append(a); arms["DIFFERENCE"].append(b)
    arms["Dij-static"].append(c); arms["Dij-dynamic"].append(d)
    print(f"{s:>6} {a:>9.1f} {b:>11.1f} {c:>11.1f} {d:>8.1f}")


def mean(x): return sum(x)/len(x)
print("\n=== means ===")
for k, v in arms.items():
    cap = (T_DUE - mean(v))/(T_DUE - T_SO)*100
    print(f"  {k:12s}: {mean(v):7.1f}s   gap-capture {cap:+.0f}% (vs T_DUE {T_DUE:.0f}/T_SO {T_SO:.0f})")


def wtest(name, x, y):
    diffs = [i-j for i, j in zip(x, y)]
    wins = sum(1 for e in diffs if e < 0)
    try:
        _, p = wilcoxon(x, y, alternative="less")
        ps = f"p={p:.4g}"
    except Exception as e:
        ps = f"n/a({e})"
    print(f"  {name:36s}: mean d={mean(diffs):+6.1f}s  faster {wins}/{len(diffs)}  Wilcoxon {ps}")


print("\n=== paired one-sided Wilcoxon (row arm < col arm) ===")
wtest("MARGINAL   < Dij-dynamic (DUE)", arms["MARGINAL"], arms["Dij-dynamic"])
wtest("MARGINAL   < Dij-static", arms["MARGINAL"], arms["Dij-static"])
wtest("DIFFERENCE < Dij-dynamic (DUE)", arms["DIFFERENCE"], arms["Dij-dynamic"])
wtest("MARGINAL   < DIFFERENCE", arms["MARGINAL"], arms["DIFFERENCE"])
wtest("DIFFERENCE < MARGINAL", arms["DIFFERENCE"], arms["MARGINAL"])
