import argparse
import copy
import csv
import os
from xml.dom.minidom import parse

import numpy as np
import traci
from sumolib import checkBinary

from controller.DijkstraController import DijkstraPolicy
from controller.QLearningController import QLearningPolicy
from core.Util import ConnectionInfo
from core.target_vehicles_generation_protocols import target_vehicles_generator

MAX_SIMULATION_STEPS = 2500


def parse_sumocfg(sumocfg_path):
    dom = parse(sumocfg_path)
    net_file = dom.getElementsByTagName("net-file")[0].attributes["value"].nodeValue
    route_file = dom.getElementsByTagName("route-files")[0].attributes["value"].nodeValue
    root = os.path.dirname(sumocfg_path)
    return os.path.join(root, net_file), os.path.join(root, route_file)


def generate_episode_vehicles(connection_info, route_file, seed, pattern, num_target, num_random, spawn_interval):
    generator = target_vehicles_generator(connection_info.net_filename)
    vehicle_list = generator.generate_vehicles(
        num_target_vehicles=num_target,
        num_random_vehicles=num_random,
        pattern=pattern,
        target_xml_file=route_file,
        net_xml_file=connection_info.net_filename,
        spawn_interval=spawn_interval,
        seed=seed,
    )
    return {str(v.vehicle_id): v for v in vehicle_list}


def run_policy(sumocfg, connection_info, vehicles, policy_name, model_path=None, seed=0):
    if policy_name == "dijkstra":
        controller = DijkstraPolicy(connection_info)
    else:
        controller = QLearningPolicy(vehicles, connection_info, model_path)

    traci.start([checkBinary("sumo"), "-c", sumocfg, "--seed", str(seed), "--quit-on-end"])
    seen_ids = set()
    arrived_ids = set()
    teleported_ids = set()
    release_step = {}
    travel_times = []
    step = 0
    try:
        while traci.simulation.getMinExpectedNumber() > 0 and step <= MAX_SIMULATION_STEPS:
            live_ids = set(traci.vehicle.getIDList())
            for edge in connection_info.edge_list:
                connection_info.edge_vehicle_count[edge] = traci.edge.getLastStepVehicleNumber(edge)

            to_direct = []
            for vid in live_ids:
                if vid in vehicles and vid not in seen_ids:
                    seen_ids.add(vid)
                    release_step[vid] = step
                if vid in vehicles:
                    current_edge = traci.vehicle.getRoadID(vid)
                    if current_edge in connection_info.edge_index_dict and current_edge != vehicles[vid].destination:
                        if current_edge != vehicles[vid].current_edge:
                            vehicles[vid].current_edge = current_edge
                            vehicles[vid].current_speed = traci.vehicle.getSpeed(vid)
                            to_direct.append(vehicles[vid])

            decisions = controller.make_decisions(to_direct, connection_info)
            for vid, decision in decisions.items():
                if vid not in live_ids:
                    continue
                try:
                    if isinstance(decision, (list, tuple)) and len(decision) >= 2:
                        traci.vehicle.setRoute(vid, list(decision))
                    else:
                        dest = vehicles[vid].destination
                        if decision != dest:
                            traci.vehicle.setVia(vid, [decision])
                        else:
                            traci.vehicle.setVia(vid, [])
                        traci.vehicle.changeTarget(vid, dest)
                except traci.TraCIException:
                    pass

            for vid in traci.simulation.getArrivedIDList():
                if vid in vehicles and vid not in arrived_ids:
                    arrived_ids.add(vid)
                    travel_times.append(step - release_step.get(vid, step))

            teleported_ids.update(traci.simulation.getStartingTeleportIDList())
            teleported_ids.update(traci.simulation.getEndingTeleportIDList())
            try:
                teleported_ids.update(traci.vehicle.getTeleportingList())
            except Exception:
                pass
            traci.simulationStep()
            step += 1
    finally:
        if traci.isLoaded():
            traci.close()

    completed = list(travel_times)
    avg_tt = float(np.mean(completed)) if completed else 0.0
    p50_tt = float(np.percentile(completed, 50)) if completed else 0.0
    p90_tt = float(np.percentile(completed, 90)) if completed else 0.0
    completion_rate = len(arrived_ids) / float(max(len(vehicles), 1))
    stuck = [vid for vid in vehicles if vid not in arrived_ids and vid not in teleported_ids]
    return {
        "avg_tt": avg_tt,
        "p50_tt": p50_tt,
        "p90_tt": p90_tt,
        "completion_rate": completion_rate,
        "teleports": len([vid for vid in teleported_ids if vid in vehicles]),
        "stuck_terminations": len(stuck),
        "episode_truncations": int(step > MAX_SIMULATION_STEPS),
    }


def main():
    p = argparse.ArgumentParser(description="Fair fixed-seed RL vs Dijkstra evaluation harness.")
    p.add_argument("--sumocfg", default="./configurations/myconfig.sumocfg")
    p.add_argument("--model", default="./configurations/model/rl_model_map.h5")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed-start", type=int, default=1000)
    p.add_argument("--target-pattern", type=int, default=2)
    p.add_argument("--num-target", type=int, default=50)
    p.add_argument("--num-random", type=int, default=50)
    p.add_argument("--spawn-interval", type=float, default=4.0)
    p.add_argument("--out-csv", default="./configurations/rl_vs_dijkstra_episode_compare.csv")
    args = p.parse_args()

    net_file, route_file = parse_sumocfg(args.sumocfg)
    conn = ConnectionInfo(net_file)

    rows = []
    for i in range(args.episodes):
        seed = args.seed_start + i
        vehicles = generate_episode_vehicles(
            conn, route_file, seed, args.target_pattern, args.num_target, args.num_random, args.spawn_interval
        )
        dij = run_policy(args.sumocfg, conn, copy.deepcopy(vehicles), "dijkstra", seed=seed)
        rl = run_policy(args.sumocfg, conn, copy.deepcopy(vehicles), "rl", model_path=args.model, seed=seed)
        rows.append({"episode": i, "seed": seed, "policy": "dijkstra", **dij})
        rows.append({"episode": i, "seed": seed, "policy": "rl", **rl})
        print(f"[EP {i:03d}] seed={seed} dijkstra_avg_tt={dij['avg_tt']:.2f} rl_avg_tt={rl['avg_tt']:.2f}")

    with open(args.out_csv, "w", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else ["episode", "seed", "policy"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved per-episode comparison to {args.out_csv}")


if __name__ == "__main__":
    main()
