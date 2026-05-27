'''
This test file needs the following files:
STR_SUMO.py, RouteController.py, Util.py, test.net.xml, test.rou.xml, myconfig.sumocfg and corresponding SUMO libraries.
'''
import argparse
from core.STR_SUMO import StrSumo, build_runtime_sumocfg
import os
import sys
from xml.dom.minidom import parse, parseString
from core.Util import *
from controller.RouteController import *
from controller.DijkstraController import DijkstraPolicy
from controller.MAPPOController import MAPPOPolicy
from core.target_vehicles_generation_protocols import *
import copy

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("No environment variable SUMO_HOME!")

from sumolib import checkBinary
import traci


DEFAULT_EVAL_SEEDS = ""
for i in range(1,50):
    DEFAULT_EVAL_SEEDS += str(4000+i) + ","
DEFAULT_EVAL_SEEDS = DEFAULT_EVAL_SEEDS[:-1]  # Remove trailing comma


def parse_seed_list(raw_value):
    raw_value = (raw_value or "").strip()
    if not raw_value:
        return []
    return [int(token.strip()) for token in raw_value.split(",") if token.strip()]


def build_parser():
    parser = argparse.ArgumentParser(description="Run inference-time routing controllers in SUMO.")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional model checkpoint path. Defaults to the best frozen-eval checkpoint when present, otherwise the final checkpoint.",
    )
    parser.add_argument(
        "--spawn-interval",
        type=float,
        default=2.0,
        help="Release spacing for generated controlled vehicles. Defaults to frozen-eval training value.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional single seed for reproducible inference vehicle generation. Overrides --seeds.",
    )
    parser.add_argument(
        "--seeds",
        default=DEFAULT_EVAL_SEEDS,
        help="Comma-separated seeds for repeated Dijkstra/MAPPO comparison runs.",
    )
    parser.add_argument(
        "--controlled-vehicles",
        type=int,
        default=50,
        help="Number of controlled vehicles. Defaults to the frozen-eval training value.",
    )
    parser.add_argument(
        "--uncontrolled-vehicles",
        type=int,
        default=350,
        help="Number of uncontrolled background vehicles. Defaults to the frozen-eval training value.",
    )
    parser.add_argument(
        "--pattern",
        type=int,
        default=3,
        help="Vehicle generation pattern. Defaults to the training/frozen-eval pattern.",
    )
    parser.add_argument(
        "--traci-port",
        type=int,
        default=8873,
        help="TraCI port for SUMO inference runs. Use a different value if the port is busy.",
    )
    parser.add_argument(
        "--fast-mode",
        dest="fast_mode",
        action="store_true",
        help="Enable the stripped runtime SUMO config, skip heavy output files, and reduce log overhead.",
    )
    parser.add_argument(
        "--no-fast-mode",
        dest="fast_mode",
        action="store_false",
        help="Disable fast mode and keep the heavier debug outputs.",
    )
    parser.set_defaults(fast_mode=True)
    return parser


def resolve_model_path(raw_model_path=None):
    if raw_model_path:
        return raw_model_path
    best_model_path = "./configurations/model/mappo_policy_nyc.best.pt"
    final_model_path = "./configurations/model/mappo_policy_nyc.pt"
    if os.path.exists(best_model_path):
        return best_model_path
    return final_model_path


def resolve_run_seeds(single_seed=None, seed_list_raw=None):
    if single_seed is not None:
        return [int(single_seed)]
    seeds = parse_seed_list(seed_list_raw)
    if seeds:
        return seeds
    return parse_seed_list(DEFAULT_EVAL_SEEDS)


# use vehicle generation protocols to generate vehicle list
def get_controlled_vehicles(route_filename, connection_info, \
    num_controlled_vehicles=20, num_uncontrolled_vehicles=30, pattern = 2,
    spawn_interval=None, seed=None):
    '''
    :param @route_filename <str>: the name of the route file to generate
    :param @connection_info <object>: an object that includes the map inforamtion
    :param @num_controlled_vehicles <int>: the number of vehicles controlled by the route controller
    :param @num_uncontrolled_vehicles <int>: the number of vehicles not controlled by the route controller
    :param @pattern <int>: one of four possible patterns. FORMAT:
    :param @spawn_interval <float|None>: optional controlled-vehicle release spacing
    :param @seed <int|None>: optional seed for reproducible route generation
            -- CASES BEGIN --
                #1. one start point, one destination for all target vehicles
                #2. ranged start point, one destination for all target vehicles
                #3. ranged start points, ranged destination for all target vehicles
            -- CASES ENDS --
    '''
    vehicle_dict = {}
    print(connection_info.net_filename)
    generator = target_vehicles_generator(connection_info.net_filename)

    # list of target vehicles is returned by generate_vehicles
    vehicle_list = generator.generate_vehicles(
        num_controlled_vehicles,
        num_uncontrolled_vehicles,
        pattern,
        route_filename,
        connection_info.net_filename,
        spawn_interval=spawn_interval,
        seed=seed,
    )

    for vehicle in vehicle_list:
        vehicle_dict[str(vehicle.vehicle_id)] = vehicle

    return vehicle_dict


def test_dijkstra_policy(vehicles, fast_mode=False, traci_port=8873):
    print("Testing Dijkstra's Algorithm Route Controller")
    scheduler = DijkstraPolicy(init_connection_info)
    return run_simulation(scheduler, vehicles, fast_mode=fast_mode, traci_port=traci_port)


def test_mappo(vehicles, model_path, fast_mode=False, traci_port=8873):
    print("Testing MAPPO Route Controller")
    scheduler = MAPPOPolicy(vehicles, init_connection_info, model_path)
    return run_simulation(scheduler, vehicles, fast_mode=fast_mode, traci_port=traci_port)


def run_simulation(scheduler, vehicles, fast_mode=False, traci_port=8873):

    simulation = StrSumo(scheduler, init_connection_info, vehicles)
    runtime_sumocfg = build_runtime_sumocfg("./configurations/myconfig.sumocfg", fast_mode=fast_mode)

    """
    The traci start below use sumocfg files from configurations, which match with the one you use in train_rl.py. If you change in either place, you need to change in the other one too.

    Output from main.py will be located in main_output to avoid confusion with the output generated by train_rl.py which located in configurations
    """
    traci_command = [sumo_binary, "-c", runtime_sumocfg, "--quit-on-end"]
    if fast_mode:
        traci_command.extend(["--no-step-log", "--no-warnings"])
    else:
        traci_command.extend([
            "--tripinfo-output", "./main_output/trips.trips.xml",
            "--fcd-output", "./main_output/testTrace.xml",
        ])
    traci.start(traci_command, port=int(traci_port))
    try:
        total_time, end_number, deadlines_missed, stats = simulation.run(
            verbose=not fast_mode,
            return_stats=True,
            print_runtime_summary=True,
        )
        if end_number > 0:
            avg_timespan = str(total_time / end_number)
        else:
            avg_timespan = "N/A (no vehicles reached destination)"
        print("Average timespan: {}, total vehicle number: {}, total vehicles reached destination: {}".format(avg_timespan,\
            str(len(vehicles)), str(end_number)))
        if isinstance(stats, dict):
            print(
                "Travel summary: completion={:.3f}, p50={:.2f}, p90={:.2f}, tail_gap={:.2f}, p95/p50={:.2f}, timeout_rate={:.3f}".format(
                    float(stats.get("completion_rate", 0.0)),
                    float(stats.get("p50_travel_time", 0.0)),
                    float(stats.get("p90_travel_time", 0.0)),
                    float(stats.get("tail_completion_gap_steps", 0.0)),
                    float(stats.get("p95_to_p50_travel_ratio", 0.0)),
                    float(stats.get("timeout_rate", 0.0)),
                )
            )
        print(str(deadlines_missed) + ' deadlines missed.')
        return stats
    finally:
        if traci.isLoaded():
            traci.close()


def summarize_runs(label, rows):
    rows = [row for row in rows if isinstance(row, dict)]
    if not rows:
        return

    def mean_metric(key):
        values = [float(row.get(key, 0.0)) for row in rows]
        return sum(values) / float(len(values))

    print(
        "{} summary over {} run(s): completion={:.3f}, avg={:.2f}, p50={:.2f}, p90={:.2f}, tail_gap={:.2f}, p95/p50={:.2f}, timeout={:.3f}".format(
            label,
            len(rows),
            mean_metric("completion_rate"),
            mean_metric("avg_travel_time"),
            mean_metric("p50_travel_time"),
            mean_metric("p90_travel_time"),
            mean_metric("tail_completion_gap_steps"),
            mean_metric("p95_to_p50_travel_ratio"),
            mean_metric("timeout_rate"),
        )
    )


if __name__ == "__main__":
    args = build_parser().parse_args()
    model_path = resolve_model_path(args.model_path)
    sumo_binary = checkBinary('sumo')
    # sumo_binary = checkBinary('sumo')#use this line if you do not want the UI of SUMO

    # parse config file for map file name
    dom = parse("./configurations/myconfig.sumocfg")

    net_file_node = dom.getElementsByTagName('net-file')
    net_file_attr = net_file_node[0].attributes

    net_file = net_file_attr['value'].nodeValue
    init_connection_info = ConnectionInfo("./configurations/"+net_file)

    route_file_node = dom.getElementsByTagName('route-files')
    route_file_attr = route_file_node[0].attributes
    route_file = "./configurations/"+route_file_attr['value'].nodeValue
    run_seeds = resolve_run_seeds(args.seed, args.seeds)
    print(
        "Inference scenario: controlled={}, uncontrolled={}, pattern={}, spawn_interval={}, seeds={}".format(
            args.controlled_vehicles,
            args.uncontrolled_vehicles,
            args.pattern,
            args.spawn_interval,
            ",".join(str(seed) for seed in run_seeds),
        )
    )
    dijkstra_results = []
    rl_results = []
    for seed in run_seeds:
        vehicles = get_controlled_vehicles(
            route_file,
            init_connection_info,
            args.controlled_vehicles,
            args.uncontrolled_vehicles,
            pattern=args.pattern,
            spawn_interval=args.spawn_interval,
            seed=seed,
        )
        print("Scenario seed:", seed)
        #print the controlled vehicles generated
        if not args.fast_mode:
            for vid, v in vehicles.items():
                print("id: {}, destination: {}, start time:{}, deadline: {};".format(vid, \
                    v.destination, v.start_time, v.deadline))
        dijkstra_results.append(
            test_dijkstra_policy(copy.deepcopy(vehicles), fast_mode=args.fast_mode, traci_port=args.traci_port)
        )
        print("Using MAPPO checkpoint:", model_path)
        rl_results.append(
            test_mappo(copy.deepcopy(vehicles), model_path, fast_mode=args.fast_mode, traci_port=args.traci_port)
        )
    summarize_runs("Dijkstra", dijkstra_results)
    summarize_runs("MAPPO", rl_results)
