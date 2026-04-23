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
from controller.QLearningController import QLearningPolicy
from core.target_vehicles_generation_protocols import *
import copy

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("No environment variable SUMO_HOME!")

from sumolib import checkBinary
import traci


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
        default=None,
        help="Optional release spacing for generated controlled vehicles.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible inference vehicle generation.",
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
    best_model_path = "./configurations/model/rl_model_map.best.h5"
    final_model_path = "./configurations/model/rl_model_map.h5"
    if os.path.exists(best_model_path):
        return best_model_path
    return final_model_path


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


def test_dijkstra_policy(vehicles, fast_mode=False):
    print("Testing Dijkstra's Algorithm Route Controller")
    scheduler = DijkstraPolicy(init_connection_info)
    run_simulation(scheduler, vehicles, fast_mode=fast_mode)


def test_q_learning(vehicles, model_path, fast_mode=False):
    print("Testing Q Learning Route Controller")
    scheduler = QLearningPolicy(vehicles, init_connection_info, model_path)
    run_simulation(scheduler, vehicles, fast_mode=fast_mode)


def run_simulation(scheduler, vehicles, fast_mode=False):

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
    traci.start(traci_command)
    try:
        total_time, end_number, deadlines_missed, _ = simulation.run(
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
        print(str(deadlines_missed) + ' deadlines missed.')
    finally:
        if traci.isLoaded():
            traci.close()


if __name__ == "__main__":
    args = build_parser().parse_args()
    model_path = resolve_model_path(args.model_path)
    sumo_binary = checkBinary('sumo-gui')
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
    # Pattern 2: multiple origins with one shared destination.
    vehicles = get_controlled_vehicles(
        route_file,
        init_connection_info,
        100,
        100,
        pattern=3,
        spawn_interval=args.spawn_interval,
        seed=args.seed,
    )
    #print the controlled vehicles generated
    if not args.fast_mode:
        for vid, v in vehicles.items():
            print("id: {}, destination: {}, start time:{}, deadline: {};".format(vid, \
                v.destination, v.start_time, v.deadline))
    test_dijkstra_policy(copy.deepcopy(vehicles), fast_mode=args.fast_mode)
    print("Using RL checkpoint:", model_path)
    test_q_learning(copy.deepcopy(vehicles), model_path, fast_mode=args.fast_mode)
