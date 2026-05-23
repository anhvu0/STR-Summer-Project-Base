import importlib
import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock


class FakeSimulationAPI:
    def __init__(self, states):
        self._states = list(states)
        self._index = 0
        self._arrived_last_step = []

    def getMinExpectedNumber(self):
        return int(self._states[self._index]["min_expected"])

    def getArrivedIDList(self):
        return list(self._arrived_last_step)

    def simulationStep(self):
        next_index = min(self._index + 1, len(self._states) - 1)
        self._arrived_last_step = list(self._states[next_index].get("arrived", []))
        self._index = next_index


class FakeVehicleAPI:
    def __init__(self, simulation_api):
        self._simulation_api = simulation_api

    def _state(self):
        return self._simulation_api._states[self._simulation_api._index]

    def getIDList(self):
        return list(self._state().get("ids", []))

    def getRoadID(self, vehicle_id):
        return self._state()["roads"][vehicle_id]

    def getSpeed(self, vehicle_id):
        return float(self._state().get("speeds", {}).get(vehicle_id, 0.0))

    def setColor(self, vehicle_id, color):
        return None

    def setRoute(self, vehicle_id, route):
        return None

    def setVia(self, vehicle_id, via_edges):
        return None

    def changeTarget(self, vehicle_id, destination):
        return None

    def getAllSubscriptionResults(self):
        return {}


class FakeEdgeAPI:
    def getLastStepVehicleNumber(self, edge_id):
        return 0

    def subscribe(self, edge_id, variables):
        return None

    def getAllSubscriptionResults(self):
        return {}


class DummyScheduler:
    def make_decisions(self, vehicles, connection_info):
        return {}


class StrSumoArrivalAccountingTests(unittest.TestCase):
    def _load_str_sumo_module(self, fake_traci):
        sys.modules.pop("core.STR_SUMO", None)
        fake_sumolib = types.ModuleType("sumolib")
        fake_sumolib.net = SimpleNamespace()
        fake_constants = types.ModuleType("traci.constants")
        fake_constants.VAR_ROAD_ID = 80
        fake_constants.VAR_SPEED = 64
        fake_constants.LAST_STEP_VEHICLE_NUMBER = 16
        fake_traci_module = types.ModuleType("traci")
        fake_traci_module.__dict__.update(vars(fake_traci))
        fake_traci_module.constants = fake_constants
        fake_modules = {
            "traci": fake_traci_module,
            "traci.constants": fake_constants,
            "sumolib": fake_sumolib,
            "core.Util": types.ModuleType("core.Util"),
            "core.target_vehicles_generation_protocols": types.ModuleType("core.target_vehicles_generation_protocols"),
            "controller.RouteController": types.ModuleType("controller.RouteController"),
        }
        with mock.patch.dict(os.environ, {"SUMO_HOME": "/tmp"}, clear=False):
            with mock.patch.dict(sys.modules, fake_modules, clear=False):
                return importlib.import_module("core.STR_SUMO")

    def test_counts_vehicle_arriving_on_final_step_before_loop_exits(self):
        simulation_api = FakeSimulationAPI([
            {
                "min_expected": 1,
                "ids": ["veh0"],
                "roads": {"veh0": "edgeA"},
                "speeds": {"veh0": 12.0},
                "arrived": [],
            },
            {
                "min_expected": 0,
                "ids": [],
                "roads": {},
                "speeds": {},
                "arrived": ["veh0"],
            },
        ])
        fake_traci = SimpleNamespace(
            simulation=simulation_api,
            simulationStep=simulation_api.simulationStep,
            vehicle=FakeVehicleAPI(simulation_api),
            edge=FakeEdgeAPI(),
            exceptions=SimpleNamespace(TraCIException=Exception),
            TraCIException=Exception,
        )
        str_sumo_module = self._load_str_sumo_module(fake_traci)
        vehicle = SimpleNamespace(
            vehicle_id="veh0",
            destination="edgeZ",
            start_time=-1.0,
            deadline=100.0,
            current_edge="",
            current_speed=0.0,
            local_destination="",
        )
        connection_info = SimpleNamespace(
            edge_list=["edgeA"],
            edge_vehicle_count={"edgeA": 0},
            edge_index_dict={"edgeA": 0},
        )
        simulation = str_sumo_module.StrSumo(
            DummyScheduler(),
            connection_info,
            {"veh0": vehicle},
        )

        total_time, end_number, deadlines_missed, stats = simulation.run(
            verbose=False,
            return_stats=True,
            print_runtime_summary=False,
        )

        self.assertEqual(end_number, 1)
        self.assertEqual(total_time, 1.0)
        self.assertEqual(deadlines_missed, 0)
        self.assertEqual(stats["vehicles_reached_destination"], 1)
        self.assertEqual(stats["completion_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
