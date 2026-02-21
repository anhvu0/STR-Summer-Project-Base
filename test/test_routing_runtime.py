import unittest

from core.routing_runtime import apply_routing_decision


class _RouteResult:
    def __init__(self, edges):
        self.edges = edges


class _Simulation:
    def __init__(self, routes):
        self.routes = routes

    def findRoute(self, src, dst):
        return _RouteResult(self.routes.get((src, dst), []))


class _Vehicle:
    def __init__(self):
        self.last_route = None

    def setRoute(self, vehicle_id, edges):
        self.last_route = (vehicle_id, list(edges))


class _Traci:
    def __init__(self, routes):
        self.simulation = _Simulation(routes)
        self.vehicle = _Vehicle()


class ApplyRoutingDecisionTests(unittest.TestCase):
    def test_prefers_local_target_when_reachable(self):
        traci = _Traci({("e1", "e2"): ["e1", "e2"]})

        result = apply_routing_decision(traci, "veh0", "e1", "e2", "e9")

        self.assertEqual(result, "e2")
        self.assertEqual(traci.vehicle.last_route, ("veh0", ["e1", "e2"]))

    def test_falls_back_to_destination_when_local_target_invalid(self):
        traci = _Traci({("e1", "e9"): ["e1", "e4", "e9"]})

        result = apply_routing_decision(traci, "veh0", "e1", "bad_local", "e9")

        self.assertEqual(result, "e9")
        self.assertEqual(traci.vehicle.last_route, ("veh0", ["e1", "e4", "e9"]))

    def test_returns_none_when_no_route_found(self):
        traci = _Traci({})

        result = apply_routing_decision(traci, "veh0", "e1", "e2", "e9")

        self.assertIsNone(result)
        self.assertIsNone(traci.vehicle.last_route)


if __name__ == "__main__":
    unittest.main()
