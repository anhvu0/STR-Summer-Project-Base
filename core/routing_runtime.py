"""Runtime helpers for safely applying controller routing decisions in SUMO."""


def _build_route_edges(route_result, current_edge):
    """Normalize a TraCI route result into a route that can be fed to setRoute."""
    if route_result is None:
        return []

    edges = list(getattr(route_result, "edges", []) or [])
    if not edges:
        return []

    if edges[0] != current_edge:
        edges.insert(0, current_edge)

    return edges


def apply_routing_decision(traci_module, vehicle_id, current_edge, local_target_edge, final_destination):
    """
    Replace `changeTarget` with explicit route assignment.

    We first try to build a full route to the local target. If unreachable, we
    fall back to the final destination. Returns the edge this routing call used
    as a destination, or ``None`` when no valid route could be assigned.
    """
    candidate_destinations = []
    if local_target_edge:
        candidate_destinations.append(local_target_edge)
    if final_destination and final_destination not in candidate_destinations:
        candidate_destinations.append(final_destination)

    for destination in candidate_destinations:
        try:
            route_result = traci_module.simulation.findRoute(current_edge, destination)
            route_edges = _build_route_edges(route_result, current_edge)
            if not route_edges:
                continue

            traci_module.vehicle.setRoute(vehicle_id, route_edges)
            return destination
        except Exception:
            continue

    return None

