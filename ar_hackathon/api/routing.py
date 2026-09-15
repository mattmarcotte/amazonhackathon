"""
Amazon Robotics Hackathon - Routing API

This module defines the routing API for the Amazon Robotics Hackathon.
Students will implement the drive_unit_next_move function in this module.

*****IMPORTANT*****
Team name:
Email address:
*******************
"""

import heapq
from typing import Dict, List, Optional, Tuple

from ar_hackathon.models.graph_state import GraphState
from ar_hackathon.utils.routing_utils import is_valid_move


def shortest_path(
    state: GraphState,
    start_node: int,
    destination_node: int,
    *,
    avoid_traffic: bool = False,
) -> Optional[List[int]]:
    """Return the minimum-travel-time route between two nodes.

    The returned list includes both ``start_node`` and ``destination_node``.
    ``None`` is returned when either node does not exist or no route connects
    them. Edge weights are used as travel times, so this works for weighted
    graphs as well as one-way and bidirectional aisles. When
    ``avoid_traffic`` is true, edges and destination nodes that are currently
    at capacity are treated as unavailable so the result can be used as a
    traffic-aware alternate route.
    """
    if state.get_node(start_node) is None:
        return None
    if state.get_node(destination_node) is None:
        return None
    if start_node == destination_node:
        return [start_node]

    distances: Dict[int, float] = {start_node: 0.0}
    previous: Dict[int, int] = {}
    frontier: List[Tuple[float, int]] = [(0.0, start_node)]

    while frontier:
        current_distance, current_node = heapq.heappop(frontier)

        # Ignore an older queue entry if a shorter route was found later.
        if current_distance != distances.get(current_node):
            continue
        if current_node == destination_node:
            break

        for neighbor in state.neighbors(current_node):
            edge = state.get_edge(current_node, neighbor)
            if edge is None:
                continue

            if avoid_traffic:
                if (
                    edge.capacity is not None
                    and state.edge_occupancy(current_node, neighbor) >= edge.capacity
                ):
                    continue

                neighbor_node = state.get_node(neighbor)
                if (
                    neighbor_node is not None
                    and neighbor_node.capacity is not None
                    and state.node_occupancy(neighbor) >= neighbor_node.capacity
                ):
                    continue

            candidate_distance = current_distance + edge.weight
            if candidate_distance < distances.get(neighbor, float("inf")):
                distances[neighbor] = candidate_distance
                previous[neighbor] = current_node
                heapq.heappush(frontier, (candidate_distance, neighbor))

    if destination_node not in distances:
        return None

    route = [destination_node]
    while route[-1] != start_node:
        route.append(previous[route[-1]])
    route.reverse()
    return route


def _route_cost(state: GraphState, route: List[int]) -> float:
    """Calculate a route's total edge weight."""
    total = 0.0
    for from_node, to_node in zip(route, route[1:]):
        edge = state.get_edge(from_node, to_node)
        if edge is None:
            return float("inf")
        total += edge.weight
    return total


def _best_route(state: GraphState, start_node: int, targets: List[int]) -> Optional[List[int]]:
    """Find the cheapest reachable route from start_node to any target."""
    best_route = None
    best_cost = float("inf")

    # Sorting makes equally good choices deterministic across runs.
    for target in sorted(set(targets)):
        route = shortest_path(state, start_node, target)
        if route is None:
            continue
        cost = _route_cost(state, route)
        if cost < best_cost:
            best_cost = cost
            best_route = route

    return best_route


def _estimated_wait_for_move(state: GraphState, from_node: int, to_node: int) -> int:
    """Estimate how many time steps the blocked next move may need to clear."""
    estimated_wait = 1
    edge = state.get_edge(from_node, to_node)

    if edge is not None and edge.capacity is not None:
        edge_remaining_times = [
            unit.transit_remaining_time
            for unit in state.drive_units
            if unit.in_transit
            and edge.connects(unit.current_node, unit.transit_destination)
        ]
        if len(edge_remaining_times) >= edge.capacity:
            estimated_wait = max(estimated_wait, min(edge_remaining_times))

    node = state.get_node(to_node)
    if node is not None and node.capacity is not None:
        node_release_times = []
        for unit in state.drive_units:
            if unit.in_transit and unit.transit_destination == to_node:
                node_release_times.append(max(1, unit.transit_remaining_time))
            elif not unit.in_transit and unit.current_node == to_node:
                # An idle unit might leave on this step, but the caller cannot
                # use the released slot until its next routing decision.
                node_release_times.append(1)

        if len(node_release_times) >= node.capacity:
            estimated_wait = max(estimated_wait, min(node_release_times))

    return estimated_wait


def drive_unit_next_move(drive_unit_id: int, state: GraphState) -> Optional[int]:
    """
    Determine the next node for a drive unit to move to.

    This is the function that students will implement. The game engine will
    call this function for each idle drive unit at each time step to
    determine where it should go next.

    Pickups and deliveries are automatic: a drive unit with free capacity
    that stops at (or passes through) a node with a waiting pod picks it up,
    and a drive unit that reaches a carried pod's destination station drops
    it off.

    Args:
        drive_unit_id: ID of the drive unit being routed
        state: GraphState object containing the current state of the floor

    Returns:
        next_node_id: ID of an adjacent node to move to, or None to wait
                      at the current node
    """
    unit = state.get_drive_unit(drive_unit_id)
    if unit is None or unit.in_transit:
        return None

    if unit.carrying:
        # With multi-pod capacity, deliver whichever carried pod has the
        # cheapest destination to reach from the unit's current position.
        targets = []
        for pod_id in unit.carrying:
            pod = state.get_pod(pod_id)
            if pod is not None:
                targets.append(pod.destination_station)
    else:
        # An empty unit heads toward the closest waiting, unclaimed pod.
        targets = [
            pod.current_node
            for pod in state.active_pods
            if pod.carried_by is None and pod.current_node is not None
        ]

    route = _best_route(state, unit.current_node, targets)
    if route is None or len(route) < 2:
        return None

    next_node = route[1]
    if is_valid_move(state, unit, next_node):
        return next_node

    # The preferred route is blocked. Look for a route to the same target
    # that avoids every edge and node currently at capacity.
    alternate_route = shortest_path(
        state,
        unit.current_node,
        route[-1],
        avoid_traffic=True,
    )
    if alternate_route is None or len(alternate_route) < 2:
        return None

    alternate_next_node = alternate_route[1]
    if not is_valid_move(state, unit, alternate_next_node):
        return None

    wait_cost = _route_cost(state, route) + _estimated_wait_for_move(
        state,
        unit.current_node,
        next_node,
    )
    alternate_cost = _route_cost(state, alternate_route)
    if alternate_cost <= wait_cost:
        return alternate_next_node

    return None
