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
from itertools import combinations
from typing import Dict, List, Optional, Tuple

from ar_hackathon.models.drive_unit import DriveUnit
from ar_hackathon.models.graph_state import GraphState
from ar_hackathon.models.pod import Pod
from ar_hackathon.utils.routing_utils import is_valid_move


# Assignments persist between calls so two robots do not deliberately chase
# the same waiting pod. The engine calls this module once per idle robot.
_pod_assignments: Dict[str, int] = {}
_last_time_step = -1

# Keep the exhaustive stop-order search small enough for the one-second limit.
_MAX_PLANNED_PODS = 4
_MAX_PICKUP_CANDIDATES = 6


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


def _nearest_route(
    state: GraphState,
    start_node: int,
    targets: List[int],
) -> Optional[List[int]]:
    """Return the cheapest route from start_node to any supplied target."""
    choices = []
    for target in sorted(set(targets)):
        route = shortest_path(state, start_node, target)
        if route is not None:
            choices.append((_route_cost(state, route), tuple(route), route))
    return min(choices)[2] if choices else None


def _refresh_assignments(state: GraphState) -> None:
    """Remove stale pod claims and reset claims for a new simulation."""
    global _last_time_step

    if state.current_time_step < _last_time_step:
        _pod_assignments.clear()
    _last_time_step = state.current_time_step

    active_ids = {pod.id for pod in state.active_pods}
    valid_unit_ids = {unit.id for unit in state.drive_units}
    for pod_id, unit_id in list(_pod_assignments.items()):
        if pod_id not in active_ids or unit_id not in valid_unit_ids:
            del _pod_assignments[pod_id]

    # The physical state is authoritative if automatic pickup assigned a pod.
    for pod in state.active_pods:
        if pod.carried_by is not None:
            _pod_assignments[pod.id] = pod.carried_by


def _plan_stop_sequence(
    state: GraphState,
    start_node: int,
    capacity: int,
    carried_ids: List[str],
    waiting_pods: List[Pod],
) -> Optional[Tuple[float, float, List[int]]]:
    """Find the cheapest valid pickup/delivery stop order for one robot.

    Pickups must occur before their delivery and the simulated load may never
    exceed the robot's capacity. Pods at the same source or destination are
    handled together, matching the engine's automatic pickup and delivery.
    """
    pod_by_id = {}
    for pod_id in carried_ids:
        pod = state.get_pod(pod_id)
        if pod is not None:
            pod_by_id[pod.id] = pod
    for pod in waiting_pods:
        pod_by_id[pod.id] = pod

    initial_carried = tuple(sorted(pod_id for pod_id in carried_ids if pod_id in pod_by_id))
    initial_unpicked = tuple(sorted(pod.id for pod in waiting_pods))
    leg_cache: Dict[Tuple[int, int], float] = {}
    memo: Dict[
        Tuple[int, Tuple[str, ...], Tuple[str, ...]],
        Optional[Tuple[float, float, Tuple[int, ...]]],
    ] = {}

    def leg_cost(from_node: int, to_node: int) -> float:
        key = (from_node, to_node)
        if key not in leg_cache:
            route = shortest_path(state, from_node, to_node)
            leg_cache[key] = float("inf") if route is None else _route_cost(state, route)
        return leg_cache[key]

    def solve(
        current_node: int,
        carried: Tuple[str, ...],
        unpicked: Tuple[str, ...],
    ) -> Optional[Tuple[float, float, Tuple[int, ...]]]:
        key = (current_node, carried, unpicked)
        if key in memo:
            return memo[key]
        if not carried and not unpicked:
            return 0.0, 0.0, ()

        choices: List[Tuple[float, float, Tuple[int, ...]]] = []
        undelivered_count = len(carried) + len(unpicked)

        if len(carried) < capacity:
            pickup_nodes = sorted(
                {
                    pod_by_id[pod_id].current_node
                    for pod_id in unpicked
                    if pod_by_id[pod_id].current_node is not None
                }
            )
            for pickup_node in pickup_nodes:
                slots = capacity - len(carried)
                pods_here = sorted(
                    (
                        pod_id
                        for pod_id in unpicked
                        if pod_by_id[pod_id].current_node == pickup_node
                    ),
                    key=lambda pod_id: (
                        pod_by_id[pod_id].entry_time,
                        pod_id,
                    ),
                )[:slots]
                if not pods_here:
                    continue

                distance = leg_cost(current_node, pickup_node)
                if distance == float("inf"):
                    continue
                picked = set(pods_here)
                next_carried = tuple(sorted(set(carried).union(picked)))
                next_unpicked = tuple(pod_id for pod_id in unpicked if pod_id not in picked)
                remainder = solve(pickup_node, next_carried, next_unpicked)
                if remainder is not None:
                    remaining_latency, remaining_cost, remaining_stops = remainder
                    choices.append(
                        (
                            distance * undelivered_count + remaining_latency,
                            distance + remaining_cost,
                            (pickup_node,) + remaining_stops,
                        )
                    )

        delivery_nodes = sorted(
            {pod_by_id[pod_id].destination_station for pod_id in carried}
        )
        for delivery_node in delivery_nodes:
            distance = leg_cost(current_node, delivery_node)
            if distance == float("inf"):
                continue
            next_carried = tuple(
                pod_id
                for pod_id in carried
                if pod_by_id[pod_id].destination_station != delivery_node
            )
            remainder = solve(delivery_node, next_carried, unpicked)
            if remainder is not None:
                remaining_latency, remaining_cost, remaining_stops = remainder
                choices.append(
                    (
                        distance * undelivered_count + remaining_latency,
                        distance + remaining_cost,
                        (delivery_node,) + remaining_stops,
                    )
                )

        result = min(choices, key=lambda choice: (choice[0], choice[1], choice[2])) if choices else None
        memo[key] = result
        return result

    result = solve(start_node, initial_carried, initial_unpicked)
    if result is None:
        return None
    latency, cost, stops = result
    return latency, cost, list(stops)


def _choose_trip_plan(state: GraphState, unit: DriveUnit) -> List[int]:
    """Select pods for this robot and return its optimized sequence of stops."""
    owned_waiting = [
        pod
        for pod in state.active_pods
        if pod.carried_by is None
        and pod.current_node is not None
        and _pod_assignments.get(pod.id) == unit.id
    ]
    unassigned = [
        pod
        for pod in state.active_pods
        if pod.carried_by is None
        and pod.current_node is not None
        and pod.id not in _pod_assignments
    ]

    room_in_trip = max(
        0,
        min(
            unit.capacity - len(unit.carrying) - len(owned_waiting),
            _MAX_PLANNED_PODS - len(unit.carrying) - len(owned_waiting),
        ),
    )

    ranked_candidates = []
    for pod in unassigned:
        route = shortest_path(state, unit.current_node, pod.current_node)
        if route is not None:
            ranked_candidates.append(
                (_route_cost(state, route), pod.entry_time, pod.id, pod)
            )
    ranked_candidates.sort(key=lambda item: item[:3])
    candidate_pool = [
        item[3] for item in ranked_candidates[:_MAX_PICKUP_CANDIDATES]
    ]

    if unit.carrying:
        # Preserve parallelism: a loaded robot only claims extra work that
        # cannot fit on the other robots which are currently free to collect.
        helper_capacity = 0
        for other in state.drive_units:
            if other.id == unit.id or other.carrying:
                continue
            existing_claims = sum(
                1 for owner in _pod_assignments.values() if owner == other.id
            )
            helper_capacity += max(0, other.capacity - existing_claims)
        desired_new_pods = max(0, len(unassigned) - helper_capacity)
    else:
        # Divide new work across robots that do not already have a pickup
        # assignment instead of letting the first robot claim every pod.
        available_collectors = 0
        for other in state.drive_units:
            if other.carrying:
                continue
            has_claim = any(owner == other.id for owner in _pod_assignments.values())
            if not has_claim:
                available_collectors += 1
        available_collectors = max(1, available_collectors)
        desired_new_pods = (
            len(unassigned) + available_collectors - 1
        ) // available_collectors

    number_to_add = min(room_in_trip, len(candidate_pool), desired_new_pods)
    candidate_groups = (
        combinations(candidate_pool, number_to_add)
        if number_to_add > 0
        else [()]
    )

    best_choice = None
    for group in candidate_groups:
        selected_waiting = owned_waiting + list(group)
        plan = _plan_stop_sequence(
            state,
            unit.current_node,
            unit.capacity,
            list(unit.carrying),
            selected_waiting,
        )
        if plan is None:
            continue
        latency, cost, stops = plan
        choice_key = (
            latency,
            cost,
            tuple(stops),
            tuple(sorted(pod.id for pod in group)),
        )
        if best_choice is None or choice_key < best_choice[0]:
            best_choice = (choice_key, stops, list(group))

    if best_choice is None:
        return []

    _, stops, newly_selected = best_choice
    for pod in newly_selected:
        _pod_assignments[pod.id] = unit.id
    return stops


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


def _vacate_station(state: GraphState, unit: DriveUnit) -> Optional[int]:
    """Move an idle robot out of a station so it does not occupy a dock."""
    current = state.get_node(unit.current_node)
    if (
        current is None
        or current.node_type != "station"
        or current.capacity is None
    ):
        return None

    exits = []
    for neighbor in state.neighbors(unit.current_node):
        if not is_valid_move(state, unit, neighbor):
            continue
        edge = state.get_edge(unit.current_node, neighbor)
        node = state.get_node(neighbor)
        if edge is not None:
            # Prefer parking/travel nodes over another station.
            station_penalty = 1 if node is not None and node.node_type == "station" else 0
            exits.append((station_penalty, edge.weight, neighbor))

    return min(exits)[2] if exits else None


def _next_move_toward(state: GraphState, unit: DriveUnit, target: int) -> Optional[int]:
    """Take the shortest clear step, or a worthwhile traffic detour."""
    route = shortest_path(state, unit.current_node, target)
    if route is None or len(route) < 2:
        return None

    next_node = route[1]
    if is_valid_move(state, unit, next_node):
        return next_node

    alternate_route = shortest_path(
        state,
        unit.current_node,
        target,
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

    # Capacity-one robots have no batching decision to make. Preserve the
    # simple nearest-pickup/nearest-delivery behavior used by Levels 1 and 2.
    if unit.capacity <= 1:
        if unit.carrying:
            targets = []
            for pod_id in unit.carrying:
                pod = state.get_pod(pod_id)
                if pod is not None:
                    targets.append(pod.destination_station)
        else:
            targets = [
                pod.current_node
                for pod in state.active_pods
                if pod.carried_by is None and pod.current_node is not None
            ]

        route = _nearest_route(state, unit.current_node, targets)
        if route is not None and len(route) >= 2:
            return _next_move_toward(state, unit, route[-1])
        return _vacate_station(state, unit)

    _refresh_assignments(state)
    planned_stops = _choose_trip_plan(state, unit)
    if planned_stops:
        return _next_move_toward(state, unit, planned_stops[0])

    # A robot with no current work should never permanently occupy a dock.
    return _vacate_station(state, unit)
