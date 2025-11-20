from copy import deepcopy
import random
import numpy as np
import time

from src.core.se3 import SE3
from src.core.so3 import SO3
from src.interface.robot_interface import RobotInterface
from src.core.obstacles import Obstacle


class PathFollowingPlanner:
    def __init__(self, robot_interface: RobotInterface, obstacle: Obstacle, waypoints: list[SE3], ik_func):
        self.robot_interface = robot_interface
        self.obstacle = obstacle
        self.waypoints = waypoints
        self.q_max = robot_interface.q_max
        self.q_min = robot_interface.q_min
        self.ik_func = ik_func
        self.Z_LIMIT = 0.01

    def get_list_of_best_q(self) -> np.ndarray:
        """
        Follows the path of waypoints, calculating the best inverse kinematics solution for each.
        This version uses an iterative refinement approach with an exhaustive initial search.

        """
        planning_start_time = time.time()
        print("Starting PathFollowingPlanner with iterative refinement...")

        if not self.waypoints:
            raise ValueError("No waypoints available.")

        # Get all IK solutions for each waypoint
        all_ik_solutions = []
        tangent_only_ik_solutions = []
        for waypoint_idx, waypoint in enumerate(self.waypoints):
            # Generate multiple orientations around the tangent axis
            ik_sols_all = []
            ik_sols_tangent_only = []

            if waypoint_idx == len(self.waypoints) - 1:
                # For the last waypoint, only rotate around tangent
                tilt_angles_deg = [0]
                roll_angles_deg = [0]
            else:
                # For all other waypoints, include tilt and roll
                tilt_angles_deg = [-45, -30, -15, -5, 0, 5, 15, 30, 45]  # Degrees of tilt around Y
                roll_angles_deg = [-45, -30, -15, -5, 0, 5, 15, 30, 45]  # Degrees of roll around X
            tilt_angles_rad = [np.radians(deg) for deg in tilt_angles_deg]
            roll_angles_rad = [np.radians(deg) for deg in roll_angles_deg]

            for tilt_angle in tilt_angles_rad:
                for roll_angle in roll_angles_rad:
                    # Create tilt (Y) and roll (X) rotations
                    tilt_rotation = SO3.from_angle_axis(tilt_angle, np.array([0, 1, 0]))
                    roll_rotation = SO3.from_angle_axis(roll_angle, np.array([1, 0, 0]))

                    # Apply tilt and roll to the base waypoint rotation
                    tilted_and_rolled_rotation = waypoint.rotation * tilt_rotation * roll_rotation

                    # --- Existing: Rotate around the tangent (local Z-axis) ---
                    num_rotations = 12
                    for angle in np.linspace(0, 2 * np.pi, num_rotations, endpoint=False):
                        # Rotate around z-axis (tangent direction)
                        rot_around_tangent = SO3.from_angle_axis(angle, np.array([0, 0, 1]))

                        # Apply tangent rotation to the already modified rotation
                        final_rotation = tilted_and_rolled_rotation * rot_around_tangent

                        modified_waypoint = SE3(translation=waypoint.translation, rotation=final_rotation)

                        ik_sols = np.asarray(self.ik_func(modified_waypoint))
                        if len(ik_sols) > 0:
                            ik_sols_mask = np.all(ik_sols < self.robot_interface.q_max, axis=1) & np.all(ik_sols > self.robot_interface.q_min, axis=1)
                            ik_sols = ik_sols[ik_sols_mask]
                        if len(ik_sols) > 0:
                            ik_sols_mask = [not self.obstacle.check_arm_colision(ik_sol)[0] for ik_sol in ik_sols]
                            ik_sols = ik_sols[ik_sols_mask]
                        if len(ik_sols) > 0:
                            if tilt_angle == 0 and roll_angle == 0:

                                ik_sols_tangent_only.append(ik_sols)
                            # print(
                            #    f"Waypoint {waypoint_idx}, tangent rot {np.degrees(angle):.1f}°, tilt {np.degrees(tilt_angle):.1f}°, roll {np.degrees(roll_angle):.1f}°: Found {len(ik_sols)} IK solutions."
                            # )
                            ik_sols_all.append(ik_sols)

            if len(ik_sols_all) == 0:
                raise ValueError(f"No IK solution found for waypoint {waypoint_idx} at {waypoint}")

            # --- Process and store all solutions (for refinement step) ---
            ik_sols_combined = np.vstack(ik_sols_all)
            # ik_sols_mask = np.all(ik_sols_combined < self.robot_interface.q_max, axis=1) & np.all(ik_sols_combined > self.robot_interface.q_min, axis=1)
            # ik_sols_filtered = ik_sols_combined[ik_sols_mask]
            ik_sols_filtered = ik_sols_combined
            if len(ik_sols_filtered) == 0:
                raise ValueError(f"No valid IK solution within joint limits for waypoint {waypoint_idx} at {waypoint}.")
            all_ik_solutions.append(ik_sols_filtered)

            # --- Process and store tangent-only solutions (for initial search) ---
            if len(ik_sols_tangent_only) > 0:
                if len(ik_sols_tangent_only) < 20 and len(ik_sols_filtered) > 20:
                    k = 20 - len(ik_sols_tangent_only)

                    ik_sols_tangent_only_extended = ik_sols_tangent_only.copy()

                    n = len(ik_sols_filtered)

                    # 1. Create k evenly spaced indices from 0 to n-1
                    # np.linspace(start, stop, num_items)
                    evenly_spaced_indices = np.linspace(0, n - 1, k, dtype=int)

                    # 2. Select the items at those indices
                    samples = [ik_sols_filtered[i] for i in evenly_spaced_indices]

                    ik_sols_tangent_only_extended.extend(samples)

                    ik_sols_tangent_combined = np.vstack(ik_sols_tangent_only_extended)
                else:
                    ik_sols_tangent_combined = np.vstack(ik_sols_tangent_only)

                # ik_sols_mask = np.all(ik_sols_tangent_combined < self.robot_interface.q_max, axis=1) & np.all(ik_sols_tangent_combined > self.robot_interface.q_min, axis=1)
                # tangent_only_ik_solutions.append(ik_sols_tangent_combined[ik_sols_mask])
                tangent_only_ik_solutions.append(ik_sols_tangent_combined)
            else:
                # Fallback to all solutions if no tangent-only solutions are found for this waypoint
                max_ik_sols = 35
                if len(ik_sols_filtered) > max_ik_sols:

                    k = max_ik_sols
                    n = len(ik_sols_filtered)

                    evenly_spaced_indices = np.linspace(0, n - 1, k, dtype=int)

                    # 2. Select the items at those indices
                    samples = [ik_sols_filtered[i] for i in evenly_spaced_indices]

                    ik_sols_picked = np.vstack(samples)
                    tangent_only_ik_solutions.append(ik_sols_picked)
                else:
                    tangent_only_ik_solutions.append(ik_sols_filtered)

        print("done generating ik solutions")

        # --- Iterative Refinement Planner with Exhaustive Initial Search ---

        def get_transition_cost(candidate_q, prev_q):
            """Calculates the transition cost from a previous configuration to a candidate."""
            T_pose = SE3().from_homogeneous(self.robot_interface.fk(candidate_q))
            # if T_pose.translation[2] < self.Z_LIMIT:
            #    return np.inf  # Invalid configuration

            hoop_pose = self.robot_interface.hoop_fk(candidate_q)
            hoop_x_axis = hoop_pose.rotation.rot[:, 0]
            me_vector = np.array([1, 0, -3])
            dot_prod = np.dot(hoop_x_axis, me_vector)
            collision, dist, hoop_dist = self.obstacle.check_arm_colision(candidate_q)
            # collision_in_path_cost = 1000 if self.obstacle.is_path_viable(prev_q, candidate_q) else 0

            cost = (
                2 * np.linalg.norm(candidate_q[:] - prev_q[:]) ** 2
                + 30 * np.linalg.norm(candidate_q[-3:] - prev_q[-3:]) ** 2
                + 1 * np.sum(np.maximum(0, self.Z_LIMIT * 2 - T_pose.translation[2]))
                + 0.1 * np.linalg.norm(candidate_q[-2:] - (self.robot_interface.q_max[-2:] + self.robot_interface.q_min[-2:]) / 2)
                + 5 * (-dot_prod)
                + 5 * (-dist)
            )
            return cost if cost < 50 else cost + 200

        # 1. Initialization: Exhaustively find the best coarse path
        num_initial_points = 6
        initial_indices = np.linspace(0, len(self.waypoints) - 1, num_initial_points, dtype=int)

        # Use only tangent-rotated solutions for the coarse search
        coarse_ik_solutions = [tangent_only_ik_solutions[i] for i in initial_indices]

        for waypoint_idx, sols in enumerate(all_ik_solutions):
            print(f"All search waypoint {waypoint_idx} has {len(sols)} IK solutions.")

        for waypoint_idx, sols in zip(initial_indices, coarse_ik_solutions):
            print(f"Initial coarse search waypoint {waypoint_idx} has {len(sols)} IK solutions.")

        min_total_cost = np.inf
        best_q_path = []

        start_time = time.time()

        # This recursive function will perform a depth-first search through all combinations.
        def find_best_path_recursive(waypoint_level, current_path, current_cost):
            nonlocal min_total_cost, best_q_path

            # If we have a complete path, check if it's the best one so far
            if waypoint_level == len(coarse_ik_solutions):
                if current_cost < min_total_cost:
                    min_total_cost = current_cost
                    best_q_path = list(current_path)  # Make a copy
                return

            # Pruning: If the current path is already more expensive than the best found so far, stop.
            if current_cost * 1.5 >= min_total_cost:
                return
            if start_time + 30 < time.time():
                print("Exhaustive search timeout reached.")
                return

            prev_q = current_path[-1]
            # Explore all solutions for the current waypoint level
            for candidate_q in coarse_ik_solutions[waypoint_level]:
                transition_cost = get_transition_cost(candidate_q, prev_q)
                if transition_cost == np.inf:
                    continue

                find_best_path_recursive(waypoint_level + 1, current_path + [candidate_q], current_cost + transition_cost)

        # Start the recursive search for each possible starting configuration
        print("Starting exhaustive search for the best initial path...")
        for q_start in coarse_ik_solutions[0]:
            # The "cost" to start is 0, as there's no transition.
            find_best_path_recursive(1, [q_start], 0.0)

        if not best_q_path:
            raise ValueError("Exhaustive search failed to find any valid initial path.")

        print(f"Optimal coarse path found with cost: {min_total_cost}")
        q_path = best_q_path
        path_indices = list(initial_indices)

        # 2. Iterative Refinement Loop
        while len(q_path) < len(self.waypoints):
            new_indices_to_add = []
            for i in range(len(path_indices) - 1):
                mid_index = (path_indices[i] + path_indices[i + 1]) // 2
                if mid_index > path_indices[i] and mid_index < path_indices[i + 1]:
                    new_indices_to_add.append((i + 1, mid_index))

            if not new_indices_to_add:
                break  # No more points to add

            new_indices_to_add.sort(key=lambda x: x[0], reverse=True)

            for insert_pos, waypoint_idx in new_indices_to_add:
                prev_q = q_path[insert_pos - 1]
                next_q = q_path[insert_pos]
                candidate_qs = all_ik_solutions[waypoint_idx]

                min_total_segment_cost = np.inf
                best_q_for_midpoint = None
                for q_mid in candidate_qs:
                    cost1 = get_transition_cost(q_mid, prev_q)
                    cost2 = get_transition_cost(next_q, q_mid)
                    # cost2 = 0
                    if cost1 == np.inf or cost2 == np.inf:
                        continue
                    total_segment_cost = cost1 + cost2
                    if total_segment_cost < min_total_segment_cost:
                        min_total_segment_cost = total_segment_cost
                        best_q_for_midpoint = q_mid

                if best_q_for_midpoint is None:
                    # Fallback if no valid mid-point is found
                    costs = [get_transition_cost(q, prev_q) for q in candidate_qs]
                    best_q_for_midpoint = candidate_qs[np.argmin(costs)]

                q_path.insert(insert_pos, best_q_for_midpoint)
                path_indices.insert(insert_pos, waypoint_idx)

            print(f"Refined path. New length: {len(q_path)}")

        # 3. Path Smoothing

        before_smoothing_cost = sum(get_transition_cost(q_path[i], q_path[i - 1]) if i > 0 else 0 for i in range(len(q_path)))
        before_smoothing_path = deepcopy(q_path)

        num_smoothing_iterations = 3
        print("Starting path smoothing...")
        for iter_num in range(num_smoothing_iterations):
            # Iterate backwards through the path, from the last point to the first
            for i in range(len(q_path) - 1, -1, -1):
                if planning_start_time + 55 < time.time():
                    print("Smoothing timeout reached.")
                    break

                waypoint_idx = path_indices[i]
                candidate_qs = all_ik_solutions[waypoint_idx]
                min_total_segment_cost = np.inf
                best_q_for_point = q_path[i]  # Keep original if no better is found

                if i == 0:
                    # First point: only consider cost to the next point
                    next_q = q_path[i + 1]
                    for q_mid in candidate_qs:
                        # The cost function for the start has no "prev_q"
                        cost = get_transition_cost(next_q, q_mid)
                        if cost < min_total_segment_cost:
                            min_total_segment_cost = cost
                            best_q_for_point = q_mid
                elif i == len(q_path) - 1:
                    # Last point: only consider cost from the previous point
                    prev_q = q_path[i - 1]
                    for q_mid in candidate_qs:
                        cost = get_transition_cost(q_mid, prev_q)
                        if cost < min_total_segment_cost:
                            min_total_segment_cost = cost
                            best_q_for_point = q_mid
                else:
                    # Intermediate point: consider cost from previous and to next
                    prev_q = q_path[i - 1]
                    next_q = q_path[i + 1]
                    for q_mid in candidate_qs:
                        cost1 = get_transition_cost(q_mid, prev_q)
                        cost2 = get_transition_cost(next_q, q_mid)

                        if cost1 == np.inf or cost2 == np.inf:
                            continue
                        # prefer cost 2
                        total_segment_cost = cost1 * 0.9 + cost2
                        if total_segment_cost < min_total_segment_cost:
                            min_total_segment_cost = total_segment_cost
                            best_q_for_point = q_mid

                # Update the path with the best configuration found for this point
                q_path[i] = best_q_for_point
            print(f"  -> Smoothing iteration {iter_num + 1}/{num_smoothing_iterations} complete.")

        for i in range(len(q_path)):
            print(f"transition cost {i}: {get_transition_cost(q_path[i], q_path[i - 1]) if i > 0 else 0}")

        if before_smoothing_cost < sum(get_transition_cost(q_path[i], q_path[i - 1]) if i > 0 else 0 for i in range(len(q_path))):
            q_path = before_smoothing_path
            print(f"Smoothing increased cost; reverting to pre-smoothing path with cost {before_smoothing_cost}.")

        print(f"Final path found with {len(q_path)} points after smoothing.")
        planner_end_time = time.time()
        print("PathFollowingPlanner took total time:", planner_end_time - planning_start_time, "seconds")
        return np.array(q_path)

    def _is_within_limits(self, q: np.ndarray) -> bool:
        """Check if configuration is within joint limits.

        Args:
            q (np.ndarray): Joint configuration to check.

        Returns:
            bool: True if within limits, False otherwise.
        """
        for i in range(len(q)):
            if q[i] < self.q_min[i] or q[i] > self.q_max[i]:
                return False
        return True
