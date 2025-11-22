import numpy as np
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from src.core.se3 import SE3
from src.core.so3 import SO3
from src.interface.robot_interface import RobotInterface
from src.core.obstacles import Obstacle
from copy import deepcopy


# Global lock for thread-safe printing
print_lock = Lock()
Z_LIMIT = 0.02


class Q:
    def __init__(self, q_array):
        self.q_array: np.ndarray = q_array
        self.distance: float = 0.0
        self.hoop_distance: float = 0.0
        self.dot_product: float = 0.0
        self.tangent_dot_product: float = 0.0
        self.T_pose: SE3 | None = None
        self.hoop_pose: SE3 | None = None
        self.height_penalty: float = 0.0
        self.q_penalty: float = 0.0


def _process_waypoint_ik(waypoint_idx, waypoint: SE3, total_waypoints, q_max, q_min, ik_func, obstacle: Obstacle, robot_interface: RobotInterface) -> list[Q]:
    """Worker function to process IK solutions for a single waypoint."""
    # Generate multiple orientations around the tangent axis
    ik_sols_all = []

    if waypoint_idx == total_waypoints - 1:
        # For the last waypoint, only rotate around tangent
        tilt_angles_deg = [0]
        roll_angles_deg = [0]
    elif waypoint_idx == total_waypoints - 2:
        # For the last waypoint, only rotate around tangent
        tilt_angles_deg = [-35, -15, -5, 0, 5, 15, 35]
        roll_angles_deg = [-35, -15, -5, 0, 5, 15, 35]
    else:
        # For all other waypoints, include tilt and roll
        tilt_angles_deg = [-35, -15, -5, 0, 5, 15, 35]  # Degrees of tilt around Y
        roll_angles_deg = [-35, -15, -5, 0, 5, 15, 35]  # Degrees of roll around X
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

                ik_sols = np.asarray(ik_func(modified_waypoint))
                if len(ik_sols) > 0:
                    ik_sols_mask = np.all(ik_sols < q_max, axis=1) & np.all(ik_sols > q_min, axis=1)
                    ik_sols = ik_sols[ik_sols_mask]
                if len(ik_sols) > 0:
                    ik_sols_mask = [not obstacle.check_arm_colision(ik_sol)[0] for ik_sol in ik_sols]
                    ik_sols = ik_sols[ik_sols_mask]
                if len(ik_sols) > 0:
                    ik_sols_all.append(ik_sols)

    if len(ik_sols_all) == 0:
        raise ValueError(f"No IK solution found for waypoint {waypoint_idx} at {waypoint}")
    ik_sols_combined = np.vstack(ik_sols_all)
    ik_sols_filtered = ik_sols_combined
    if len(ik_sols_filtered) == 0:
        raise ValueError(f"No valid IK solution within joint limits for waypoint {waypoint_idx} at {waypoint}.")

    # Convert to Q objects with pre-calculated costs
    q_objects = []
    for q_array in ik_sols_filtered:
        q_obj = Q(q_array)

        # Pre-calculate all costs
        q_obj.T_pose = SE3().from_homogeneous(robot_interface.fk(q_array))
        q_obj.hoop_pose = robot_interface.hoop_fk(q_array)

        # Type guard assertions since we just set these values
        assert q_obj.hoop_pose is not None
        assert q_obj.T_pose is not None

        hoop_x_axis = q_obj.hoop_pose.rotation.rot[:, 0]
        me_vector = np.array([0, 0, -1])
        q_obj.dot_product = float(np.dot(hoop_x_axis, me_vector))
        collision, dist, circle_dist = obstacle.check_arm_colision(q_array)
        q_obj.distance = dist
        q_obj.hoop_distance = circle_dist
        q_obj.tangent_dot_product = float(np.dot(waypoint.rotation.rot[:, 2], q_obj.hoop_pose.rotation.rot[:, 2]))
        q_obj.height_penalty = float(np.sum(np.maximum(0, Z_LIMIT * 2 - q_obj.T_pose.translation[2])))
        q_obj.q_penalty = float(np.linalg.norm(q_obj.q_array[-2:] - (robot_interface.q_max[-2:] + robot_interface.q_min[-2:]) / 2))

        q_objects.append(q_obj)

    with print_lock:
        print(f"Processed waypoint {waypoint_idx}/{total_waypoints-1}")
    return q_objects


class PathFollowingPlanner:
    def __init__(self, robot_interface: RobotInterface, obstacle: Obstacle, waypoints: list[SE3], ik_func):
        self.robot_interface = robot_interface
        self.obstacle = obstacle
        self.waypoints = waypoints
        self.q_max = robot_interface.q_max
        self.q_min = robot_interface.q_min
        self.q_max[0] = np.pi / 3
        self.q_min[0] = -np.pi / 3
        self.ik_func = ik_func
        self.Z_LIMIT = Z_LIMIT

    def get_all_ik_solutions(self) -> list[list[Q]]:
        # Process waypoints in parallel using threads
        max_workers = min(len(self.waypoints), max(1, (os.cpu_count() or 1)))
        print(f"Generating IK solutions using {max_workers} threads...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for waypoint_idx, waypoint in enumerate(self.waypoints):
                future = executor.submit(
                    _process_waypoint_ik, waypoint_idx, waypoint, len(self.waypoints), self.robot_interface.q_max, self.robot_interface.q_min, self.ik_func, self.obstacle, self.robot_interface
                )
                futures.append(future)

            # Collect results in order
            all_ik_solutions = [future.result() for future in futures]

        print("done generating ik solutions")
        for waypoint_idx, sols in enumerate(all_ik_solutions):
            print(f"All search waypoint {waypoint_idx} has {len(sols)} IK solutions.")
        return all_ik_solutions

    def get_transition_cost(self, waypoint, candidate_q: Q, prev_q: Q) -> float:
        """Calculates the transition cost from a previous configuration to a candidate."""
        # Use pre-calculated values from Q object
        prev_q_array = prev_q.q_array

        # Type guard assertion since Q objects should have these values set
        assert candidate_q.T_pose is not None

        # important to tune dot prod weight
        cost = (
            2 * np.linalg.norm(candidate_q.q_array[:] - prev_q_array[:]) ** 2
            + 30 * np.linalg.norm(candidate_q.q_array[-3:] - prev_q_array[-3:]) ** 2
            + 1 * candidate_q.height_penalty
            + 0.1 * candidate_q.q_penalty
            + 15 * (-candidate_q.dot_product)
            + 20 * (-candidate_q.distance)
            + 70 * (1 - candidate_q.tangent_dot_product)
            + 10 * (1 if candidate_q.hoop_distance < 0.01 else 0)
            + 20
        )
        return float(cost) if cost < 70 else float(cost) + 200  ## neeeds to change constant for penalty if weights adjusted!!!

    def backward_greedy_search(self, all_ik_solutions):
        # Shared state across all threads
        shared_state = {"min_total_cost": np.inf, "best_q_path": [], "lock": Lock(), "start_time": time.time()}

        def search_from_end_point(q_end):
            """Search for best path starting from a specific end configuration."""

            def find_best_path_recursive(waypoint_level: int, current_path: list[Q], current_cost: float):
                # Check shared minimum cost for early termination
                with shared_state["lock"]:
                    current_min = shared_state["min_total_cost"]

                if waypoint_level == -1:
                    # Found a complete path - update shared state if better
                    with shared_state["lock"]:
                        if current_cost < shared_state["min_total_cost"]:
                            shared_state["min_total_cost"] = current_cost
                            shared_state["best_q_path"] = list(current_path)  # Make a copy
                            with print_lock:
                                print(f"  -> New best path with cost: {shared_state['min_total_cost']:.2f}")
                    return

                # Early termination based on shared minimum
                if current_cost * 1.01 >= current_min:
                    return

                if shared_state["start_time"] + 55 < time.time() and shared_state["best_q_path"]:
                    with print_lock:
                        print("Thread timeout reached.")
                    return  # Timeout

                next_q = current_path[-1]
                sorted_candidates = sorted(all_ik_solutions[waypoint_level], key=lambda q: self.get_transition_cost(self.waypoints[waypoint_level], q, next_q))

                # Try candidates in order of cost until we find a viable path
                found_viable = False
                for candidate_q in sorted_candidates:
                    if self.obstacle.check_path_viable(q_start=next_q.q_array, q_end=candidate_q.q_array):
                        transition_cost = self.get_transition_cost(self.waypoints[waypoint_level], candidate_q, next_q)
                        find_best_path_recursive(waypoint_level - 1, current_path + [candidate_q], current_cost + transition_cost)
                        found_viable = True
                        break  # Only take the first (cheapest) viable candidate

                # Also try a candidate from one-quarter of the way through the sorted list for more exploration
                """
                if found_viable and len(sorted_candidates) > 7:
                    one_quarter_idx = len(sorted_candidates) // 4
                    for i in range(one_quarter_idx, len(sorted_candidates)):
                        candidate_q = sorted_candidates[i]
                        if self.obstacle.check_path_viable(q_start=next_q.q_array, q_end=candidate_q.q_array):
                            transition_cost = self.get_transition_cost(self.waypoints[waypoint_level], candidate_q, next_q)
                            find_best_path_recursive(waypoint_level - 1, current_path + [candidate_q], current_cost + transition_cost)
                            break  # Only take the first viable candidate from this section
                """

                if not found_viable:
                    # No viable path found from this waypoint, terminate this branch
                    with print_lock:
                        print(f"Path from waypoint {waypoint_level} candidate rejected due to collision. Terminating branch.")
                    return

            # Start recursive search from this end point
            print(f"Starting search from end configuration[-1]: {q_end.q_array[-1]:.2f}")
            find_best_path_recursive(len(all_ik_solutions) - 2, [q_end], 0.0)

        print("Starting parallel exhaustive search for the best initial path...")

        # Process each end configuration in parallel
        max_workers = min(len(all_ik_solutions[-1]), max(1, (os.cpu_count() or 1)))
        print(f"Searching using {max_workers} threads...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(search_from_end_point, q_end) for q_end in all_ik_solutions[-1]]

            # Wait for all threads to complete
            for future in as_completed(futures):
                try:
                    future.result()  # This will raise any exceptions that occurred
                except Exception as e:
                    print(f"Thread encountered error: {e}")

        if not shared_state["best_q_path"]:
            raise ValueError("Parallel exhaustive search failed to find any valid initial path.")

        print(f"Optimal coarse path found with cost: {shared_state['min_total_cost']:.2f}")
        return shared_state["best_q_path"][::-1]

    def smooth_path(
        self,
        q_path: list[Q],
        all_ik_solutions,
        planner_start_time,
        num_smoothing_iterations: int = 1,
    ) -> list[Q]:
        print("Starting path smoothing...")
        for iter_num in range(num_smoothing_iterations):
            # Iterate backwards through the path, from the last point to the first
            for i in range(len(q_path) - 1, 0, -1):

                if planner_start_time + 55 < time.time():
                    print("Smoothing timeout reached.")
                    break

                candidate_qs = all_ik_solutions[i]
                min_total_segment_cost = np.inf
                best_q_for_point = q_path[i]

                if i == 0:
                    # First point
                    next_q = q_path[i + 1]
                    for q_mid in candidate_qs:
                        cost = self.get_transition_cost(self.waypoints[i + 1], next_q, q_mid)
                        if cost < min_total_segment_cost:
                            if self.obstacle.check_path_viable(q_start=q_mid.q_array, q_end=next_q.q_array):
                                min_total_segment_cost = cost
                                best_q_for_point = q_mid
                elif i == len(q_path) - 1:
                    # Last point
                    prev_q = q_path[i - 1]
                    for q_mid in candidate_qs:
                        cost = self.get_transition_cost(self.waypoints[i], q_mid, prev_q)
                        if cost < min_total_segment_cost:
                            if self.obstacle.check_path_viable(q_start=prev_q.q_array, q_end=q_mid.q_array):
                                min_total_segment_cost = cost
                                best_q_for_point = q_mid
                else:

                    prev_q = q_path[i - 1]
                    next_q = q_path[i + 1]
                    for q_mid in candidate_qs:
                        cost1 = self.get_transition_cost(self.waypoints[i], q_mid, prev_q)
                        cost2 = self.get_transition_cost(self.waypoints[i + 1], next_q, q_mid)

                        if cost1 == np.inf or cost2 == np.inf:
                            continue
                        total_segment_cost = cost1 + cost2
                        if total_segment_cost < min_total_segment_cost:
                            if self.obstacle.check_path_viable(q_start=prev_q.q_array, q_end=q_mid.q_array) and self.obstacle.check_path_viable(q_start=q_mid.q_array, q_end=next_q.q_array):
                                min_total_segment_cost = total_segment_cost
                                best_q_for_point = q_mid
                            # else:
                            #    print(f"Rejected candidate at path index {i} due to collision during smoothing.")

                old_total_cost = sum(self.get_transition_cost(self.waypoints[j], q_path[j], q_path[j - 1]) if j > 0 else 0 for j in range(len(q_path)))
                temp_path = q_path.copy()
                temp_path[i] = best_q_for_point
                new_total_cost = sum(self.get_transition_cost(self.waypoints[j], temp_path[j], temp_path[j - 1]) if j > 0 else 0 for j in range(len(temp_path)))

                if new_total_cost <= old_total_cost:
                    q_path[i] = best_q_for_point

            print(
                f"  -> Smoothing iteration {iter_num + 1}/{num_smoothing_iterations} complete, new cost {sum(self.get_transition_cost(self.waypoints[i], q_path[i], q_path[i - 1]) if i > 0 else 0 for i in range(len(q_path))):.2f}. "
            )

        return q_path

    def get_list_of_best_q(self) -> np.ndarray:
        planner_start_time = time.time()
        if not self.waypoints:
            raise ValueError("No waypoints available.")

        all_ik_solutions = self.get_all_ik_solutions()

        q_path = self.backward_greedy_search(all_ik_solutions)  # Reverse to get from start to end
        coarse_cost = sum(self.get_transition_cost(self.waypoints[i], q_path[i], q_path[i - 1]) if i > 0 else 0 for i in range(len(q_path)))
        coarse_path = deepcopy(q_path)

        num_smoothing_iterations = 1
        q_path = self.smooth_path(q_path, all_ik_solutions, planner_start_time, num_smoothing_iterations)



        new_total_cost = sum(self.get_transition_cost(self.waypoints[i], q_path[i], q_path[i - 1]) if i > 0 else 0 for i in range(len(q_path)))

        if coarse_cost < new_total_cost:
            q_path = coarse_path
            print(f"Smoothing increased cost; reverting to coarse path with cost {coarse_cost:.2f}.")

        else:
            print(f"Final path found with {len(q_path)} points after smoothing for cost {new_total_cost:.2f}.")
        
        for i in range(len(q_path)):
            transition_cost = self.get_transition_cost(self.waypoints[i], q_path[i], q_path[i - 1]) if i > 0 else 0
            assert transition_cost < 350, f"Transition cost too high: {transition_cost:.2f}"
            print(f"transition cost {i}: {transition_cost:.2f}")

        print(f"took total time: {time.time() - planner_start_time:.2f} seconds")

        return np.array([q.q_array for q in q_path])
