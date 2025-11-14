import numpy as np
from scipy.optimize import minimize, Bounds
from typing import List, Optional

from src.core.se3 import SE3

from src.interface.robot_interface import RobotInterface
from src.core.planning import PathFollowingPlanner


class HoopPathOptimizer:
    """
    Implements the global "batch" optimization logic (Solution 1).
    """

    def __init__(self, robot_interface: RobotInterface, waypoints: List[SE3], fk_hoop, fk_arm):
        self.robot_interface = robot_interface
        self.waypoints = waypoints
        self.fk_hoop = fk_hoop
        self.fk_end = fk_arm

        self.num_waypoints = len(self.waypoints)
        self.num_joints = self.robot_interface.q_min.shape[0]

        # --- Optimization Weights (TUNE THESE!) ---
        self.W_POS = 1000.0  # Must be high
        self.W_MOVE = 1.0  # Smoothness
        self.W_PENALTY = 0.1  # Hoop orientation

        # --- Constraints ---
        self.TABLE_Z_MIN = 0.06
        self.TANGENT_MAX_ANGLE_RAD = np.radians(30.0)
        self.TANGENT_MIN_DOT_PROD = np.cos(self.TANGENT_MAX_ANGLE_RAD)

        self.AXIS_MIN_DOT_PROD = [0, 1, 0]

    def _unpack_X(self, X: np.ndarray) -> np.ndarray:
        """Helper to reshape the flat 1D variable array into a 2D array."""
        return X.reshape(self.num_waypoints, self.num_joints)

    def _objective_function(self, X: np.ndarray) -> float:
        """
        Calculates the total cost for the entire trajectory 'X'.
        """
        # Reshape flat array X into 2D array Q
        Q = self._unpack_X(X)

        total_cost = 0.0

        for i in range(self.num_waypoints):
            q_i = Q[i]
            waypoint_i = self.waypoints[i]

            # --- Forward Kinematics (the bottleneck) ---
            T_matrix = self.fk_hoop(q_i)
            T_pose = T_matrix

            # --- Cost 1: Position Error ---
            pos_error = T_pose.translation - waypoint_i.translation
            total_cost += self.W_POS * np.dot(pos_error, pos_error)

            # --- Cost 2: "Hoop Facing Away" Penalty ---
            hoop_y_axis = T_pose.rotation.rot[:, 1]
            world_z_axis = np.array([0, 0, 1])
            dot_prod = np.dot(hoop_y_axis, world_z_axis)
            total_cost += self.W_PENALTY * (-dot_prod)

            # --- Cost 3: Joint Movement (Smoothness) ---
            if i > 0:
                q_prev = Q[i - 1]
                move_error = q_i - q_prev
                total_cost += self.W_MOVE * np.dot(move_error, move_error)

        return total_cost

    def _constraints_table(self, X: np.ndarray) -> np.ndarray:
        """
        Returns an array of table constraint values (one per waypoint).
        All values must be >= 0.
        """
        Q = self._unpack_X(X)
        results = np.zeros(self.num_waypoints)
        for i in range(self.num_waypoints):
            T_pose = SE3().from_homogeneous(self.fk_end(Q[i]))
            results[i] = T_pose.translation[2] - self.TABLE_Z_MIN
        return results

    def _constraints_tangent(self, X: np.ndarray) -> np.ndarray:
        """
        Returns an array of tangent constraint values (one per waypoint).
        All values must be >= 0.
        """
        Q = self._unpack_X(X)
        results = np.zeros(self.num_waypoints)
        for i in range(self.num_waypoints):
            T_pose = self.fk_hoop(Q[i])
            stick_tangent = self.waypoints[i].rotation.rot[:, 2]
            hoop_z_axis = T_pose.rotation.rot[:, 2]

            dot_prod = np.dot(hoop_z_axis, stick_tangent)
            results[i] = dot_prod - self.TANGENT_MIN_DOT_PROD
        return results

    def get_list_of_best_q(self) -> np.ndarray:
        """
        Calculates the list of optimal joint configurations using
        global "batch" optimization.
        """
        if self.num_waypoints == 0:
            raise ValueError("No waypoints available.")

        print(f"Starting global optimization for {self.num_waypoints} waypoints...")
        print(f"Total variables: {self.num_waypoints * self.num_joints}")

        # 1. Define Global Bounds

        min_bounds = np.tile(self.robot_interface.q_min, self.num_waypoints)
        max_bounds = np.tile(self.robot_interface.q_max, self.num_waypoints)
        global_bounds = Bounds(min_bounds, max_bounds)

        # 2. Define Initial Guess (X_guess)

        intial_planner = PathFollowingPlanner(self.robot_interface, self.waypoints, self.robot_interface.hoop_ik)
        q_list_guess = intial_planner.get_list_of_best_q()

        q_list_guess_2d = np.array(q_list_guess)
        X_guess = q_list_guess_2d.flatten()

        # 3. Define Global Constraints

        constraints = [{"type": "ineq", "fun": self._constraints_table}, {"type": "ineq", "fun": self._constraints_tangent}]

        # 4. Run the Optimizer

        # started minimizing
        print("Running global optimization (this may take a while)...")

        result = minimize(self._objective_function, X_guess, method="SLSQP", bounds=global_bounds, constraints=constraints, options={"maxiter": 200, "disp": True})  # Be patient!

        # done minimizing
        print("Global optimization completed.")

        if not result.success:
            raise ValueError("Global optimization failed to converge. " f"Message: {result.message}")

        print("\nGlobal optimization successful.")

        # 5. Unpack the final solution
        final_Q = self._unpack_X(result.x)
        return final_Q
