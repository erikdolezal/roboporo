import numpy as np
from scipy.optimize import minimize, Bounds
from typing import List, Optional

from src.core.se3 import SE3

from src.interface.robot_interface import RobotInterface


class HoopPathOptimizer:
    """
    Implements the global "batch" optimization logic (Solution 1).
    """

    def __init__(self, robot_interface: RobotInterface, waypoints: List[SE3], fk_hoop, fk_arm, init_guess: np.ndarray, max_iter: int = 200) -> None:
        self.robot_interface = robot_interface
        self.waypoints = waypoints
        self.fk_hoop = fk_hoop
        self.fk_end = fk_arm
        self.init_guess = init_guess
        self.max_iter = max_iter

        self.num_waypoints = len(self.waypoints)
        self.num_joints = self.robot_interface.q_min.shape[0]

        # --- Optimization Weights (TUNE THESE!) ---
        self.W_POS = 200.0  # Must be high
        self.W_MOVE = 1.0  # Smoothness
        self.W_PENALTY = 1.0  # Hoop orientation
        self.W_ORTHO = 10.0  # Orthogonality to z-axis for last two waypoints

        # --- Constraints ---
        self.TABLE_Z_MIN = 0.06
        self.TANGENT_MAX_ANGLE_RAD = np.radians(20.0)
        self.TANGENT_MIN_DOT_PROD = np.cos(self.TANGENT_MAX_ANGLE_RAD)

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

            #!!!change to penalize specic q positions!!!

            # --- Cost 2: "Hoop Facing Away" Penalty ---
            hoop_x_axis = T_pose.rotation.rot[:, 0]
            # The direction "toward me" (World +X)
            me_vector = np.array([1, 0, 0])
            # Calculate the dot product
            # +1 = facing me (good)
            # -1 = facing away (bad)
            dot_prod = np.dot(hoop_x_axis, me_vector)
            # The optimizer minimizes, so we want to minimize (-dot_prod)
            # This maximizes the dot product.

            # find out whether to put + or - sign !!!!!
            total_cost += self.W_PENALTY * (-dot_prod)

            # --- Cost 3: Joint Movement (Smoothness) ---
            if i > 0:
                q_prev = Q[i - 1]
                move_error = q_i - q_prev
                total_cost += self.W_MOVE * np.dot(move_error, move_error)

            # --- Cost 4: Orthogonality to Z-axis for last two waypoints ---
            if i >= self.num_waypoints - 2:
                world_z_axis = np.array([0, 0, 1])
                # Penalize if hoop axes are NOT orthogonal to z (i.e., dot product != 0)
                # Check all three hoop axes
                hoop_x_axis = T_pose.rotation.rot[:, 0]
                hoop_y_axis = T_pose.rotation.rot[:, 1]

                # Orthogonal means dot product should be 0
                # Penalize the squared dot products
                dot_x = np.dot(hoop_x_axis, world_z_axis)
                dot_y = np.dot(hoop_y_axis, world_z_axis)

                # Cost increases as any axis becomes non-orthogonal to z
                ortho_cost = dot_x**2 + dot_y**2
                total_cost += self.W_ORTHO * ortho_cost

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

        q_list_guess = self.init_guess

        q_list_guess_2d = np.array(q_list_guess)
        X_guess = q_list_guess_2d.flatten()

        # 3. Define Global Constraints

        constraints = [{"type": "ineq", "fun": self._constraints_table}, {"type": "ineq", "fun": self._constraints_tangent}]

        # 4. Run the Optimizer

        # started minimizing
        print("Running global optimization (this may take a while)...")

        result = minimize(self._objective_function, X_guess, method="SLSQP", bounds=global_bounds, constraints=constraints, options={"maxiter": self.max_iter, "disp": True})  # Be patient!

        # done minimizing
        print("Global optimization completed.")

        if not result.success:
            # Check *why* it failed.
            if "Iteration limit reached" in result.message:
                print(f"WARNING: Iteration limit reached. Using best solution found so far.")
                print(f"         Final cost: {result.fun}")
                # We accept this, but it's not guaranteed to be optimal OR valid.
            else:
                # This is a *real* failure (e.g., incompatible constraints)
                raise ValueError("Global optimization failed to converge. " f"Message: {result.message}")

        print("\nGlobal optimization successful.")

        # 5. Unpack the final solution
        optimized_Q = self._unpack_X(result.x)

        return optimized_Q


class SmoothingPathOptimizer:
    """
    Implements a local "smoothing" optimization logic (Solution 2).
    """

    def __init__(self, robot_interface: RobotInterface, waypoints: List[SE3], fk_hoop, fk_arm, init_guess: np.ndarray, max_iter: int = 100) -> None:
        self.robot_interface = robot_interface
        self.waypoints = waypoints
        self.fk_hoop = fk_hoop
        self.fk_end = fk_arm
        self.init_guess = init_guess
        self.max_iter = max_iter

    def get_list_of_best_q(self) -> np.ndarray:
        """
        Placeholder for smoothing optimizer.
        Currently just returns the initial guess.
        """
        print("SmoothingPathOptimizer is not yet implemented. Returning initial guess.")
        return self.init_guess
