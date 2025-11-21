import time
import os
from cv2 import circle
import numpy as np
from src.core.se3 import SE3
from src.core.so3 import SO3
from typing import List, cast
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.core.helpers import draw_3d_frame
from src.interface.robot_interface import RobotInterface


class Obstacle:
    def __init__(
        self, robot_interface: RobotInterface, type: str, path: str, transform: SE3, start: float = 0.04, end: float = 10.0, num_waypoints: int = 20, num_of_colision_points: int = 50
    ) -> None:
        self.type = type
        self.path = path
        self.transform = transform
        self.line_raw = None
        self.line_final = None
        self.start = start
        self.end = end
        self.box = None
        self.box_offset = np.array([0.04, 0.04, 0.04])
        self.waypoints: List[SE3] = []
        self.num_waypoints = num_waypoints
        self.robot_interface = robot_interface
        self.ground_limit = 0.02
        self.colision_points: list[np.ndarray] = []
        self.num_of_colision_points = num_of_colision_points
        self.hoop_stick = SE3(translation=np.array([-0.1, 0.0, 0.0]), rotation=SO3().from_euler_angles(np.deg2rad(np.array([0.0, 180.0, 0.0])), "xyz"))

        # Collision detection parameters
        # self.arm_radius = 0.12  # meters
        # if self.type == "E":
        # self.arm_radius = 0.085  # DEPRICATED FOR COLISSION CHECKING

        self.thick_arm_radius = 0.1  # meters
        self.normal_arm_radius = 0.085  # meters
        self.end_arm_radius = 0.05  # meters
        self.hoop_stick_radius = 0.03  # meters
        self.hoop_thickness = 0.004  # meters

        # Major and minor radius of the torus obstacle
        self.major_radius = 0.06 / 2  # meters
        self.minor_radius = 0.01  # meters

    def prep_obstacle(self) -> None:
        """Prepare the obstacle by loading, cropping, transforming, and hiding it in a box."""
        self.open_centerline()
        self.set_crop_limits()
        self.crop_centerline_z()
        self.tranform_centerline()
        self.hide_in_box()
        self.sample_centerline_points(num_points=self.num_waypoints)
        self.colision_points = self.get_colision_points(num_of_colision_points=self.num_of_colision_points)
        # self.hide_in_box(offset=self.box_offset)

        fig = plt.figure(figsize=(8, 8), layout="tight")
        ax = cast(Axes3D, fig.add_subplot(111, projection="3d"))

        if self.line_final is None:
            raise ValueError("Centerline not transformed. Call tranform_centerline() first.")

        ax.plot(*self.line_final.T)

        for wp in self.waypoints:
            draw_3d_frame(ax, wp.rotation.rot, wp.translation, scale=0.02)

        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.show()

    def get_colision_points(self, num_of_colision_points=50) -> np.ndarray:
        """Get 50 evenly spaced colision points along the centerline."""
        points = []
        if self.line_final is None:
            raise ValueError("Centerline not transformed. Call tranform_centerline() first.")
        if num_of_colision_points > 0:
            if num_of_colision_points > len(self.line_final):
                num_of_colision_points = len(self.line_final)
                points = [self.line_final[i] for i in range(len(self.line_final))]
            else:
                total_points = len(self.line_final)
                step = total_points / num_of_colision_points
                points = [self.line_final[int(i * step)] for i in range(num_of_colision_points)]

            print(f"Generated {len(points)} collision points.")
            return np.array(points)

        return np.array(points)

    def get_centerline(self) -> np.ndarray:
        """Get the transformed centerline points."""
        if self.line_final is None:
            raise ValueError("Centerline not transformed. Call tranform_centerline() first.")
        return self.line_final

    def sample_centerline_points(self, num_points: int = 30) -> list[SE3]:
        num_points = int(num_points - 1)
        if self.line_final is None:
            raise ValueError("Centerline not transformed. Call tranform_centerline() first.")

        if num_points <= 0:
            raise ValueError("num_points must be positive.")

        total_points = len(self.line_final)
        if num_points > total_points:
            raise ValueError(f"num_points ({num_points}) cannot exceed total points ({total_points}).")

        # Calculate step size to sample every xth point
        if num_points == 1:
            indices = [0]
        elif num_points == 2:
            indices = [0, total_points - 1]
        else:
            step = (total_points - 1) / (num_points - 1)
            indices = [int(round(i * step)) for i in range(num_points)]
            # Ensure first and last indices are exact
            indices[0] = 0
            indices[-1] = total_points - 1

        # Create SE3 transformations with tangent directions
        se3_list = []
        positions = []
        tangents = []
        for i in range(len(self.line_final)):
            position = np.mean(self.line_final[max(0, i - 4) : min(total_points, i + 5)], axis=0)

            # Calculate tangent direction
            if i == 0:
                # First point: use direction to next point
                tangent = self.line_final[i + 1] - self.line_final[i]
            elif i == len(self.line_final) - 1:
                # Last point: use direction from previous point
                tangent = self.line_final[i] - self.line_final[i - 1]
            else:
                # Middle points: use central difference
                tangent = self.line_final[i + 1] - self.line_final[i - 1]

            _, _, Vt = np.linalg.svd(self.line_final[max(0, i - 10) : min(len(self.line_final), i + 11)] - position)

            if np.dot(tangent, Vt[0]) < 0:
                tangent = -Vt[0]
            else:
                tangent = Vt[0]

            # Normalize tangent
            tangent = tangent / np.linalg.norm(tangent)
            positions.append(position)
            tangents.append(tangent)

        tangents[-1] = np.array([0, 0, -1])

        prev_tangent = tangents[0]
        prev_position = positions[0]
        dist_th = 0.02
        for i in range(len(tangents)):
            dist = np.linalg.norm(positions[i] - prev_position)
            dot_product = np.dot(tangents[i], prev_tangent)
            if i == 0 or i == len(tangents) - 1 or (dist < dist_th and dot_product < np.cos(np.deg2rad(25))) or (dist >= dist_th and dot_product < np.cos(np.deg2rad(5))) or dist > 0.03:
                prev_position = positions[i].copy()
                prev_tangent = tangents[i].copy()
                tangent = tangents[i]
                position = positions[i]

                # Create rotation matrix with tangent as z-axis
                z_axis = -tangent

                # Choose arbitrary perpendicular vector for y-axis, handle singularity
                if np.allclose(np.abs(z_axis), [0, 1, 0]):
                    # Tangent is parallel to Y-axis, use Z-axis for cross product
                    x_axis = np.cross(z_axis, [0, 0, 1])
                else:
                    # Default case
                    x_axis = np.cross(z_axis, [0, 1, 0])

                # Check for zero vector in case of unforeseen issues
                if np.linalg.norm(x_axis) < 1e-6:
                    # If still a problem, use a fallback (e.g., world X-axis)
                    x_axis = np.cross(z_axis, [1, 0, 0])

                x_axis = x_axis / np.linalg.norm(x_axis)

                # Complete the orthonormal basis
                y_axis = np.cross(z_axis, x_axis)  # Swapped order to ensure right-handed system

                # Create rotation matrix
                rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])

                # Ensure it's a valid rotation matrix (handle potential floating point inaccuracies)
                if np.linalg.det(rotation_matrix) < 0:
                    # Flip one axis to correct the handedness
                    x_axis = -x_axis
                    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])

                # print(f"Rotation matrix at waypoint {i}:\n{rotation_matrix}\n")

                # Create SE3 transformation

                se3 = SE3(translation=position, rotation=SO3(rotation_matrix))
                se3_list.append(se3)

        first_se3 = se3_list[0]
        first_tangent = first_se3.rotation.rot[:, 2]  # x-axis is the tangent
        start_position = first_se3.translation + 0.05 * first_tangent  # 5 cm = 0.05 m
        start_se3 = SE3(translation=start_position, rotation=first_se3.rotation)
        se3_list.insert(0, start_se3)

        self.waypoints = se3_list
        return se3_list

    def check_arm_colision(self, true_q: np.ndarray, for_path: bool = False, hoop_collision_radius=None) -> tuple[bool, float, float]:
        """
        Check if the robot arm at configuration true_q collides with any of the obstacle's waypoints.
        The last segment of the arm has a smaller collision radius.
        """

        hoop_thickness = self.hoop_thickness if not for_path else hoop_collision_radius

        assert hoop_thickness is not None, "hoop_collision_radius must be provided when for_path is True."

        fk_frames = self.fk_for_all(true_q)
        fk_frames.append(fk_frames[-1] * self.hoop_stick)  # Append end-effector frame again for hoop collision check

        if len(self.colision_points) == 0:
            return False, 0.4, 0.035

        # Convert collision points to numpy array for vectorized operations
        collision_points_array = np.array(self.colision_points)  # Shape: (N, 3)

        all_dists = []

        for i in [1, 3, 5, 6]:
            frame_A = fk_frames[i]
            frame_B = fk_frames[i + 1]

            # Use appropriate radius for each segment
            if i == 1:  # first segment
                # print(f"len of frame AB {frame_A}, {frame_B}")
                radius = self.thick_arm_radius
            elif i == 3:  # thick segment
                radius = self.normal_arm_radius
            elif i == 5:  # end segment
                radius = self.end_arm_radius
            elif i == 6:  # hoop stick
                radius = self.hoop_stick_radius
            else:
                radius = self.thick_arm_radius  # Default fallback

            collisions, dists = self.check_segment_to_points_collision_vectorized(i, frame_A, frame_B, collision_points_array, radius)
            all_dists.extend(dists)

            # Check if any collision occurred
            if np.any(collisions):
                # Find the first collision (to maintain same behavior as original)
                # print("Arm collision detected.")
                collision_idx = np.where(collisions)[0][0]
                return True, float(dists[collision_idx]), 0.033

        circle_collisions, circle_dists = self.check_circle_to_point_collision_vectorized(fk_frames[-1], collision_points_array, circle_radius=0.03, collision_threshold=hoop_thickness)
        if np.any(circle_collisions):

            collision_idx = np.where(circle_collisions)[0][0]
            # print(f"circle_dists: {circle_dists}, collision_idx: {collision_idx}")
            # print(f"hoop {fk_frames[-1]} colided with points: {collision_points_array[collision_idx]} at idx {collision_idx}")
            return True, float(circle_dists[collision_idx]), float(min(all_dists))

        return False, float(min(all_dists)), float(min(circle_dists))

    def check_segment_to_point_collision(self, idx, frame_A: SE3, frame_B: SE3, point_P: np.ndarray, radius: float) -> tuple[bool, float]:
        """
        Calculates the minimum distance between a line segment (A-B) and a point (P).
        Returns True if the distance is less than or equal to the given radius.
        """
        A = frame_A.translation
        B = frame_B.translation
        v = B - A

        if A[2] - radius < self.ground_limit or B[2] - radius < self.ground_limit:
            # ground check
            segment_length = np.linalg.norm(v)
            if segment_length < 1e-9:  # Degenerate case: A ≈ B
                min_z = min(A[2], B[2])
                clearance = min_z - radius - self.ground_limit
            else:
                u = v / segment_length

                if abs(u[2]) > 0.999:  # Nearly vertical cylinder
                    radius_z_component = 0
                else:

                    # The z-component of the perpendicular vector pointing most downward is:
                    # -sqrt(u[0]² + u[1]²) = -sqrt(1 - u[2]²)
                    horizontal_component = np.sqrt(u[0] ** 2 + u[1] ** 2)  # = sqrt(1 - u[2]²)
                    radius_z_component = radius * horizontal_component

                surface_z_at_A = A[2] - radius_z_component
                surface_z_at_B = B[2] - radius_z_component
                min_z = min(surface_z_at_A, surface_z_at_B)

                clearance = min_z - self.ground_limit

            if clearance <= 0.0:
                # print("Ground collision detected.")
                return True, 0.0

        # Vector from A to B (the segment)

        # Vector from A to P
        w = point_P - A

        # Projection of w onto v
        dot_vv = np.dot(v, v)
        if dot_vv < 1e-9:  # Segment is effectively a point
            return False, 0.2

        t = np.dot(w, v) / dot_vv

        # Only consider orthogonal projections that lie strictly on the segment

        if idx == 3:
            t = np.maximum(0, np.minimum(1, t))

        else:
            if not (0.0 < t < 1.0):
                return False, 0.4

        closest_point = A + t * v
        distance = np.linalg.norm(point_P - closest_point)

        clearance = distance - radius

        return bool(clearance <= 0.0), float(clearance if bool(clearance >= 0.0) else 0)

    def check_circle_to_point_collision_vectorized(self, circle_SE3: SE3, points_P: np.ndarray, circle_radius: float = 0.03, collision_threshold: float = 0.004) -> tuple[np.ndarray, list[float]]:
        """
        Vectorized collision detection between a hoop ring and multiple points for wire threading.
        Only detects collision when wire actually hits the ring material, never for center-passing wire.
        Calculates true 3D distance from points to the torus surface.

        Args:
            circle_SE3: SE3 transformation defining the circle's position and orientation
            points_P: Array of points to check collision against, shape (N, 3)
            circle_radius: Major radius of the circle (distance from center to the ring)
            collision_threshold: Thickness of the ring material for collision detection

        Returns:
            tuple: (collision_array, distances_list) where collision_array is boolean array of shape (N,)
                   and distances_list contains the clearance distances for each point
        """
        # Transform points to the local frame of the circle
        # The circle is defined in the xy-plane of the local frame
        R = circle_SE3.rotation.rot
        t = circle_SE3.translation

        # Vector from circle center to points in world frame
        diff_world = points_P - t

        # Transform to local frame: v_local = v_world @ R (since R columns are local axes)
        points_local = diff_world @ R

        # Calculate distance to the ring in the local frame
        # The ring is a torus in the xy-plane with major radius `circle_radius`
        # and minor radius `collision_threshold`.

        # Radial distance in xy-plane (distance from z-axis)
        rho = np.linalg.norm(points_local[:, :2], axis=1)

        # Vertical distance (z-axis component)
        z = points_local[:, 2]

        # Distance from the point to the circular centerline of the torus
        # This considers both the radial distance difference and the height difference
        dist_to_centerline = np.sqrt((rho - circle_radius) ** 2 + z**2)

        # Clearance is distance to the surface of the ring (minor radius is collision_threshold)
        # If dist_to_centerline < collision_threshold, the point is inside the ring material
        clearances = dist_to_centerline - collision_threshold

        collisions = clearances <= 0.0

        # For return, we want distances to be 0.0 if collision, else the clearance
        distances = np.where(clearances >= 0.0, clearances, 0.0)

        return collisions, distances.tolist()

    def check_segment_to_points_collision_vectorized(self, idx: int, frame_A: SE3, frame_B: SE3, points_P: np.ndarray, radius: float) -> tuple[np.ndarray, list[float]]:
        """
        Vectorized version that calculates the minimum distance between a line segment (A-B) and multiple points.
        Returns arrays of collision results and distances for all points at once.

        Args:
            idx: Segment index
            frame_A: Start frame of the segment
            frame_B: End frame of the segment
            points_P: Array of points to check, shape (N, 3)
            radius: Collision radius

        Returns:
            tuple: (collision_array, distances_list) where collision_array is boolean array of shape (N,)
        """
        A = frame_A.translation
        B = frame_B.translation
        v = B - A

        N = points_P.shape[0]

        # Ground collision check (vectorized)
        segment_length = np.linalg.norm(v)

        if A[2] - radius < self.ground_limit or B[2] - radius < self.ground_limit:
            if segment_length < 1e-9:  # Degenerate case: A ≈ B
                min_z = min(A[2], B[2])
                clearance = min_z - radius - self.ground_limit
                if clearance <= 0.0:
                    return np.ones(N, dtype=bool), [0.0] * N
            else:
                u = v / segment_length
                if abs(u[2]) > 0.999:  # Nearly vertical cylinder
                    radius_z_component = 0
                else:
                    horizontal_component = np.sqrt(u[0] ** 2 + u[1] ** 2)
                    radius_z_component = radius * horizontal_component

                surface_z_at_A = A[2] - radius_z_component
                surface_z_at_B = B[2] - radius_z_component
                min_z = min(surface_z_at_A, surface_z_at_B)
                clearance = min_z - self.ground_limit

                if clearance <= 0.0:
                    return np.ones(N, dtype=bool), [0.0] * N

        # Vector from A to all points (vectorized)
        w = points_P - A  # Shape: (N, 3)

        # Projection calculations (vectorized)
        dot_vv = np.dot(v, v)
        if dot_vv < 1e-9:  # Segment is effectively a point
            return np.zeros(N, dtype=bool), [0.2] * N

        # t values for all points at once
        t_values = np.dot(w, v) / dot_vv  # Shape: (N,)

        # Handle different segment rules
        if idx == 3:
            t_values = np.maximum(0, np.minimum(1, t_values))
            valid_mask = np.ones(N, dtype=bool)  # All points are considered for segment 3
        else:
            valid_mask = (t_values > 0.0) & (t_values < 1.0)

        # Initialize results
        collisions = np.zeros(N, dtype=bool)
        distances = np.full(N, 0.4)  # Default distance for invalid projections

        if np.any(valid_mask):
            # Calculate closest points for valid projections
            valid_indices = np.where(valid_mask)[0]
            t_valid = t_values[valid_mask]  # Shape: (num_valid,)

            # Vectorized closest point calculation
            # closest_points shape: (num_valid, 3)
            closest_points = A + t_valid.reshape(-1, 1) * v

            # Calculate distances for valid points
            valid_points = points_P[valid_mask]  # Shape: (num_valid, 3)
            point_distances = np.linalg.norm(valid_points - closest_points, axis=1)

            # Calculate clearances
            clearances = point_distances - radius

            # Update results for valid points
            distances[valid_indices] = np.where(clearances >= 0.0, clearances, 0.0)
            collisions[valid_indices] = clearances <= 0.0

        return collisions, distances.tolist()

    def check_path_viable(self, q_start, q_end) -> bool:
        """Check if path between q_start and q_end is viable (collision-free)."""
        num_steps = 11
        for i in range(num_steps + 1):
            alpha = i / num_steps
            q_interp = (1 - alpha) * q_start + alpha * q_end
            collision, _, _ = self.check_arm_colision(q_interp, for_path=True, hoop_collision_radius=0.001)
            if collision:
                return False
        return True

    def check_hoop_collision(self, q: np.ndarray, segment: tuple[float, float] = (0.0, 10.0)) -> bool:
        """Check if the robot hoop at configuration q collides with the obstacle.
        Args:
            q (np.ndarray): The joint configuration of the robot.
        Returns:
            bool: True if there is a collision, False otherwise.
        """
        if self.check_arm_colision(q)[0]:
            return True

        if self.check_hoop_in_box(q):
            return True

        return False

    # ----------------------Inner-Helper-Functions-----------------------------------------------

    def check_hoop_in_box(self, q: np.ndarray) -> bool:
        """Check if the robot hoop at configuration q is inside the collision box.
        Args:
            q (np.ndarray): The joint configuration of the robot.
        Returns:
            bool: True if the hoop is inside the box, False otherwise.
        """

        if self.box is None:
            raise ValueError("Collision box not defined. Call hide_in_box() first.")

        hoop_position = self.robot_interface.hoop_fk(q).translation

        min_box, max_box = self.box

        if min_box[0] <= hoop_position[0] <= max_box[0] and min_box[1] <= hoop_position[1] <= max_box[1] and min_box[2] <= hoop_position[2] <= max_box[2]:
            return True
        else:
            return False

    def hide_in_box(self) -> None:
        """Hide puzzle in the collision box"""
        if self.line_final is None:
            raise ValueError("Centerline not transformed. Call tranform_centerline() first.")

        min_z = -0.05
        max_z = np.max(self.line_final[:, 2]) + self.box_offset[2]
        min_y = np.min(self.line_final[:, 1]) - self.box_offset[1]
        max_y = np.max(self.line_final[:, 1]) + self.box_offset[1]
        min_x = np.min(self.line_final[:, 0]) - self.box_offset[0]
        max_x = np.max(self.line_final[:, 0]) + self.box_offset[0]

        self.box = (np.array([min_x, min_y, min_z]), np.array([max_x, max_y, max_z]))

    def set_crop_limits(self) -> None:
        """Set cropping limits based on obstacle type."""
        if self.type == "A":
            self.start = 0.025
            self.end = 10.0
        elif self.type == "B":
            self.start = 0.025
            self.end = 10.0
        elif self.type == "C":
            self.start = 0.025
            self.end = 10.0
        elif self.type == "D":
            self.start = 0.025
            self.end = 10.0
        elif self.type == "E":
            self.start = 0.025
            self.end = 10.0
        else:
            raise ValueError(f"Unknown obstacle type: {self.type}")

    def fk_for_all(self, q: np.ndarray) -> List[SE3]:
        """
        Compute FK for all joints of the robot, starting from the robot's base frame.
        The first frame in the returned list is the robot's base itself.
        """
        robot = self.robot_interface.robot
        # Start with the robot's base transformation matrix (assumed identity if at origin).
        T = np.eye(4)
        # The first "frame" is the base of the robot.
        transformations = [SE3().from_homogeneous(T)]

        # The provided q values are the variable joint angles.
        variable_thetas = q

        # Iterate through the DH parameters for each link.
        for i in range(len(robot.dh_d)):
            d = robot.dh_d[i]
            a = robot.dh_a[i]
            alpha = robot.dh_alpha[i]

            # The total angle is the fixed DH offset plus the variable joint angle q.
            theta = robot.dh_offset[i] + variable_thetas[i]

            # Calculate the transformation for the current link, using the correct parameter order.
            link_transform = robot.dh_to_se3(d, theta, a, alpha)

            # Post-multiply to get the cumulative transformation from the base.
            T = T @ link_transform
            transformations.append(SE3().from_homogeneous(T))

        return transformations

    def open_centerline(self) -> None:
        """Open the centerline file for the obstacle."""
        if self.line_raw is None:
            path_to_file = os.path.join(self.path, f"{self.type}_centerline.npy")
            self.line_raw = np.load(path_to_file)
            self.line_raw = self.line_raw.astype(np.float64) / 1000.0  # Convert mm to m
        if self.line_raw[0, 2] < self.line_raw[-1, 2]:
            self.line_raw = self.line_raw[::-1]

    def crop_centerline_z(self) -> None:
        """Crop the centerline based on start and end distances."""
        if self.line_raw is None:
            raise ValueError("Centerline not loaded. Call open_centerline() first.")
        z_values = self.line_raw[:, 2]
        mask = (z_values >= self.start) & (z_values <= self.end)
        idxs = np.where(mask)[0]
        if idxs.size == 0:
            raise ValueError("No points found in the specified z range.")

        start_idx = idxs[0]
        end_idx = idxs[-1]
        self.line_raw = self.line_raw[start_idx : end_idx + 1]

    def tranform_centerline(self) -> None:
        """Transform the centerline points using the obstacle's SE3 transform."""
        if self.line_raw is None:
            raise ValueError("Centerline not loaded. Call open_centerline() first.")
        transformed_points = []
        for point in self.line_raw:
            transformed_point = self.transform.act(point)
            transformed_points.append(transformed_point)
        self.line_final = np.array(transformed_points)

    def visualize_path(self, q_path: np.ndarray) -> None:
        """
        Creates a 3D animation of the robot moving through a given path of joint configurations.

        Args:
            robot_interface (RobotInterface): The robot interface object.
            obstacle (Obstacle): The obstacle object, used for FK calculations and to draw the obstacle.
            q_path (np.ndarray): A numpy array where each row is a joint configuration (q) in the path.
        """
        fig = plt.figure(figsize=(10, 10))
        ax = cast(Axes3D, fig.add_subplot(111, projection="3d"))
        plt.ion()  # Turn on interactive mode for animation

        # Plot the obstacle's centerline
        if self.line_final is None:
            raise ValueError("Centerline not transformed. Call tranform_centerline() first.")

        centerline = self.line_final
        ax.plot(centerline[:, 0], centerline[:, 1], centerline[:, 2], "r-", label="Obstacle Centerline")

        # Set plot limits based on the obstacle to keep the view consistent
        min_coords = np.min(centerline, axis=0)
        max_coords = np.max(centerline, axis=0)
        ax.set_xlim([min_coords[0] - 0.5, max_coords[0] + 0.5])
        ax.set_ylim([min_coords[1] - 0.5, max_coords[1] + 0.5])
        ax.set_zlim([min_coords[2] - 0.5, max_coords[2] + 0.5])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Robot Path Visualization")
        ax.legend()

        # This list will hold the plot objects for the robot arm so we can remove them each frame
        robot_lines = []

        for i, q in enumerate(q_path):
            # Clear the previous robot arm drawing
            for line in robot_lines:
                line.remove()
            robot_lines.clear()

            # Get the 3D coordinates of each joint for the current configuration
            fk_frames = self.fk_for_all(q)
            joint_positions = np.array([frame.translation for frame in fk_frames])

            # Draw the robot arm segments
            for j in range(len(joint_positions) - 1):
                p1 = joint_positions[j]
                p2 = joint_positions[j + 1]
                (line,) = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], "b-o", linewidth=4, markersize=6)
                robot_lines.append(line)

            # Draw a hoop at the end effector
            end_effector_frame = self.robot_interface.hoop_fk(q)
            hoop_radius = 0.06 / 2  # Major radius of the hoop
            num_hoop_points = 50
            theta = np.linspace(0, 2 * np.pi, num_hoop_points)
            hoop_points = np.array([hoop_radius * np.cos(theta), hoop_radius * np.sin(theta), np.zeros(num_hoop_points)]).T  # Shape (num_hoop_points, 3)
            hoop_points_transformed = np.array([end_effector_frame.act(point) for point in hoop_points])
            (hoop_line,) = ax.plot(hoop_points_transformed[:, 0], hoop_points_transformed[:, 1], hoop_points_transformed[:, 2], "g-", linewidth=2)
            robot_lines.append(hoop_line)
            # Update the title to show progress
            ax.set_title(f"Robot Path Visualization (Step {i+1}/{len(q_path)})")

            # Draw collision box
            if self.box is not None:
                min_box, max_box = self.box
                box_lines = [
                    # Bottom face
                    ([min_box[0], min_box[1], min_box[2]], [max_box[0], min_box[1], min_box[2]]),
                    ([max_box[0], min_box[1], min_box[2]], [max_box[0], max_box[1], min_box[2]]),
                    ([max_box[0], max_box[1], min_box[2]], [min_box[0], max_box[1], min_box[2]]),
                    ([min_box[0], max_box[1], min_box[2]], [min_box[0], min_box[1], min_box[2]]),
                    # Top face
                    ([min_box[0], min_box[1], max_box[2]], [max_box[0], min_box[1], max_box[2]]),
                    ([max_box[0], min_box[1], max_box[2]], [max_box[0], max_box[1], max_box[2]]),
                    ([max_box[0], max_box[1], max_box[2]], [min_box[0], max_box[1], max_box[2]]),
                    ([min_box[0], max_box[1], max_box[2]], [min_box[0], min_box[1], max_box[2]]),
                    # Vertical edges
                    ([min_box[0], min_box[1], min_box[2]], [min_box[0], min_box[1], max_box[2]]),
                    ([max_box[0], min_box[1], min_box[2]], [max_box[0], min_box[1], max_box[2]]),
                    ([max_box[0], max_box[1], min_box[2]], [max_box[0], max_box[1], max_box[2]]),
                    ([min_box[0], max_box[1], min_box[2]], [min_box[0], max_box[1], max_box[2]]),
                ]
                for line_start, line_end in box_lines:
                    (box_line,) = ax.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], [line_start[2], line_end[2]], "k--", linewidth=1)
                    robot_lines.append(box_line)

            # Pause to create the animation effect
            plt.pause(0.75)

        ax.set_title("Robot Path Visualization (Finished)")
        plt.ioff()  # Turn off interactive mode
        plt.show()
