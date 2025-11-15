import os
import numpy as np
from src.core.se3 import SE3
from src.core.so3 import SO3
from typing import List
import matplotlib.pyplot as plt
from src.core.helpers import draw_3d_frame


class Obstacle:
    def __init__(self, type: str, path: str, transform: SE3, radius: float = 0.005, start: float = 0.04, end: float = 10.0, num_waypoints: int = 20) -> None:
        self.type = type
        self.path = path
        self.transform = transform
        self.line_raw = None
        self.line_final = None
        self.start = start
        self.end = end
        self.box = None
        self.box_offset = np.array([0.1, 0.1, 0.1])
        self.waypoints: List[SE3] = []
        self.num_waypoints = num_waypoints

    def prep_obstacle(self) -> None:
        """Prepare the obstacle by loading, cropping, transforming, and hiding it in a box."""
        self.open_centerline()
        self.crop_centerline_z()
        self.tranform_centerline()
        self.sample_centerline_points(num_points=self.num_waypoints)
        self.hide_in_box(offset=self.box_offset)

        fig = plt.figure(figsize=(8, 8), layout="tight")
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(*self.line_final.T)

        for wp in self.waypoints:
            draw_3d_frame(ax, wp.rotation.rot, wp.translation, scale=0.02)

        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.show()

    def check_point_in_box(self, point: np.ndarray) -> bool:
        """Check if a given point is inside the obstacle's box."""
        if self.box is None:
            raise ValueError("Box not defined. Call hide_in_box() first.")
        min_coords, max_coords = self.box
        return np.all(point >= min_coords) and np.all(point <= max_coords)

    def get_centerline(self) -> np.ndarray:
        """Get the transformed centerline points."""
        if self.line_final is None:
            raise ValueError("Centerline not transformed. Call tranform_centerline() first.")
        return self.line_final

    def change_box_offset(self, new_offset: np.ndarray) -> None:
        """Change the box offset for hiding the obstacle."""
        if new_offset.shape != (3,):
            raise ValueError("Offset must be a 3D vector.")
        self.box_offset = new_offset
        self.hide_in_box(offset=self.box_offset)

    def sample_centerline_points(self, num_points: int = 30) -> list[SE3]:
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

        # Sample points
        sampled_points = self.line_final[indices]

        # Create SE3 transformations with tangent directions
        se3_list = []
        for i, idx in enumerate(indices):
            position = np.mean(self.line_final[max(0, idx - 2) : min(total_points, idx + 3)], axis=0)

            # Calculate tangent direction
            if idx == 0:
                # First point: use direction to next point
                tangent = self.line_final[idx + 1] - self.line_final[idx]
            elif idx == total_points - 1:
                # Last point: use direction from previous point
                tangent = self.line_final[idx] - self.line_final[idx - 1]
            else:
                # Middle points: use central difference
                tangent = self.line_final[idx + 1] - self.line_final[idx - 1]

            _, _, Vt = np.linalg.svd(self.line_final[max(0, idx - 2) : min(total_points, idx + 3)] - position)

            if np.dot(tangent, Vt[0]) < 0:
                tangent = -Vt[0]
            else:
                tangent = Vt[0]

            # Normalize tangent
            tangent = tangent / np.linalg.norm(tangent)

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

    # ----------------------Inner-Helper-Functions-----------------------------------------------

    def open_centerline(self) -> None:
        """Open the centerline file for the obstacle."""
        if self.line_raw is None:
            path_to_file = os.path.join(self.path, f"{self.type}_centerline.npy")
            self.line_raw = np.load(path_to_file)
            self.line_raw = self.line_raw.astype(np.float64) / 1000.0  # Convert mm to m

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

    def hide_in_box(self, offset: np.ndarray) -> None:
        """Hide the obstacle within a box defined by offset and size."""
        if self.line_final is None:
            raise ValueError("Centerline not transformed. Call tranform_centerline() first.")
        min_coords = np.min(self.line_final, axis=0) - offset
        max_coords = np.max(self.line_final, axis=0) + offset
        self.box = (min_coords, max_coords)
