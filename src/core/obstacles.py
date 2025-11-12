import os
import numpy as np
from src.core.se3 import SE3
from src.core.so3 import SO3

class Obstacle:
    def __init__(self, type: str, path: str, transform: SO3, radius: float = 0.005, start: float = 0.0, end: float = 10.0) -> None:
        self.type = type
        self.path = path
        self.transform = transform
        self.line_raw = None
        self.line_final = None
        self.start = start
        self.end = end
        self.box = None
        self.box_offset = np.array([0.1, 0.1, 0.1])
        
    def prep_obstacle(self) -> None:
        """Prepare the obstacle by loading, cropping, transforming, and hiding it in a box.
        """
        self.open_centerline()
        self.crop_centerline_z()
        self.tranform_centerline()
        self.hide_in_box(offset=self.box_offset)        
            
    def check_point_in_box(self, point: np.ndarray) -> bool:
        """Check if a given point is inside the obstacle's box.
        """
        if self.box is None:
            raise ValueError("Box not defined. Call hide_in_box() first.")
        min_coords, max_coords = self.box
        return np.all(point >= min_coords) and np.all(point <= max_coords)
    
    def get_centerline(self) -> np.ndarray:
        """Get the transformed centerline points.
        """
        if self.line_final is None:
            raise ValueError("Centerline not transformed. Call tranform_centerline() first.")
        return self.line_final
    
    def change_box_offset(self, new_offset: np.ndarray) -> None:
        """Change the box offset for hiding the obstacle.
        """
        if new_offset.shape != (3,):
            raise ValueError("Offset must be a 3D vector.")
        self.box_offset = new_offset
        self.hide_in_box(offset=self.box_offset)

    # ----------------------Inner-Helper-Functions-----------------------------------------------

    def open_centerline(self) -> None:
        """Open the centerline file for the obstacle.
        """
        if self.line_raw is None:
            path_to_file = os.path.join(self.path, f"{self.type}_centerline.npy")
            self.line_raw = np.load(path_to_file)
            self.line_raw = self.line_raw.astype(np.float64)/1000.0  # Convert mm to m
            
    def crop_centerline_z(self) -> None:
        """Crop the centerline based on start and end distances.
        """
        if self.line_raw is None:
            raise ValueError("Centerline not loaded. Call open_centerline() first.")
        z_values = self.line_raw[:, 2]
        mask = (z_values >= self.start) & (z_values <= self.end)
        idxs = np.where(mask)[0]
        if idxs.size == 0:
            raise ValueError("No points found in the specified z range.")

        start_idx = idxs[0]
        end_idx = idxs[-1]
        self.line_raw = self.line_raw[start_idx:end_idx+1]
        
    def tranform_centerline(self) -> None:
        """Transform the centerline points using the obstacle's SE3 transform.
        """
        if self.line_raw is None:
            raise ValueError("Centerline not loaded. Call open_centerline() first.")
        transformed_points = []
        for point in self.line_raw:
            transformed_point = self.transform.act(point)
            transformed_points.append(transformed_point)
        self.line_final = np.array(transformed_points)
    
    def hide_in_box(self, offset: np.ndarray) -> None:
        """Hide the obstacle within a box defined by offset and size.
        """
        if self.line_final is None:
            raise ValueError("Centerline not transformed. Call tranform_centerline() first.")
        min_coords = np.min(self.line_final, axis=0) - offset
        max_coords = np.max(self.line_final, axis=0) + offset
        self.box = (min_coords, max_coords)
       


