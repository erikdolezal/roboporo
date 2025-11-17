import random
import numpy as np
import time

from src.core.se3 import SE3
from src.core.so3 import SO3
from src.interface.robot_interface import RobotInterface
from src.core.obstacles import Obstacle


class RRTPlanner:
    def __init__(self, robot_interface: RobotInterface, obstacle: Obstacle, step_size: float = 0.1, max_iter: int = 1000) -> None:
        self.robot_interface = robot_interface
        self.obstacle = obstacle
        self.step_size = step_size
        self.max_iter = max_iter
        
        self.tree = {}  # key: tuple(q), value: Node
        
    def plan(self, start_q: np.ndarray, goal_q: np.ndarray) -> list[np.ndarray]:
        """
        Plans a path from start_q to goal_q using the RRT algorithm.
        """
        start_node = self.Node(start_q)
        self.tree[tuple(start_q)] = start_node
        
        for iteration in range(self.max_iter):
            rand_q = self.sample_random_configuration()
            nearest_node = self.find_nearest_node(rand_q)
            new_q = self.steer_towards(nearest_node.q, rand_q)
            
            if not self.check_collision(new_q):
                new_node = self.Node(new_q, parent=nearest_node)
                self.tree[tuple(new_q)] = new_node
                
                if np.linalg.norm(new_q - goal_q) < self.step_size:
                    goal_node = self.Node(goal_q, parent=new_node)
                    self.tree[tuple(goal_q)] = goal_node
                    return self.reconstruct_path(goal_node)
        
        return []  # No path found
    
    def sample_random_configuration(self) -> np.ndarray:
        """Samples a random configuration within the robot's joint limits."""
        q_min = self.robot_interface.robot.q_min
        q_max = self.robot_interface.robot.q_max
        return np.array([random.uniform(q_min[i], q_max[i]) for i in range(len(q_min))])
    
    def find_nearest_node(self, q: np.ndarray) -> 'RRTPlanner.Node':
        """Finds the nearest node in the tree to the given configuration q."""
        nearest_node = None
        min_dist = float('inf')
        
        for node in self.tree.values():
            dist = np.linalg.norm(node.q - q)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
                
        return nearest_node
    
    def steer_towards(self, from_q: np.ndarray, to_q: np.ndarray) -> np.ndarray:
        """Steers from from_q towards to_q by step_size."""
        direction = to_q - from_q
        length = np.linalg.norm(direction)
        if length <= self.step_size:
            return to_q
        else:
            return from_q + (direction / length) * self.step_size
        
    def check_collision(self, q: np.ndarray) -> bool:
        """Checks if the configuration q is in collision with the obstacle."""
        # TODO: Implement collision checking logic
        return False  # Placeholder, assume no collision
    
    def reconstruct_path(self, goal_node: 'RRTPlanner.Node') -> list[np.ndarray]:
        """Reconstructs the path from start to goal by backtracking from the goal node."""
        path = []
        current_node = goal_node
        while current_node is not None:
            path.append(current_node.q)
            current_node = current_node.parent
        path.reverse()
        return path
    
    class Node:
        def __init__(self, q: np.ndarray, parent: 'RRTPlanner.Node' = None) -> None:
            self.q = q
            self.parent = parent
        
        
        