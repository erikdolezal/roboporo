import random
import numpy as np
import time

from src.core.se3 import SE3
from src.core.so3 import SO3
from src.interface.robot_interface import RobotInterface
from src.core.obstacles import Obstacle


class RRTPlanner:
    def __init__(self, robot_interface: RobotInterface, obstacle: Obstacle, step_size: float = 0.25, goal_tol: float = 0.25, max_iter: int = 1000) -> None:
        self.robot_interface = robot_interface
        self.obstacle = obstacle
        self.step_size = step_size
        self.max_iter = max_iter
        self.goal_tol = goal_tol
        
        self.tree = {}  # key: tuple(q), value: Node
        
        self.seed_q = None
        self.seed_sigma = 0.3
        self.seed_probability = 0.6
        
    def plan(self, start_q: np.ndarray, goal_q: np.ndarray) -> list[np.ndarray]:
        """
        Plans a path from start_q to goal_q using the RRT algorithm.
        """
        start_node = self.Node(start_q)
        self.tree[tuple(start_q)] = start_node
        self.seed_q = start_q.copy()
        
        for iteration in range(self.max_iter):
            rand_q = self.sample_random_configuration(goal_q)
            nearest_node = self.find_nearest_node(rand_q)
            new_q = self.steer_towards(nearest_node.q, rand_q)
            if not self.check_collision(new_q):
                new_node = self.Node(new_q, parent=nearest_node)
                self.tree[tuple(new_q)] = new_node
                self.check_best_seed(goal_q, new_q)
                
                if np.linalg.norm(new_q - goal_q) < self.goal_tol:
                    goal_node = self.Node(goal_q, parent=new_node)
                    self.tree[tuple(goal_q)] = goal_node
                    print(f"Goal reached in iteration {iteration}!, distance to goal: {np.linalg.norm(new_q - goal_q)}")
                    return self.reconstruct_path(goal_node)
            else:
                print(f"Iteration {iteration}: Collision at {new_q}")
            
        print("Failed to find a path within the maximum iterations.")
        return []  # No path found
    
    def path_smooth(self):
        """Placeholder for path smoothing method."""
        # TODO: Implement path smoothing if needed
        pass  
    
    def check_best_seed(self, goal: np.ndarray, new: np.ndarray) -> None:
        """Updates the seed configuration to the best node found so far towards the goal."""
        if self.seed_q is None or np.linalg.norm(new - goal) < np.linalg.norm(self.seed_q - goal):
            self.seed_q = new.copy()
    
    def sample_random_configuration(self, q_goal: np.ndarray, goal_bias: float = 0.1) -> np.ndarray:
        """Samples a random configuration within the robot's joint limits."""
        
        randomization = random.random()
        if randomization < goal_bias:
            return q_goal.copy()
        
        if randomization < self.seed_probability:
            return np.array([np.clip(random.gauss(self.seed_q[i], self.seed_sigma), self.robot_interface.robot.q_min[i], self.robot_interface.robot.q_max[i]) for i in range(len(self.seed_q))])
        
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
        return self.obstacle.check_hoop_collision(q)
    
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
        
        
        