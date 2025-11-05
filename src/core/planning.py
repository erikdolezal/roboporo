import numpy as np
from src.core.se3 import SE3, SO3
from src.interface.robot_interface import RobotInterface
import yaml


class Planning:
    def __init__(self, robot_interface: RobotInterface, planning_params_path: str, obstacles: list = None):
        self.robot_interface = robot_interface
        
        # Load parameters from yaml file
        self.planning_params = self.load_planning_params(planning_params_path)

        self.obstacles = obstacles if obstacles is not None else []
        
        # Initialize RRT* planner
        self.planner = RRTStarPlanner(robot_interface, self.planning_params)
        if self.obstacles:
            self.planner.set_obstacles(self.obstacles)

    def load_planning_params(self, planning_params_path: str) -> dict:
        """Load planning parameters from a YAML file.

        Args:
            planning_params_path (str): Path to the YAML file containing planning parameters.
        Returns:
            dict: A dictionary containing the planning parameters.
        """
        with open(planning_params_path, 'r') as file:
            planning_params = yaml.safe_load(file)
        return planning_params

class RRTStarPlanner:
    def __init__(self, robot_interface: RobotInterface, planning_params: dict):
        self.robot_interface = robot_interface
        self.planning_params = planning_params
        self.obstacles = []
        # C-space requires joint limits; expect them in planning_params
        joint_limits = planning_params.get('joint_limits', None)
        self.c_space = self.CSpace(robot_interface, joint_limits)

        self.tree_root = None
        self.tree_goal = None
        # list of Node objects (will initialize with root in plan())
        self.tree_nodes = []

    def plan(self, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
        self.tree_root = self.Node(q=start)
        self.tree_goal = self.Node(q=goal)
        # initialize tree
        self.tree_root.parent = None
        self.tree_root.cost = 0.0
        self.tree_nodes = [self.tree_root]

        # best goal tracking (RRT* will improve cost over time)
        best_goal_node = None
        best_goal_cost = np.inf
        
        # Check if start and goal are connected directly
        if self.collision_free_edge(start, goal):
            self.tree_goal.parent = self.tree_root
            self.tree_goal.cost = np.linalg.norm(goal - start)
            self.tree_nodes.append(self.tree_goal)
            return self.get_path()
        
        # RRT* main loop
        for _ in range(self.planning_params['num_iterations']):
            # Sample a random configuration
            q_rand = self.sample_random_configuration()

            # Find the nearest node in the trees
            nearest_node = self.find_nearest_node(q_rand)

            # If the tree is empty or no nearest found, skip this sample
            if nearest_node is None:
                continue

            # Steer from the nearest node towards the random configuration
            new_node = self.steer(nearest_node, q_rand)

            if not self.collision_free_edge(nearest_node.q, new_node.q):
                continue

            # RRT* neighbor radius (can be provided or computed)
            neighbor_radius = self.planning_params.get('neighbor_radius', None)
            if neighbor_radius is None:
                # default dynamic radius: gamma * (log(n)/n)^(1/d)
                n = max(1, len(self.tree_nodes))
                d = float(len(start)) if start is not None else 1.0
                gamma = float(self.planning_params.get('rrt_star_gamma', 1.5))
                neighbor_radius = gamma * (np.log(n + 1) / (n + 1)) ** (1.0 / d)

            neighbors = self.find_neighbors(new_node.q, neighbor_radius)

            # Choose best parent among neighbors (including nearest)
            candidate_parents = neighbors.copy()
            if nearest_node not in candidate_parents:
                candidate_parents.append(nearest_node)

            best_parent = nearest_node
            best_cost = nearest_node.cost + np.linalg.norm(new_node.q - nearest_node.q)
            for p in candidate_parents:
                if not self.collision_free_edge(p.q, new_node.q):
                    continue
                cost_through_p = p.cost + np.linalg.norm(p.q - new_node.q)
                if cost_through_p < best_cost:
                    best_cost = cost_through_p
                    best_parent = p

            new_node.parent = best_parent
            new_node.cost = best_cost
            self.tree_nodes.append(new_node)

            # Rewire neighbors through new_node if it reduces cost
            for nbr in neighbors:
                if nbr is new_node or nbr is best_parent:
                    continue
                if not self.collision_free_edge(new_node.q, nbr.q):
                    continue
                new_cost = new_node.cost + np.linalg.norm(new_node.q - nbr.q)
                if new_cost < nbr.cost:
                    nbr.parent = new_node
                    self._update_subtree_costs(nbr)

            # Goal check: try to connect new_node to goal
            dist_to_goal = np.linalg.norm(new_node.q - goal)
            if dist_to_goal < self.planning_params.get('goal_tolerance', 1e-3):
                if self.collision_free_edge(new_node.q, goal):
                    goal_cost = new_node.cost + dist_to_goal
                    if goal_cost < best_goal_cost:
                        best_goal_cost = goal_cost
                        best_goal_node = new_node

        # attach best goal if found
        if best_goal_node is not None:
            self.tree_goal.parent = best_goal_node
            self.tree_goal.cost = best_goal_cost
            self.tree_nodes.append(self.tree_goal)
        else:
            # No feasible goal connection found
            pass

        return self.get_path()
    
    def get_path(self) -> np.ndarray:
        """Retrieve the planned path from start to goal.
         Returns:
            path (np.ndarray): The planned path as an array of configurations.
        """
        path = []
        current_node = self.tree_goal
        while current_node is not None:
            path.append(current_node.q)
            current_node = current_node.parent
        path.reverse()
        return np.array(path)

    def steer(self, from_node: 'RRTStarPlanner.Node', to_q: np.ndarray) -> 'RRTStarPlanner.Node':
        """Steers the branch from a tree node in direction of chosen node

        Args:
            from_node (RRTPlanner.Node): The node to steer from.
            to_q (np.ndarray): The target configuration to steer towards.

        Returns:
            RRTPlanner.Node: The new node created by steering.
        """
        vec = to_q - from_node.q
        norm = np.linalg.norm(vec)
        if norm <= 0.0:
            return self.Node(parent=from_node, q=from_node.q.copy(), cost=from_node.cost)
        step = float(self.planning_params.get('step_size', 0.1))
        direction = (vec / norm) * min(step, norm)
        new_q = from_node.q + direction
        new_node = self.Node(parent=from_node, q=new_q, cost=from_node.cost + np.linalg.norm(direction))
        return new_node
    
    def find_nearest_node(self, q_rand: np.ndarray) -> 'RRTStarPlanner.Node':
        """Find the nearest node in the tree to the random configuration.
        Args:
            q_rand (np.ndarray): The random configuration.
        Returns:
            nearest_node (RRTPlanner.Node): The nearest node in the tree.
        """
        if not self.tree_nodes:
            return None
        # vectorized distance computation for speed
        q_array = np.vstack([np.atleast_1d(n.q) for n in self.tree_nodes])
        diffs = q_array - q_rand
        d2 = np.einsum('ij,ij->i', diffs, diffs)
        idx = int(np.argmin(d2))
        return self.tree_nodes[idx]
    
    def sample_random_configuration(self) -> np.ndarray:
        """Sample a random configuration within the robot's joint limits.

        Returns:
            q_rand (np.ndarray): Random configuration.
        """
        joint_limits = self.planning_params['joint_limits']
        q_rand = np.array([np.random.uniform(low, high) for low, high in joint_limits])
        return q_rand

    def find_neighbors(self, q: np.ndarray, radius: float):
        """Return list of nodes within `radius` of q."""
        if not self.tree_nodes:
            return []
        q_array = np.vstack([np.atleast_1d(n.q) for n in self.tree_nodes])
        diffs = q_array - q
        d2 = np.einsum('ij,ij->i', diffs, diffs)
        idxs = np.where(d2 <= radius * radius)[0]
        return [self.tree_nodes[int(i)] for i in idxs]

    def collision_free_edge(self, q1: np.ndarray, q2: np.ndarray) -> bool:
        """Check that straight-line interpolation between q1 and q2 is collision-free.

        Samples points along the segment using step size `edge_check_step` (or half
        of step_size by default).
        """
        dist = np.linalg.norm(q2 - q1)
        if dist == 0.0:
            return not self.c_space.is_in_c_space(q1)
        step = float(self.planning_params.get('edge_check_step', self.planning_params.get('step_size', 0.1)))
        n_samples = max(1, int(np.ceil(dist / step)))
        for i in range(1, n_samples + 1):
            t = i / float(n_samples)
            q = q1 + t * (q2 - q1)
            if self.c_space.is_in_c_space(q):
                return False
        return True

    def update_subtree_costs(self, root_node):
        """Update costs for root_node and its descendants after a parent change."""
        # update root_node cost from its parent
        if root_node.parent is None:
            root_node.cost = 0.0
        else:
            root_node.cost = root_node.parent.cost + np.linalg.norm(root_node.q - root_node.parent.q)

        # recursively update children
        children = [n for n in self.tree_nodes if n.parent is root_node]
        for c in children:
            self.update_subtree_costs(c)

    def set_obstacles(self, obstacles: list):
        """Sets the obstacles for the planner.
            Args:
                obstacles (List): list of obstacles to set
        """
        self.obstacles = obstacles
        pass
    
    def get_C_space(self):
        """ Get the configuration space by limiting the robot's joint limits and considering obstacles.
        Returns:
            C_space (object): The configuration space object.
        """
        pass
    
    def set_number_of_iterations(self, n: int):
        """Set the number of iterations for the planner.

        Args:
            n (int): The number of iterations.
        """
        self.planning_params['num_iterations'] = n
        
    def set_step_size(self, step_size: float):
        """Set the step size for the planner.

        Args:
            step_size (float): The step size.
        """
        self.planning_params['step_size'] = step_size
        
    def set_goal_tolerance(self, tolerance: float):
        """Set the goal tolerance for the planner.

        Args:
            tolerance (float): The goal tolerance.
        """
        self.planning_params['goal_tolerance'] = tolerance
        
    
    class Node:
        def __init__(self, parent = None, q = None, cost = 0.0):
            self.parent = parent
            self.q = q
            self.cost = cost
            
            
    class CSpace:
        def __init__(self, robot_interface: RobotInterface, joint_limits: np.ndarray):
            self.robot_interface = robot_interface
            self.joint_limits = joint_limits
            self.obstacles = []
            
        def add_obstacle(self, obstacle):
            """Adds obstacles to existing ones

            Args:
                obstacle (List): list of obstacles to add
            """
            self.obstacles.extend(obstacle)
        
        def is_in_c_space(self, current_q: np.ndarray) -> bool:
            """Check if the given configuration is within the configuration space.
            Args:
                current_q (np.ndarray): The configuration to check.
            Returns:
                bool: True if in configuration space, False otherwise.
            """
            
            for i, (low, high) in enumerate(self.joint_limits):
                if not (low <= current_q[i] <= high):
                    return False
                
            for obstacle in self.obstacles:
                if obstacle.is_in_collision(current_q, self.robot_interface):
                    return False
                
            return True

        def is_in_collision(self, current_q: np.ndarray) -> bool:
            """Check if the given configuration is in collision.
            Args:
                current_q (np.ndarray): The configuration to check.
            Returns:
                bool: True if in collision, False otherwise.
            """
            for obstacle in self.obstacles:
                if self.is_in_obstacle(current_q, obstacle, self.robot_interface):
                     return True
            return False
        
        def is_in_obstacle(self, current_q: np.ndarray, obstacle, robot_interface: RobotInterface) -> bool:
            """Check if the given configuration is in collision with a specific obstacle.
            Args:
                current_q (np.ndarray): The configuration to check.
                obstacle: The obstacle to check against.
                robot_interface (RobotInterface): The robot interface for kinematics.
            Returns:
                bool: True if in collision with the obstacle, False otherwise.
            """
            # Implement specific collision checking logic here
            pass