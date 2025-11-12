import numpy as np
from src.core.se3 import SE3
from src.core.so3 import SO3
from src.interface.robot_interface import RobotInterface
from src.core.planning import Planning
from src.core.obstacles import Obstacle
from ctu_crs import CRS97, CRS93



if __name__ == "__main__":
    robot_interface = RobotInterface(CRS97(None))
    planning_params_path = "/media/onvan/data/ctu/ROB/roboporo/src/core/data/planning_params.yaml"
    obstacle = Obstacle(
        type="B",
        path="/media/onvan/data/ctu/ROB/roboporo/src/tools/models",
        transform=SE3(np.array([0, 0, 0]), SO3())
    )
    obstacle.prep_obstacle()
    obstacle.change_box_offset(np.array([0.05, 0.05, 0.05]))
    print(obstacle.box)
    planner = Planning(robot_interface, planning_params_path, obstacles=[obstacle])
    start_q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    goal_q = np.array([0.3, 0.2, 0.1, 0.0, -5.0, 0.0])
    path = planner.planner.plan(start_q, goal_q)
    if path is not None:
        print("Planned path:")
        for q in path:
            print(q)
    else:
        print("No path found.")