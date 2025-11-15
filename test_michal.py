from eel import init
from src.core.planning import PathFollowingPlanner
from src.core.obstacles import Obstacle
from ctu_crs import CRS97, CRS93
from src.interface.robot_interface import RobotInterface
from src.core.obstacles import Obstacle
from src.core.planning_michal import HoopPathOptimizer
import numpy as np
from src.core.se3 import SE3
import matplotlib.pyplot as plt
from src.core.helpers import visualize_homography, project_homography, draw_3d_frame


if __name__ == "__main__":
    robot = RobotInterface(CRS97(tty_dev=None))
    maze_position = SE3(translation=np.array([0.35, 0.2, 0.1]))
    obstacle = Obstacle("D", "src/tools/models", maze_position, num_waypoints=15)
    obstacle.prep_obstacle()
    maze_waypoints = obstacle.waypoints

    # Michals stupid optimizer

    init_planner = PathFollowingPlanner(robot, maze_waypoints, robot.hoop_ik)
    init_q_list = np.array(init_planner.get_list_of_best_q())

    print("Initial guess from PathFollowingPlanner:")
    fig = plt.figure(figsize=(8, 8), layout="tight")
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=40.0, azim=-150)
    # vizualize
    for q in init_q_list:
        T = robot.hoop_fk(q)
        draw_3d_frame(ax, T.rotation.rot, T.translation, scale=0.02)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()

    print("Starting Michal's HoopPathOptimizer...")  # take down max_iter if it takes too long
    planner = HoopPathOptimizer(robot, maze_waypoints, robot.hoop_fk, robot.fk, init_q_list, max_iter=50)
    best_q_list = planner.get_list_of_best_q()

    print(best_q_list)

    fig = plt.figure(figsize=(8, 8), layout="tight")
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=40.0, azim=-150)
    # vizualize
    for q in best_q_list:
        T = robot.hoop_fk(q)
        draw_3d_frame(ax, T.rotation.rot, T.translation, scale=0.02)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()
