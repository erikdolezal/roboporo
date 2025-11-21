from sympy import rad
from src.core.planning import PathFollowingPlanner as PlannerIterative
from src.core.planning_michal import PathFollowingPlanner as PlannerGreedyBckwrd
from src.core.obstacles import Obstacle
from ctu_crs import CRS97, CRS93
from src.interface.robot_interface import RobotInterface
from src.core.obstacles import Obstacle
import numpy as np
from src.core.se3 import SE3
from src.core.so3 import SO3
import matplotlib.pyplot as plt
from src.core.helpers import visualize_homography, project_homography, draw_3d_frame


if __name__ == "__main__":
    robot = RobotInterface(CRS97(tty_dev=None))
    maze_position = SE3(
        translation=np.array([0.30, -0.16, 0.05]),
        rotation=SO3.from_euler_angles(
            np.deg2rad(np.array([0.0, 0, -42])),
            ["x", "y", "z"],
        ),
    )
    """"
    sus C
    maze_position = SE3(
        translation=np.array([0.34, -0.21, 0.05]),
        rotation=SO3.from_euler_angles(
            np.deg2rad(np.array([0.0, 0, -75])),
            ["x", "y", "z"],
        ),
    )
    
    
    
    fucked E
    maze_position = SE3(
        translation=np.array([0.30, 0.01, 0.05]),
        rotation=SO3.from_euler_angles(
            np.deg2rad(np.array([0.0, 0, -25])), or np.array([0.0, 0, -30]))
            ["x", "y", "z"],
        ),
    )
    maze_position = SE3(
        translation=np.array([0.30, -0.05, 0.05]),
        rotation=SO3.from_euler_angles(
            np.deg2rad(np.array([0.0, 0, -40])),
            ["x", "y", "z"],
        ),
    )
    
    """
    obstacle = Obstacle(robot, "B", "src/tools/models", maze_position, num_of_colision_points=200)
    obstacle.prep_obstacle()
    maze_waypoints = obstacle.waypoints

    # print("Waypoints:")
    # for wp in maze_waypoints:
    # print(wp)

    # Michals stupid optimizer

    init_planner = PlannerGreedyBckwrd(robot, obstacle, maze_waypoints, robot.hoop_ik)
    best_q_list = np.array(init_planner.get_list_of_best_q())

    fig = plt.figure(figsize=(8, 8), layout="tight")
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=40.0, azim=-150)

    interpolated_qs = best_q_list  # np.linspace(prev_q, q, num=int(2 + 0*np.linalg.norm(actual_pose.translation - prev_pose.translation) / 0.01), endpoint=True)
    for iq in interpolated_qs:
        T = robot.hoop_fk(iq)
        draw_3d_frame(ax, T.rotation.rot, T.translation, scale=0.02)
        ax.plot(*T.translation, marker="o", color="blue", markersize=2)
    ax.plot(*obstacle.line_final.T, color="black")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()

    obstacle.visualize_path(best_q_list)

    """
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
    """
