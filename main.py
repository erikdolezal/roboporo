import argparse
import numpy as np
from ctu_crs import CRS97, CRS93
from src.interface.robot_interface import RobotInterface
from src.core.obstacles import Obstacle
from src.core.planning import PathFollowingPlanner
from src.core.se3 import SE3
from src.core.so3 import SO3
import matplotlib.pyplot as plt
from configs.aruco_config import aruco_config
from src.core.helpers import visualize_homography, project_homography, draw_3d_frame


def main(args):
    tty_dev = None if args.local else "/dev/mars"
    if args.robot == "CRS97":
        robot = RobotInterface(CRS97(tty_dev=tty_dev))
    elif args.robot == "CRS93":
        robot = RobotInterface(CRS93(tty_dev=tty_dev))
    if not args.local:
        robot.initialize(home=(args.home))
        robot.soft_home()
        if args.calibrate_camera:
            robot.calibrate_camera()
        else:
            maze_position = robot.get_maze_position()

            obstacle = Obstacle(args.maze, "src/tools/models", maze_position)
            obstacle.prep_obstacle()
            maze_waypoints = obstacle.waypoints
            planner = PathFollowingPlanner(robot, maze_waypoints, robot.hoop_ik)
            best_q_list = planner.get_list_of_best_q()

            fig = plt.figure(figsize=(8, 8), layout="tight")
            ax = fig.add_subplot(111, projection="3d")
            ax.view_init(elev=40.0, azim=-150)

            for q in best_q_list:
                T = robot.hoop_fk(q)
                draw_3d_frame(ax, T.rotation.rot, T.translation, scale=0.02)
            ax.set_aspect("equal")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            plt.show()

            robot.follow_q_list(best_q_list)

            robot.follow_q_list(best_q_list[::-1])
            robot.soft_home()
    else:
        maze_position = SE3(translation=np.array([0.35, -0.09, 0.05]), rotation=SO3().from_euler_angles(np.deg2rad(np.array([0.0, 0.0, 120.0])), "xyz"))

        obstacle = Obstacle(args.maze, "src/tools/models", maze_position)
        obstacle.prep_obstacle()
        maze_waypoints = obstacle.waypoints
        planner = PathFollowingPlanner(robot, maze_waypoints, robot.hoop_ik)
        best_q_list = planner.get_list_of_best_q()

        fig = plt.figure(figsize=(8, 8), layout="tight")
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=40.0, azim=-150)

        for q in best_q_list:
            T = robot.hoop_fk(q)
            draw_3d_frame(ax, T.rotation.rot, T.translation, scale=0.02)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.show()

        # robot.soft_home()
        # robot.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize CRS Robot")
    parser.add_argument(
        "--robot",
        type=str,
        choices=["CRS97", "CRS93"],
        required=True,
        help="Type of CRS robot to initialize",
    )
    parser.add_argument(
        "--home",
        action="store_true",
        help="Home the robot after initialization",
    )

    parser.add_argument(
        "--local",
        action="store_true",
        help="Use class in local",
    )

    parser.add_argument(
        "--calibrate_camera",
        action="store_true",
        help="Calibrate camera after initialization",
    )

    parser.add_argument("--maze", type=str, help="Maze configuration")
    args = parser.parse_args()
    main(args)
