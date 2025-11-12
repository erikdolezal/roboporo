import argparse
from ctu_crs import CRS97, CRS93
from src.interface.robot_interface import RobotInterface
from src.core.obstacles import Obstacle
from src.core.planning import PathFollowingPlanner

def main(args):
    tty_dev = None if args.local else "/dev/mars"
    if args.robot == "CRS97":
        robot = RobotInterface(CRS97(tty_dev=tty_dev))
    elif args.robot == "CRS93":
        robot = RobotInterface(CRS93(tty_dev=tty_dev))
    if not args.local:
        robot.initialize(home = (args.home))
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

            robot.follow_q_list(best_q_list)

            robot.follow_q_list(best_q_list[::-1])
            robot.soft_home()








        #robot.soft_home()
        #robot.close()

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

    parser.add_argument(
        "--maze",
        type=str,
        help="Maze configuration"
    )
    args = parser.parse_args()
    main(args)
