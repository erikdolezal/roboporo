import argparse
from ctu_crs import CRS97, CRS93
from src.interface.robot_interface import RobotInterface

def main(args):
    tty_dev = None if args.local else "/dev/mars"
    if args.robot == "CRS97":
        robot = RobotInterface(CRS97(tty_dev=tty_dev))
    elif args.robot == "CRS93":
        robot = RobotInterface(CRS93(tty_dev=tty_dev))
    if not args.local:
        robot.initialize(home = args.home)
        print(robot.get_actual_pose())
        #robot.move_absolute(0,92,0,0.4,0.1,0.4)
        robot.calibrate_camera()
    robot.soft_home()
    robot.close()

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

    args = parser.parse_args()
    main(args)
