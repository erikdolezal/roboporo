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
        robot.calibrate_camera()

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
