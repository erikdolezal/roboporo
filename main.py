import argparse
from ctu_crs import CRS97, CRS93
from src.interface.robot_interface import RobotInterface


def main(args):
    if args.robot == "CRS97":
        robot = RobotInterface(CRS97())
    elif args.robot == "CRS93":
        robot = RobotInterface(CRS93())
    robot.initialize(home = args.home)

    



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
    args = parser.parse_args()
    main(args)
