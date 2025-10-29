import numpy as np
from ctu_crs import CRS97
from src.core.so3 import SO3
from src.core.se3 import SE3

robot = CRS97()
robot.initialize()


def move_relative(dx, dy, dz):
    q0 = robot.get_q()
    current_pose = robot.fk(q0)
    current_pose[:3, 3] += np.array([dx, dy, dz])
    ik_sols = robot.ik(current_pose)
    print(current_pose)
    if len(ik_sols) > 0:
        closest_solution = min(ik_sols, key=lambda q: np.linalg.norm(q - q0))
        robot.move_to_q(closest_solution)
        robot.wait_for_motion_stop()
    else:
        print("tos prestrelil miso")

def move_absolute(phi, theta, psi, x, y, z):
    target_pose = SE3(translation = np.array([x, y, z]), 
                        rotation=SO3().from_euler_angles(np.deg2rad(np.array([phi, theta, psi])), "xyz"))
    q0 = robot.get_q()
    ik_sols = robot.ik(target_pose.homogeneous())
    if len(ik_sols) > 0:
        closest_solution = min(ik_sols, key=lambda q: np.linalg.norm(q - q0))
        robot.move_to_q(closest_solution)
        robot.wait_for_motion_stop()
        print("target_pose:\n", target_pose)
        q0 = robot.get_q()
        current_pose = robot.fk(q0)
        print("actual pose:\n", SE3().from_homogeneous(current_pose))
    else:
        print("tos prestrelil miso")
    

def move_joint_relative(joint_index, delta_angle_deg):
    q0 = robot.get_q()
    q = robot.get_q()
    q[joint_index] += np.deg2rad(delta_angle_deg)
    assert robot.in_limits(q)
    desired_pose = robot.fk(q)
    ik_sols = robot.ik(desired_pose)
    assert len(ik_sols) > 0
    closest_solution = min(ik_sols, key=lambda q: np.linalg.norm(q - q0))
    robot.move_to_q(closest_solution)
    robot.wait_for_motion_stop()


q0 = robot.get_q()
current_pose = robot.fk(q0)
print(SE3().from_homogeneous(current_pose))
move_absolute(0, 165, 0, 0.4, 0, 0.2)
move_absolute(0, 165, 0, 0.4, 0.2, 0.2)
move_absolute(0, 165, 0, 0.4, -0.2, 0.2)
#move_absolute(0, -60, 0, 0.3, 0.0, 0.2)
#move_absolute(0, -60, 0, 0.3, 0.0, 0.4)

#move_relative(0.2, 0, -0.5)
#
#move_joint_relative(-2, -90)
#
#move_relative(0.1, 0.1, 0)
#
#move_relative(-0.2, 0, 0)
#
#move_relative(0 , -0.2, 0)
#
#move_relative(0 , 0.2, 0)
#
#move_relative(0.2 , 0, 0)

robot.close()