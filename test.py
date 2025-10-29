import numpy as np
from ctu_crs import CRS97

robot = CRS97()
robot.initialize()


def move_relative(dx, dy, dz):
    q0 = robot.get_q()
    current_pose = robot.fk(q0)
    current_pose[:3, 3] += np.array([dx, dy, dz])
    ik_sols = robot.ik(current_pose)
    assert len(ik_sols) > 0
    closest_solution = min(ik_sols, key=lambda q: np.linalg.norm(q - q0))
    robot.move_to_q(closest_solution)
    robot.wait_for_motion_stop()

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


move_relative(0.5, 0, -0.5)

move_joint_relative(-2, 15)

move_relative(0.2, 0.2,)

move_relative(-0.4, 0, 0)

move_relative(0 , -0.4, 0)

move_relative(0 , 0.4, 0)

move_relative(0.4 , 0, 0)

robot.close()