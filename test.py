import numpy as np
from ctu_crs import CRS97

robot = CRS97()
robot.initialize()

q0 = robot.q_home
current_pose = robot.fk(robot.get_q())
current_pose[:3, 3] -= np.array([0.0, 0.0, 0.1])
ik_sols = robot.ik(current_pose)
assert len(ik_sols) > 0
closest_solution = min(ik_sols, key=lambda q: np.linalg.norm(q - q0))
robot.move_to_q(closest_solution)
robot.wait_for_motion_stop()
robot.close()