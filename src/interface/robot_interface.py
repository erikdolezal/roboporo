import cv2
from src.core.se3 import SE3
from src.core.so3 import SO3
import numpy as np
from src.core.perception import find_hoop_homography, visualize_homography

class RobotInterface:
    def __init__(self, robot):
        self.robot = robot
        self.camera2robot_H = np.eye(3)

    def __getattr__(self, name):
        return getattr(self.robot, name)

    def move_relative(self, dx, dy, dz):
        q0 = self.robot.get_q()
        current_pose = self.robot.fk(q0)
        current_pose[:3, 3] += np.array([dx, dy, dz])
        ik_sols = self.robot.ik(current_pose)
        print(current_pose)
        if len(ik_sols) > 0:
            closest_solution = min(ik_sols, key=lambda q: np.linalg.norm(q - q0))
            self.robot.move_to_q(closest_solution)
            self.robot.wait_for_motion_stop()
        else:
            print("tos prestrelil miso")

    def move_absolute(self, phi, theta, psi, x, y, z):
        target_pose = SE3(translation = np.array([x, y, z]), 
                            rotation=SO3().from_euler_angles(np.deg2rad(np.array([phi, theta, psi])), "xyz"))
        q0 = self.robot.get_q()
        ik_sols = self.robot.ik(target_pose.homogeneous())
        if len(ik_sols) > 0:
            closest_solution = min(ik_sols, key=lambda q: np.linalg.norm(q - q0))
            self.robot.move_to_q(closest_solution)
            self.robot.wait_for_motion_stop()
            print("target_pose:\n", target_pose)
            print("actual pose:\n", self.get_actual_pose())
        else:
            print("tos prestrelil miso")

    def get_actual_pose(self):
        q0 = self.robot.get_q()
        current_pose = self.robot.fk(q0)
        return SE3().from_homogeneous(current_pose)

    def move_joint_relative(self, joint_index, delta_angle_deg):
        q0 = self.robot.get_q()
        q = self.robot.get_q()
        q[joint_index] += np.deg2rad(delta_angle_deg)
        assert self.robot.in_limits(q)
        desired_pose = self.robot.fk(q)
        ik_sols = self.robot.ik(desired_pose)
        assert len(ik_sols) > 0
        closest_solution = min(ik_sols, key=lambda q: np.linalg.norm(q - q0))
        self.robot.move_to_q(closest_solution)
        self.robot.wait_for_motion_stop()

    def calibrate_camera(self):
        target_positions = np.array([
            [0.4, 0.0, 0.2],
            [0.4, 0.2, 0.2],
            [0.4, -0.2, 0.2],
            [0.3, 0.0, 0.2],
            [0.3, 0.2, 0.2],
            [0.3, -0.2, 0.2],
        ])
        images = []
        real_positions = []
        for pos in target_positions:
            self.move_absolute(0, 0, 0, pos[0], pos[1], pos[2])
            img = self.robot.grab_image()
            images.append(img)
            actual_pose = self.get_actual_pose()
            real_positions.append({
                "RPY": actual_pose.rotation.to_euler_angles("xyz").tolist(),
                "translation_vector": actual_pose.translation.tolist()
            })

        self.camera2robot_H = find_hoop_homography(images, real_positions)
        print("Computed homography:\n", self.camera2robot_H)
        visualize_homography(images[0], self.camera2robot_H)
