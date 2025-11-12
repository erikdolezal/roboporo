import cv2
from src.core.se3 import SE3
from src.core.so3 import SO3
import numpy as np
import os
from src.core.perception import find_hoop_homography, visualize_homography

class RobotInterface:
    def __init__(self, robot):
        self.robot = robot
        self.camera2robot_H = np.eye(3) if not os.path.exists("camera2robot_H.npy") else np.load("camera2robot_H.npy") 
        self.robot2hoop = SE3(translation=np.array([0.0, 0.0, 0.135]), 
                              rotation=SO3().from_euler_angles(np.deg2rad(np.array([0.0, 180.0, 0.0])), "xyz"))

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
                            rotation=SO3().from_euler_angles(np.deg2rad(np.array([phi, theta, psi])), "xyz")) * self.robot2hoop.inverse()
        q0 = self.robot.get_q()
        ik_sols = np.asarray(self.robot.ik(target_pose.homogeneous()))
        ik_sols_mask = np.all(ik_sols < self.robot.q_max, axis=1) & np.all(ik_sols > self.robot.q_min, axis=1) 
        ik_sols = ik_sols[ik_sols_mask]
        if len(ik_sols) > 0:
            closest_solution = min(ik_sols, key=lambda q: np.linalg.norm(q - q0))
            print("target_pose:\n", target_pose)
            self.robot.move_to_q(closest_solution)
            self.robot.wait_for_motion_stop()
            print("actual pose:\n", self.get_actual_pose())
        else:
            print("tos prestrelil miso")

    def get_actual_pose(self):
        q0 = self.robot.get_q()
        current_pose = self.robot.fk(q0) * self.robot2hoop
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
            [0.35, 0.0, 0.045],
            [0.35, 0.1, 0.045],
            [0.35, -0.1, 0.045],
            [0.38, 0.0, 0.045],
            [0.38, 0.1, 0.045],
            [0.38, -0.1, 0.045],
            [0.4, 0.0, 0.045],
            [0.4, 0.1, 0.045],
            [0.4, -0.1, 0.045],

        ])
        images = []
        real_positions = []
        for pos in target_positions:
            self.move_absolute(0., 180., 0., pos[0], pos[1], pos[2])
            img = self.robot.grab_image()
            images.append(img)
            actual_pose = self.get_actual_pose()
            real_positions.append({
                "translation_vector": actual_pose.translation.tolist()
            })

        self.camera2robot_H = find_hoop_homography(images, real_positions)
        print("Computed homography:\n", self.camera2robot_H)
        visualize_homography(images[0], self.camera2robot_H)
        np.save("camera2robot_H.npy", self.camera2robot_H)

    def get_maze_position(self):
        img = self.robot.grab_image()
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()

        # Create the ArUco detector
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        # Detect the markers
        corners, ids, _ = detector.detectMarkers(gray_img)   

        print("Detected markers:", ids)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(img, corners, ids)
            cv2.imshow('Detected Markers', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        

        