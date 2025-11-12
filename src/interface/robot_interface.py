import cv2
from src.core.se3 import SE3
from src.core.so3 import SO3
import numpy as np
import os
from src.core.perception import find_hoop_homography, visualize_homography, project_homography

class RobotInterface:
    def __init__(self, robot):
        self.robot = robot
        self.camera2robot_H = np.eye(3) if not os.path.exists("camera2robot_H.npy") else np.load("camera2robot_H.npy") 
        self.robot2hoop = SE3(translation=np.array([0.135, 0.0, 0.0]), 
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

    def hoop_ik(self, target_pose):
        flange_pose = target_pose * SE3(
                      rotation = self.robot2hoop.rotation, translation=-self.robot2hoop.translation).inverse()
        return np.asarray(self.robot.ik(flange_pose.homogeneous()))

    def move_absolute(self, phi, theta, psi, x, y, z):
        target_pose = SE3(translation = np.array([x, y, z]), 
                      rotation=SO3().from_euler_angles(np.deg2rad(np.array([phi, theta, psi])), "xyz"))
        q0 = self.robot.get_q()
        print("target_pose:\n", target_pose)
        ik_sols = self.hoop_ik(target_pose)
        ik_sols_mask = np.all(ik_sols < self.robot.q_max, axis=1) & np.all(ik_sols > self.robot.q_min, axis=1) 
        ik_sols = ik_sols[ik_sols_mask]
        if len(ik_sols) > 0:
            closest_solution = min(ik_sols, key=lambda q: np.linalg.norm(q - q0))
            self.robot.move_to_q(closest_solution)
            self.robot.wait_for_motion_stop()
            print("actual pose:\n", self.get_actual_pose())
            return True
        else:
            print("tos prestrelil miso")
            return False

    def get_actual_pose(self):
        q0 = self.robot.get_q()
        current_pose = SE3().from_homogeneous(self.robot.fk(q0))
        current_pose = current_pose * self.robot2hoop
        return current_pose

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
        x_positions = np.arange(0.35, 0.56, 0.05)
        y_positions = np.arange(-0.2, 0.21, 0.05)
        target_positions = np.array([[x, y, 0.05] for x in x_positions for y in y_positions])
        images = []
        real_positions = []
        for pos in target_positions:
            if self.move_absolute(0., 0., 0., pos[0], pos[1], pos[2]):
                img = self.robot.grab_image()
                images.append(img)
                actual_pose = self.get_actual_pose()
                real_positions.append({
                    "translation_vector": actual_pose.translation.tolist()
                })

        self.camera2robot_H = find_hoop_homography(images, real_positions)
        print("Computed homography:\n", self.camera2robot_H)
        visualize_homography(images[0], self.camera2robot_H, real_positions)
        np.save("camera2robot_H.npy", self.camera2robot_H)

    def get_maze_position(self):
        img = self.robot.grab_image()
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()

        # Create the ArUco detector
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        # Detect the markers
        corners, ids, _ = detector.detectMarkers(gray_img)   
        print(corners)
        projected_corners = [project_homography(self.camera2robot_H, corner_set[0]) for corner_set in corners]
        print(projected_corners)
        board_center = np.mean(np.vstack((projected_corners[0], projected_corners[1])), axis=0)
        diag_vec = np.mean(projected_corners[np.where((ids == 2).flatten())[0][0]], axis=0) - np.mean(projected_corners[np.where((ids == 1).flatten())[0][0]], axis=0)
        diag_angle = np.atan2(diag_vec[1], diag_vec[0])
        maze_pose = SE3(translation=np.array([*board_center, 0.04]), rotation=SO3().from_euler_angles(np.array([0, 0, diag_angle - np.pi/4]), "xyz"))
            
        #if ids is not None:
        #    cv2.aruco.drawDetectedMarkers(img, corners, ids)
        #    cv2.circle(img, center=project_homography(np.linalg.inv(self.camera2robot_H), board_center[None, :]).astype(int)[0], radius=50, color=(0, 255, 0), thickness=2)
        #visualize_homography(img, self.camera2robot_H)

        return maze_pose

        

        

        