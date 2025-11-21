import cv2
from src.core.se3 import SE3
from src.core.so3 import SO3
import numpy as np
import os
from src.core.perception import find_hoop_homography
from src.core.helpers import visualize_homography, project_homography
from configs.aruco_config import aruco_config


class RobotInterface:
    def __init__(self, robot):
        self.robot = robot
        self.camera2robot_H = np.eye(3) if not os.path.exists("camera2robot_H.npy") else np.load("camera2robot_H.npy")
        self.robot2hoop = SE3(translation=np.array([-0.135, 0.0, 0.0]), rotation=SO3().from_euler_angles(np.deg2rad(np.array([0.0, 180.0, 0.0])), "xyz"))

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

    def follow_q_list(self, list_q):

        for q in list_q:
            self.robot.move_to_q(q)
            self.robot.wait_for_motion_stop()

    def hoop_ik(self, target_pose):
        flange_pose = target_pose * self.robot2hoop.inverse()
        return np.asarray(self.robot.ik(flange_pose.homogeneous()))

    def move_absolute(self, phi, theta, psi, x, y, z):
        target_pose = SE3(translation=np.array([x, y, z]), rotation=SO3().from_euler_angles(np.deg2rad(np.array([phi, theta, psi])), "xyz"))
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

    def hoop_fk(self, q):
        current_pose = SE3().from_homogeneous(self.robot.fk(q))
        current_pose = current_pose * self.robot2hoop
        return current_pose

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
        y_positions = np.arange(-0.2, 0.18, 0.04)
        target_positions = np.array([[x, y, 0.045] for x in x_positions for y in y_positions])
        images = []
        real_positions = []
        for pos in target_positions:
            if self.move_absolute(0.0, 0.0, 0.0, pos[0], pos[1], pos[2]):
                img = self.robot.grab_image()
                images.append(img)
                actual_pose = self.get_actual_pose()
                real_positions.append({"translation_vector": actual_pose.translation.tolist()})

        self.camera2robot_H = find_hoop_homography(images, real_positions)
        print("Computed homography:\n", self.camera2robot_H)
        visualize_homography(images[0], self.camera2robot_H, real_positions=np.array([x["translation_vector"][:2] for x in real_positions]))
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
        assert ids is not None, "No aruco found"

        target_corners = np.array([project_homography(self.camera2robot_H, corner_set[0]) for corner_set in corners]).reshape(-1, 2)
        target_corners = np.hstack((target_corners, 0.035 * np.ones((target_corners.shape[0], 1))))

        aruco_corners = np.array([aruco_config[id[0]]["corners"] for id in ids]).reshape(-1, 3) if ids is not None else np.array([[]])

        target_mean = np.mean(target_corners, axis=0) if target_corners.shape[0] > 0 else np.array([0, 0, 0])
        aruco_mean = np.mean(aruco_corners, axis=0) if aruco_corners.shape[0] > 0 else np.array([0, 0, 0])

        target_corners_centered = target_corners - target_mean
        aruco_corners_centered = aruco_corners - aruco_mean

        H = aruco_corners_centered.T @ target_corners_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T

        t = target_mean - R @ aruco_mean

        maze_pose = SE3(translation=t, rotation=SO3(R))

        aligned_aruco_positions = np.array([aruco_config[id]["corners"] for id in [1, 2]]).reshape(-1, 3) @ R.T + t[None, :]

        def draw_extra(ax):
            ax[1].plot([t[0], t[0] + 0.05 * R[0, 0]], [t[1], t[1] + 0.05 * R[1, 0]], c="r")
            ax[1].plot([t[0], t[0] + 0.05 * R[0, 1]], [t[1], t[1] + 0.05 * R[1, 1]], c="g")

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(img, corners, ids)
            cv2.circle(img, center=project_homography(np.linalg.inv(self.camera2robot_H), maze_pose.translation[None, :2]).astype(int)[0], radius=50, color=(0, 255, 0), thickness=2)
        print(maze_pose)
        visualize_homography(img, self.camera2robot_H, draw_extra=draw_extra, real_positions=aligned_aruco_positions[:, :2])

        return maze_pose * SE3(rotation=SO3().from_euler_angles(np.array([np.pi / 2]), "z"), translation=np.zeros(3))
