#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2025-09-21
#     Author: Martin Cífka <martin.cifka@cvut.cz>
#
from typing import List
from numpy.typing import ArrayLike
import numpy as np
import cv2  # noqa
from src.core.se3 import SE3
from src.core.so3 import SO3
import matplotlib.pyplot as plt
from src.core.helpers import project_homography, visualize_homography

VIS = False

def find_hoop_homography(images: ArrayLike, hoop_positions: List[dict]) -> np.ndarray:
    """
    Find homography based on images containing the hoop and the hoop positions loaded from
    the hoop_positions.json file in the following format:

    [{
        "RPY": [-0.0005572332585040621, -3.141058227474627, 0.0005185830258253442],
        "translation_vector": [0.5093259019899434, -0.17564068853313258, 0.04918733225140541]
    },
    {
        "RPY": [-0.0005572332585040621, -3.141058227474627, 0.0005185830258253442],
        "translation_vector": [0.5093569397977782, -0.08814069881074972, 0.04918733225140541]
    },
    ...
    ]
    """

    images = np.asarray(images)

    dest_points = np.array([x["translation_vector"][:2] for x in hoop_positions])
    src_points = []
    found_hoop_idx = []
    for i, img in enumerate(images):
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        edges = cv2.Canny(gray_blur, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours:
            area = cv2.contourArea(c)
            if area < 100:
                continue
            perimeter = cv2.arcLength(c, True)
            circularity = 4 * np.pi * (area/(perimeter*perimeter))
            if 0.7 < circularity <= 1.2:
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                if radius > 10:
                    center = (int(x), int(y))
                    src_points.append(center)
                    found_hoop_idx.append(i)
                    if VIS:
                        cv2.circle(img, center, int(radius), (0, 255, 0), 2)
                        cv2.circle(img, center, 2, (0, 0, 255), 3)            
                    break     
        if VIS:
            img = cv2.resize(img, None, fx=0.5, fy=0.5)
            cv2.imshow("Detected Circles", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    dest_points = dest_points[found_hoop_idx]
    src_points = np.array(src_points)
    dest_points = np.array(dest_points)
    homography, mask = cv2.findHomography(src_points, dest_points, cv2.RANSAC, 0.001)

    print(mask)
    return homography, src_points, dest_points, mask

def find_hoop_positions(images):
    images = np.asarray(images)

    src_points = []
    found_hoop_idx = []
    for i, img in enumerate(images):
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        edges = cv2.Canny(gray_blur, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours:
            area = cv2.contourArea(c)
            if area < 100:
                continue
            perimeter = cv2.arcLength(c, True)
            circularity = 4 * np.pi * (area/(perimeter*perimeter))
            if 0.7 < circularity <= 1.2:
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                if radius > 10:
                    center = (int(x), int(y))
                    src_points.append(center)
                    found_hoop_idx.append(i)
                    if VIS:
                        cv2.circle(img, center, int(radius), (0, 255, 0), 2)
                        cv2.circle(img, center, 2, (0, 0, 255), 3)            
                    break     
        if VIS:
            img = cv2.resize(img, None, fx=0.5, fy=0.5)
            cv2.imshow("Detected Circles", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    return np.array(src_points), found_hoop_idx

def optimize_homography_yaw_error(robot, src_positions, q_positions):
    img = robot.grab_image()
    fig, ax = plt.subplots(1, 2, layout='constrained', figsize=(12, 10))
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img_point_line, = ax[0].plot([], [], '.', c="red")
    img_fk_inliers, = ax[0].plot([], [], '.', c="green")
    img_fk_outliers, = ax[0].plot([], [], '.', c="black")
    ax[0].draw_artist(img_point_line)
    ax[0].draw_artist(img_fk_inliers)
    ax[0].draw_artist(img_fk_outliers)
    ax[0].grid()
    ax[0].set_title("Original Image")
    fig.canvas.blit(ax[0].bbox)

    ax[1].axis('equal')
    ax[1].set_xlim(0.3, 0.6)
    ax[1].set_ylim(-0.3, 0.3)
    world_point_line, = ax[1].plot([], [], '.', c="red")
    world_h_inliers, = ax[1].plot([], [], '.', c="green")
    world_h_outliers, = ax[1].plot([], [], '.', c="black")
    ax[1].draw_artist(world_point_line)
    ax[1].draw_artist(world_h_inliers)
    ax[1].draw_artist(world_h_outliers)
    ax[1].grid()
    ax[1].set_title("Transformed Image")
    fig.canvas.blit(ax[1].bbox)

    fig.canvas.flush_events()
    plt.pause(0.0001)

    img_background = fig.canvas.copy_from_bbox(ax[0].bbox)
    world_background = fig.canvas.copy_from_bbox(ax[1].bbox)


    init_transform = robot.camera2robot_H.copy()
    best_hoop_transform = init_transform.copy()
    best_homography = None
    best_inliers = np.zeros(src_positions.shape[0], dtype=bool)

    for yaw_angle in np.linspace(-10, 10, 100):
        robot2hoop = SE3(translation=np.array([-0.135, 0.0, 0.0]), rotation=SO3().from_euler_angles(np.deg2rad(np.array([np.deg2rad(yaw_angle), 180.0])), "zy"))
        robot.robot2hoop = robot2hoop
        fk_positions = np.array([transform.translation[:2] for transform in [robot.hoop_fk(q) for q in q_positions]])

        H, mask = cv2.findHomography(src_positions, fk_positions, cv2.RANSAC, 0.001)

        if np.sum(mask) > np.sum(best_inliers):
            best_inliers = mask.flatten().astype(bool)
            best_homography = H
            best_hoop_transform = robot.robot2hoop.copy()
            print(f"New best yaw: {yaw_angle} with {np.sum(best_inliers)} inliers")
        # Update plots
        fig.canvas.restore_region(img_background)
        img_point_line.set_data(*src_positions.T)
        mask = mask.flatten().astype(bool)

        i2w = project_homography(H, src_positions)

        img_fk_inliers.set_data(*i2w[mask].T)
        img_fk_outliers.set_data(*i2w[~mask].T)
        ax[0].draw_artist(img_point_line)
        ax[0].draw_artist(img_fk_inliers)
        ax[0].draw_artist(img_fk_outliers)
        fig.canvas.blit(ax[0].bbox)

        w2i = project_homography(np.linalg.inv(H), fk_positions)

        fig.canvas.restore_region(world_background)
        world_point_line.set_data(*fk_positions.T)
        world_h_inliers.set_data(*w2i[mask].T)
        world_h_outliers.set_data(*w2i[~mask].T)
        ax[1].draw_artist(world_point_line)
        ax[1].draw_artist(world_h_inliers)
        ax[1].draw_artist(world_h_outliers)
        fig.canvas.blit(ax[1].bbox)

        fig.canvas.flush_events()
    
    plt.close(fig)

    robot.robot2hoop = best_hoop_transform
    print("best hoop transform: ", robot.robot2hoop)
    robot.camera2robot_H = best_homography

    dest_points = np.array([transform.translation[:2] for transform in [robot.hoop_fk(q) for q in q_positions]])

    def draw_extra(ax):
        projected_points = project_homography(robot.camera2robot_H, np.array(src_positions))
        projected_positions = project_homography(np.linalg.inv(robot.camera2robot_H), dest_points)
        ax[0].plot(*src_positions.T, 'o', c="red")
        ax[0].plot(*projected_positions[~best_inliers].T, 'o', c="black")
        ax[0].plot(*projected_positions[best_inliers].T, 'o', c="green")
        ax[1].plot(*dest_points.T, 'o', c="red")
        ax[1].plot(*projected_points[~best_inliers].T, 'o', c="black")
        ax[1].plot(*projected_points[best_inliers].T, 'o', c="green")

    visualize_homography(img, robot.camera2robot_H, draw_extra=draw_extra)
    np.save("camera2robot_H.npy", robot.camera2robot_H)
    np.save("robot2hoop_SE3.npy", robot.robot2hoop.as_homogeneous())

        





