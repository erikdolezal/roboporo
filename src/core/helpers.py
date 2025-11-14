import matplotlib.pyplot as plt
import numpy as np
import cv2

def visualize_homography(img, H: np.ndarray, real_positions=None, draw_extra=None):
    fig, ax = plt.subplots(1, 2, layout='constrained', figsize=(12, 10))
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0].grid()
    ax[0].set_title("Original Image")
    h, w = img.shape[:2]

    # warp the first image into the world plane defined by H and display it
    corners = np.array([[0, 0, 1],
                        [w, 0, 1],
                        [w, h, 1],
                        [0, h, 1]]).T
    world_corners = (H @ corners)
    world_corners /= world_corners[2:3, :]
    xs = world_corners[0, :]
    ys = world_corners[1, :]
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()

    scale = 1200.0  # pixels per world unit (adjust if necessary)
    out_w = max(1, int(np.ceil((xmax - xmin) * scale)))
    out_h = max(1, int(np.ceil((ymax - ymin) * scale)))

    # map world coordinates to pixel coordinates (flip Y so origin is bottom-left)
    T = np.array([[scale, 0.0, -xmin * scale],
                [0.0, scale,  -ymin * scale],
                [0.0, 0.0, 1.0]])
    M = T @ H

    warped = cv2.warpPerspective(img, M.astype(np.float64), (out_w, out_h))
    # convert BGR->RGB for matplotlib
    if warped.ndim == 3 and warped.shape[2] == 3:
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    if draw_extra is not None:
        draw_extra(ax)

    ax[1].imshow(warped, extent=(xmin, xmax, ymin, ymax), origin='lower')
    if real_positions is not None:
        positions = np.array([x["translation_vector"][:2] for x in real_positions])
        ax[1].plot(*positions.T, '.', c="red")
    ax[1].axis('equal')
    ax[1].grid()
    ax[1].set_title("Transformed Image")
    plt.show()

def project_homography(H, points):
    projected_points = np.hstack((points, np.ones((points.shape[0], 1)))) @ H.T
    return projected_points[:, :-1] / projected_points[:, -1:]