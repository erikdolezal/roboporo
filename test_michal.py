from src.core.obstacles import Obstacle
from ctu_crs import CRS97, CRS93
from src.interface.robot_interface import RobotInterface
from src.core.obstacles import Obstacle
from src.core.planning_michal import HoopPathOptimizer
import numpy as np
from src.core.se3 import SE3
import matplotlib.pyplot as plt

if __name__ == "__main__":
    robot = RobotInterface(CRS97(tty_dev=None))
    maze_position = SE3(translation=np.array([0.2, 0.3, 0.05]))
    obstacle = Obstacle("B", "src/tools/models", maze_position)
    obstacle.prep_obstacle()
    maze_waypoints = obstacle.waypoints
    planner = HoopPathOptimizer(robot, maze_waypoints, robot.hoop_fk, robot.fk)
    best_q_list = planner.get_list_of_best_q()

    # vizualize
    list_of_poses: list[SE3] = [robot.hoop_fk(q) for q in best_q_list]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 1. Vykreslení trajektorie (Váš kód)
    xs = [pose.translation[0] for pose in list_of_poses]
    ys = [pose.translation[1] for pose in list_of_poses]
    zs = [pose.translation[2] for pose in list_of_poses]
    ax.plot(xs, ys, zs, marker="o", linestyle="--", label="Trajektorie (Posun)")

    # 2. Vykreslení orientace (přidáno)
    axis_length = 0.1  # Délka os pro jednotlivé pózy

    for pose in list_of_poses:
        t_vec = pose.translation
        r_mat = pose.rotation.rot  # Získání rotační matice 3x3

        # Osa X (červená) - první sloupec r_mat
        ax.quiver(t_vec[0], t_vec[1], t_vec[2], r_mat[0, 0], r_mat[1, 0], r_mat[2, 0], color="r", length=axis_length)
        # Osa Y (zelená) - druhý sloupec r_mat
        ax.quiver(t_vec[0], t_vec[1], t_vec[2], r_mat[0, 1], r_mat[1, 1], r_mat[2, 1], color="g", length=axis_length)
        # Osa Z (modrá) - třetí sloupec r_mat
        ax.quiver(t_vec[0], t_vec[1], t_vec[2], r_mat[0, 2], r_mat[1, 2], r_mat[2, 2], color="b", length=axis_length)

    # 3. Nastavení grafu pro lepší zobrazení
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Vizualizace trajektorie SE(3) póz")
    ax.legend()

    # Důležité: Nastavení stejného měřítka os, aby nebyly rotace zkreslené
    # (Tohle je trochu 'hack' pro matplotlib 3D)
    max_range = np.array([max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)]).max() / 2.0
    mid_x = (max(xs) + min(xs)) * 0.5
    mid_y = (max(ys) + min(ys)) * 0.5
    mid_z = (max(zs) + min(zs)) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()
