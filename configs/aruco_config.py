import numpy as np

aruco_config = {
    "aruco_dict": "DICT_4X4_50",
    "marker_length": 0.04,  # Marker size in meters
    1: {
        "id": 1, 
        "position": np.array([-0.0375, -0.0375, 0.0])
        },
    2: {
        "id": 2, 
        "position": np.array([0.0375, 0.0375, 0.0])
        },
}

aruco_config[1]["corners"] = np.array(aruco_config[1]["position"] + np.array([[-(-1) ** (i//2) * aruco_config["marker_length"]/2, -(-1) ** ((i+1)//2) * aruco_config["marker_length"]/2, 0] for i in range(4)]))
aruco_config[2]["corners"] = np.array(aruco_config[2]["position"] + np.array([[-(-1) ** (i//2) * aruco_config["marker_length"]/2, -(-1) ** ((i+1)//2) * aruco_config["marker_length"]/2, 0] for i in range(4)]))