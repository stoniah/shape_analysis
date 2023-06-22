from pathlib import Path
import numpy as np
import pandas as pd


def get_centroid_size(shape):
    """
    :param shape: npoints,ndims
    :return: float
    """
    centroid = np.mean(shape, axis=0, keepdims=True)
    landmark_distances = shape - centroid
    centroid_size = np.sqrt(np.sum(landmark_distances**2))
    return centroid_size


def get_es_id(ntimepoints, es_time_fraction=0.35):
    return np.argmin(np.abs(es_time_fraction - np.linspace(0, 1, ntimepoints)))


def get_trajectory_orientation_vector(trajectory, es_id=10):
    """
    :param trajectory: npoints,ndims
    :return:
    """
    return trajectory[es_id] - trajectory[0]


def get_angle(vector):
    """
    :param vector: 2,
    :return:
    """
    return np.arctan2(vector[1], vector[0])


if __name__ == "__main__":
    path = Path("results_norm/linear_shifted_gpa")
    shape_space = np.load(path.joinpath("shape_space.npy"))
    print("shape space shape =", shape_space.shape)
    trajectory_space = np.load(path.joinpath("trajectory_space.npy"))
    print("trajectory space shape =", trajectory_space.shape)
    shape_gpa = np.load(path.joinpath("fully_aligned_shapes.npy"))
    contour_names = np.load("results_norm/contour_names.npy")
    c_sizes = [get_centroid_size(shape) for shape in shape_space]
    es_id = get_es_id(shape_space.shape[1], 0.35)
    trajectory_orientations = [
        get_trajectory_orientation_vector(shape, es_id) for shape in shape_space
    ]
    pc1_2_angles = [get_angle(vector[[0, 1]]) for vector in trajectory_orientations]
    pc1_3_angles = [get_angle(vector[[0, 2]]) for vector in trajectory_orientations]
    pc1_4_angles = [get_angle(vector[[0, 3]]) for vector in trajectory_orientations]
    table = pd.read_excel(
        Path("../../raw_data/Strain_Алмаз.xlsx"), sheet_name=1
    ).set_index("ID")
    table["CS"] = 0.0
    table["TS_1"] = 0.0
    table["TS_2"] = 0.0
    table["TS_3"] = 0.0
    table["TS_4"] = 0.0
    table["TS_5"] = 0.0
    table["TS_6"] = 0.0
    table["angle1_2"] = 0.0
    table["angle1_3"] = 0.0
    table["angle1_4"] = 0.0
    for i, idx in enumerate(contour_names):
        if idx not in table.index:
            continue
        table.at[idx, "CS"] = c_sizes[i]
        for j in range(trajectory_space.shape[1]):
            table.at[idx, f"TS_{j+1}"] = trajectory_space[i][j]
        table.at[idx, "angle1_2"] = pc1_2_angles[i]
        table.at[idx, "angle1_3"] = pc1_3_angles[i]
        table.at[idx, "angle1_4"] = pc1_4_angles[i]
    table.to_excel("Strain_with_trajectory_features.xlsx")
    # print(len(pc1_2_angles))
    # print(len(table))
    # print(contour_names)
    # print(table)
