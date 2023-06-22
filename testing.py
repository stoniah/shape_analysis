import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from analysis import get_centroid_size, get_trajectory_orientation_vector, get_angle
from reading_utils import read_contour_file

# contour_dir = Path('../../raw_data/Data_strain')

# contours = [read_contour_file(p) for p in sorted(contour_dir.iterdir()) if not p.stem.startswith('N')]
original_space_shape = np.array([63, 30, 9, 2])
print(original_space_shape)
print(np.product(original_space_shape[2:]))
ss_comps = np.load("space_shape_components.npy")
ts_comps = np.load("space_traj_components.npy")
ss_variance = np.load("space_shape_variance.npy")
ts_variance = np.load("space_traj_variance.npy")
print(np.linalg.norm(ss_comps[0].reshape(9, 2), axis=1))
print(np.linalg.norm(ts_comps[0].reshape(30, 4), axis=1))
print(ss_comps.shape)
print(ts_comps.shape)

raise ValueError()
original_space = np.load(
    Path(f"results/linear_shifted_gpa").joinpath("fully_aligned_shapes.npy")
)
shape_space = np.load(Path(f"results/linear_shifted_gpa").joinpath("shape_space.npy"))
ts = np.load(Path("results/linear_shifted_gpa/trajectory_space.npy"))
contour_names = np.load(Path("results/contour_names.npy"))
print(original_space.shape)
print(shape_space.shape)
print(ts.shape)
norm_array = np.char.startswith(contour_names, "N")

mean_norm_in_ss = np.mean(shape_space[np.where(norm_array)], axis=0)
mean_path_in_ss = np.mean(shape_space[np.where(np.logical_not(norm_array))], axis=0)

trajectory_orientations = [
    get_trajectory_orientation_vector(trajectory) for trajectory in shape_space
]
ss1_ss2_angle = [get_angle(vector[[0, 1]]) for vector in trajectory_orientations]
ss1_ss3_angle = [get_angle(vector[[0, 2]]) for vector in trajectory_orientations]
ss2_ss3_angle = [get_angle(vector[[1, 2]]) for vector in trajectory_orientations]
centroid_sizes = [get_centroid_size(trajectory) for trajectory in shape_space]

data = pd.DataFrame(
    {
        "names": contour_names,
        "is_norm": norm_array,
        "centroid_size": centroid_sizes,
        "ss1_ss2_angle": ss1_ss2_angle,
        "ss1_ss3_angle": ss1_ss3_angle,
        "ss2_ss3_angle": ss2_ss3_angle,
    }
)

for i in range(ts.shape[1]):
    data[f"TS{i + 1}"] = ts[:, i]

data.to_csv("data.csv")
print(data.head())

# plt.plot(*mean_norm_in_ss[:, :2].T, label='Mean norm', marker='o', zorder=1, color='green')
# plt.plot(*mean_path_in_ss[:, :2].T, label='Mean pathology', marker='o', zorder=1, color='blue')
# plt.scatter(*mean_norm_in_ss[0, :2].T, color='black', label='ED', zorder=2)
# plt.scatter(*mean_norm_in_ss[10, :2].T, color='red', label='ES', zorder=2)
# plt.scatter(*mean_path_in_ss[0, :2].T, color='black', zorder=2)
# plt.scatter(*mean_path_in_ss[10, :2].T, color='red', zorder=2)
# plt.legend()
# plt.xlabel('SS1')
# plt.ylabel('SS2')
# plt.show()

# print(len(contours))

# print(ts.shape)

# print(norm_array)
# data = pd.DataFrame(ts, columns=[f'TS{i + 1}' for i in range(ts.shape[1])])
# data['names'] = contour_names
# data['is_norm'] = norm_array
# print(data.head())
# # sns.catplot(x='is_norm', y='TS1', data=data, kind='box')
# # plt.show()
# # sns.catplot(x='is_norm', y='TS2', data=data, kind='box')
# # plt.show()
# # sns.catplot(x='is_norm', y='TS3', data=data, kind='box')
# # plt.show()
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, roc_auc_score
#
# X = ts[:,[0,1,2,3,4,5]]
# y = norm_array
# lg = LogisticRegression()
# lg.fit(X, y)
# y_pred = lg.predict(X)
# y_proba = lg.predict_proba(X)
# print(y_proba)
# data['prediction'] = y_pred
# data['prediction_proba_0'] = y_proba[:,0]
# data['prediction_proba_1'] = y_proba[:,1]
# print(classification_report(y, y_pred))
# print(roc_auc_score(y, y_pred))
# # data.to_csv('data.csv')
#
