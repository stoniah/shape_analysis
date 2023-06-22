# read files
# perform normalization and save
# get shape alignment. find global mean, linear shift, gpa. save all states. final is ls+gpa.
# pca on shape space. save everything, the matrix, the explained variances, pca coordinates of shapes
# extract features from shape space. Centroid size, angles, and gpa+pca on trajectories resulting in points.
#   save everything.
# Plots should be in a different script. I need to be able to plot everything I need without recalculating.

# What plots are the most interesting?
# 1) Min max of shape pca components
# 2) gifs of shape pca components
# 3) elbow plot of explained variances in shape pca
# 4) elbow plot of explained variances in trajectory pca
# 5) boxplot comparisons of different parameters. between different groups.
# 6) scatter plot of trajectory space (but it is unclear how to plot it since there are many components,
#   I can try tsne).
# I can also try tsne on shape pca.

from pathlib import Path
import joblib
import numpy as np
import reading_utils
import shape_analysis
from sklearn import decomposition

contour_dir = Path("../../raw_data/datastrain_2023")
result_dir = Path("may")
result_dir.mkdir(exist_ok=True)

raw_contours = reading_utils.load_data(contour_dir, only_norm=False)
scale=True
record_lengths = [record['coordinates'].shape[0] for record in raw_contours]
n_time_points = np.max(record_lengths)  # this and prev lines should be encapsulated into a method

n_harmonics = 6
contours = reading_utils.prepare_contours(
    raw_contours,
    scale=True,
    center=True,
    interpolate_time_points=n_time_points,
    fourier_interpolate_kwargs={"n_harmonics": n_harmonics, "n_points_out": None},
    close_contours=False,
)
np.save(result_dir.joinpath('prepared_contours.npy'), contours)

contour_names, shape_series = reading_utils.get_contour_tensor(
        contours, only_endo=True
    )
np.save(result_dir.joinpath("contour_names.npy"), contour_names)
np.save(result_dir.joinpath('shape_series.npy'), shape_series)

shapes = shape_series.reshape(-1, *shape_series.shape[2:])
np.save(result_dir.joinpath('shapes.npy'), shapes)

global_mean, gpa_shapes = shape_analysis.generalized_procrustes_analysis(
    shapes, scale=scale)
np.save(result_dir.joinpath('global_mean.npy'), global_mean)
np.save(result_dir.joinpath('gpa_shapes.npy'), gpa_shapes)

ls_shape_series = shape_analysis.linear_shift(
    shape_series, global_mean, scale=scale)
np.save(result_dir.joinpath('ls_shape_series.npy'), ls_shape_series)
ls_shapes = ls_shape_series.reshape(-1, *ls_shape_series.shape[2:])
np.save(result_dir.joinpath('ls_shapes.npy'), ls_shapes)

new_global_mean, lsgpa_shapes = shape_analysis.generalized_procrustes_analysis(
    ls_shapes, scale=scale)
np.save(result_dir.joinpath('lsgpa_global_mean.npy'), new_global_mean)
np.save(result_dir.joinpath('lsgpa_shapes.npy'), lsgpa_shapes)
lsgpa_shape_series = lsgpa_shapes.reshape(*shape_series.shape)
np.save(result_dir.joinpath('lsgpa_shape_series.npy'), lsgpa_shape_series)


n_components1 = 20
pca1 = decomposition.PCA(svd_solver='full', n_components=n_components1)
pca_shapes = pca1.fit_transform(lsgpa_shapes.reshape(lsgpa_shapes.shape[0], -1))
np.save(result_dir.joinpath('pca1_shapes.npy'), pca_shapes)
trajectories = pca_shapes.reshape(*lsgpa_shape_series.shape[:2], -1)
joblib.dump(pca1, result_dir.joinpath('pca1.joblib'))
np.save(result_dir.joinpath('trajectories.npy'), trajectories)


