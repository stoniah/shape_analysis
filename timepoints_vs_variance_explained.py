from pathlib import Path
from sklearn.decomposition import PCA

from reading_utils import prepare_contours, load_data, get_contour_tensor
import shape_analysis
from plots import (
    plot_contours,
    plot_areas,
    pca_animation1,
    plot_trajectory_animation,
    plot_trajectory_animation_diff,
)
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

import numpy as np

if __name__ == "__main__":
    data = []
    for n_time_points in np.arange(2, 30, 1):
        contour_dir = Path("../../raw_data/Data_strain")
        n_harmonics = 6
        contours = prepare_contours(
            load_data(contour_dir, only_norm=False),
            scale=True,
            center=True,
            interpolate_time_points=None,
            fourier_interpolate_kwargs=None,
            close_contours=False,
        )
        contours_time_interpolated = prepare_contours(
            contours,
            scale=False,
            center=False,
            interpolate_time_points=n_time_points,
            fourier_interpolate_kwargs=None,
            close_contours=False,
        )
        contours_fourier_interpolated = prepare_contours(
            contours_time_interpolated,
            scale=False,
            center=False,
            interpolate_time_points=None,
            fourier_interpolate_kwargs={
                "n_harmonics": n_harmonics,
                "n_points_out": None,
            },
            close_contours=False,
        )

        contour_names, shape_series = get_contour_tensor(
            contours_fourier_interpolated, only_endo=True
        )
        shapes = shape_series.reshape(-1, *shape_series.shape[2:])

        scale = True
        global_mean, gpa_shapes = shape_analysis.generalized_procrustes_analysis(
            shapes, scale=scale
        )

        gpa_shape_series = gpa_shapes.reshape(*shape_series.shape)
        ls_shape_series = shape_analysis.linear_shift(
            shape_series, global_mean, scale=scale
        )
        ls_shapes = ls_shape_series.reshape(-1, *ls_shape_series.shape[2:])

        new_global_mean, lsgpa_shapes = shape_analysis.generalized_procrustes_analysis(
            ls_shapes, scale=scale
        )
        lsgpa_shape_series = lsgpa_shapes.reshape(*shape_series.shape)

        shapes_list = [shapes, gpa_shapes, ls_shapes, lsgpa_shapes]
        shape_series_list = [
            shape_series,
            gpa_shape_series,
            ls_shape_series,
            lsgpa_shape_series,
        ]

        aligned_shapes = lsgpa_shape_series
        bounds = np.array(
            [
                np.min(aligned_shapes, axis=(0, 1, 2)),
                np.max(aligned_shapes, axis=(0, 1, 2)),
            ]
        )

        n_components = 5
        pca1 = PCA(svd_solver="full", n_components=n_components)
        trajectories = pca1.fit_transform(
            aligned_shapes.reshape(np.product(aligned_shapes.shape[:2]), -1)
        ).reshape(*aligned_shapes.shape[:2], -1)
        print(n_time_points)
        data.append(pca1.explained_variance_ratio_)
        # plt.plot(pca1.explained_variance_ratio_, marker='o', label=f'{n_time_points}')
        # plt.plot(np.cumsum(pca1.explained_variance_ratio_), marker='o', label=f'{n_time_points}')
        # plt.ylim(0, 1)
        # plt.xlabel('n_components')
        # plt.ylabel('Explained variance ratio')
        # plt.title('Shape space components explained variance ratio')
        (
            mean_trajectory,
            trajectories_gpa,
        ) = shape_analysis.generalized_procrustes_analysis(trajectories, scale=scale)
        n_components2 = 6
        pca2 = PCA(svd_solver="full", n_components=n_components2)
        final_space = pca2.fit_transform(
            trajectories_gpa.reshape(trajectories_gpa.shape[0], -1)
        )
        plt.plot(
            np.cumsum(pca2.explained_variance_ratio_)
            * np.sum(pca1.explained_variance_ratio_),
            marker="o",
            label=f"{n_time_points}",
        )
        plt.ylim(0, 1)
        plt.xlabel("n_components")
        plt.ylabel("Explained variance ratio")
    plt.title("Trajectory space components explained variance ratio")
    plt.legend()
    plt.show()
    # np.save('pca1_vs_timepoints', data)
