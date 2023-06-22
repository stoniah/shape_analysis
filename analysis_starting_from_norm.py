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


def explained_variance(model, X):
    result = np.zeros(model.n_components)
    for ii in range(model.n_components):
        X_trans = model.transform(X)
        X_trans_ii = np.zeros_like(X_trans)
        X_trans_ii[:, ii] = X_trans[:, ii]
        X_approx_ii = model.inverse_transform(X_trans_ii)

        result[ii] = (
            1 - (np.linalg.norm(X_approx_ii - X) / np.linalg.norm(X - model.mean_)) ** 2
        )
    return result


def get_shape_space_from_trajectory_space(
    trajectory_space_vectors, trajectory_space_map, shape_space_shape
):
    new_shape = (
        (len(trajectory_space_vectors), *shape_space_shape)
        if trajectory_space_vectors.ndim > 1
        else shape_space_shape
    )
    shape_space_vectors = trajectory_space_map.inverse_transform(
        trajectory_space_vectors
    ).reshape(*new_shape)
    return shape_space_vectors


def get_original_space_from_shape_space(
    shape_space_vectors, shape_space_map, original_space_shape
):
    new_shape = (
        (len(shape_space_vectors), *original_space_shape)
        if shape_space_vectors.ndim > 2
        else original_space_shape
    )
    original_space_vectors = shape_space_map.inverse_transform(
        shape_space_vectors
    ).reshape(*new_shape)
    return original_space_vectors


def get_original_space_from_trajectory_space(
    trajectory_space_vectors,
    trajectory_space_map,
    shape_space_shape,
    shape_space_map,
    original_space_shape,
):
    shape_space_vectors = get_shape_space_from_trajectory_space(
        trajectory_space_vectors, trajectory_space_map, shape_space_shape
    )
    print("shape_space_vectors", shape_space_vectors.shape)
    original_space_vectors = get_original_space_from_shape_space(
        shape_space_vectors, shape_space_map, original_space_shape
    )
    return original_space_vectors


if __name__ == "__main__":
    contour_dir = Path("../../raw_data/Results2")
    n_time_points = 30
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
        fourier_interpolate_kwargs={"n_harmonics": n_harmonics, "n_points_out": None},
        close_contours=False,
    )
    start_with_norm = True

    print("n_contours=", len(contours))
    result_dir = Path("results_from_norm")
    # result_dir = Path('results_from_norm_wolastscale')
    # result_dir = Path('results_together')
    result_dir.mkdir(exist_ok=True)
    data_dir = result_dir.joinpath("data")
    data_dir.mkdir(exist_ok=True)
    np.save(data_dir.joinpath("contours"), contours)
    np.save(data_dir.joinpath("contours_time_interpolated"), contours_time_interpolated)
    np.save(
        data_dir.joinpath("contours_fourier_interpolated"),
        contours_fourier_interpolated,
    )

    # contour_graphs_dir = Path('results/contour_graphs_3')
    # contour_graphs_dir.mkdir(parents=True, exist_ok=True)
    # plot_contours(contour_graphs_dir, contours, contours_time_interpolated, contours_fourier_interpolated, n_harmonics)
    #
    # area_lists = [[[Polygon(time).area for time in patient['coordinates']] for patient in c] for c in
    #               (contours, contours_time_interpolated, contours_fourier_interpolated)]
    # names = ('', '_time_interpolated', '_fourier_interpolated')
    # for i in range(3):
    #     np.save('results/data/' + 'areas' + names[i], area_lists[i])
    # area_graph_dir = Path('results/area_graphs')
    # plot_areas(area_graph_dir, contours, *area_lists)

    contour_names, shape_series = get_contour_tensor(
        contours_fourier_interpolated, only_endo=True
    )
    print("contour_tensor", len(contour_names), shape_series.shape)
    np.save(result_dir.joinpath("contour_names.npy"), contour_names)
    shapes = shape_series.reshape(-1, *shape_series.shape[2:])

    scale = True
    norm_check = np.zeros(shape=len(shapes))
    idx = 0
    # for i, name in enumerate(contour_names):
    #     val = 1 if 'norm' in name else 0
    #     for j in range(shape_series[i].shape[0]):
    #         norm_check[idx + j] = val
    #     idx += shape_series[i].shape[0]
    norm_check = [1 if "norm" in name else 0 for name in contour_names]
    norm_shape_series = shape_series[np.where(norm_check)]
    norm_shapes = norm_shape_series.reshape(-1, *norm_shape_series.shape[2:])
    print(norm_check, len(norm_shapes))
    print(contour_names)
    if start_with_norm:
        global_mean, norm_gpa_shapes = shape_analysis.generalized_procrustes_analysis(
            norm_shapes, scale=scale
        )
        pathology_shape_series = shape_series[np.where(np.logical_not(norm_check))]
        pathology_shapes = pathology_shape_series.reshape(
            -1, *pathology_shape_series.shape[2:]
        )
        pathology_gpa_shapes = [
            shape_analysis.procrustes_analysis(shape, global_mean)[1]
            for shape in pathology_shapes
        ]
        pathology_gpa_shapes = np.array(pathology_gpa_shapes)
        norm_gpa_shape_series = np.array(norm_gpa_shapes).reshape(
            *norm_shape_series.shape
        )
        pathology_gpa_shape_series = pathology_gpa_shapes.reshape(
            *pathology_shape_series.shape
        )
        gpa_shape_series = np.zeros_like(shape_series)
        gpa_shape_series[np.where(norm_check)] = norm_gpa_shape_series[:]
        gpa_shape_series[
            np.where(np.logical_not(norm_check))
        ] = pathology_gpa_shape_series[:]
        gpa_shapes = gpa_shape_series.reshape(*shapes.shape)
    else:
        global_mean, gpa_shapes = shape_analysis.generalized_procrustes_analysis(
            shapes, scale=scale
        )
        gpa_shape_series = gpa_shapes.reshape(*shape_series.shape)
        norm_gpa_shape_series = gpa_shape_series[np.where(norm_check)]
        norm_gpa_shapes = norm_gpa_shape_series.reshape(
            -1, *norm_gpa_shape_series.shape[2:]
        )
        pathology_gpa_shape_series = gpa_shape_series[
            np.where(np.logical_not(norm_check))
        ]
        pathology_gpa_shapes = pathology_gpa_shape_series.reshape(
            -1, *pathology_gpa_shape_series.shape[2:]
        )

    print(len(pathology_gpa_shapes), len(norm_gpa_shapes), len(shapes))
    print("norm_shapes", len(np.unique(norm_shapes, axis=0)))
    print("norm_gpa_shapes", len(np.unique(norm_gpa_shapes, axis=0)))
    print("pathology_gpa_shapes", len(np.unique(pathology_gpa_shapes, axis=0)))
    print("gpa_shapes", len(np.unique(gpa_shapes, axis=0)))
    np.save(result_dir.joinpath("gpa_shape_series.npy"), gpa_shape_series)

    ls_shape_series = shape_analysis.linear_shift(
        shape_series, global_mean, scale=scale
    )
    np.save(result_dir.joinpath("ls_shape_series.npy"), ls_shape_series)
    print(ls_shape_series.shape)
    ls_shapes = ls_shape_series.reshape(-1, *ls_shape_series.shape[2:])
    norm_ls_shape_series = ls_shape_series[np.where(norm_check)]
    norm_ls_shapes = norm_ls_shape_series.reshape(-1, *norm_ls_shape_series.shape[2:])
    pathology_ls_shape_series = ls_shape_series[np.where(np.logical_not(norm_check))]
    pathology_ls_shapes = pathology_ls_shape_series.reshape(
        -1, *pathology_ls_shape_series.shape[2:]
    )
    if start_with_norm:
        (
            new_global_mean,
            norm_lsgpa_shapes,
        ) = shape_analysis.generalized_procrustes_analysis(norm_ls_shapes, scale=scale)
        pathology_lsgpa_shapes = [
            shape_analysis.procrustes_analysis(shape, new_global_mean)[1]
            for shape in pathology_ls_shapes
        ]
        norm_lsgpa_shape_series = norm_lsgpa_shapes.reshape(*norm_ls_shape_series.shape)
        pathology_lsgpa_shape_series = np.array(pathology_lsgpa_shapes).reshape(
            *pathology_ls_shape_series.shape
        )
        lsgpa_shape_series = np.zeros_like(ls_shape_series)
        lsgpa_shape_series[np.where(norm_check)] = norm_lsgpa_shape_series
        lsgpa_shape_series[
            np.where(np.logical_not(norm_check))
        ] = pathology_lsgpa_shape_series
        lsgpa_shapes = lsgpa_shape_series.reshape(*ls_shapes.shape)
    else:
        new_global_mean, lsgpa_shapes = shape_analysis.generalized_procrustes_analysis(
            ls_shapes, scale=scale
        )
        lsgpa_shape_series = lsgpa_shapes.reshape(*shape_series.shape)
        norm_lsgpa_shape_series = lsgpa_shape_series[np.where(norm_check)]
        pathology_lsgpa_shape_series = lsgpa_shape_series[
            np.where(np.logical_not(norm_check))
        ]
        norm_lsgpa_shapes = norm_lsgpa_shape_series.reshape(
            -1, *norm_lsgpa_shape_series.shape[2:]
        )
        pathology_lsgpa_shapes = pathology_lsgpa_shape_series.reshape(
            -1, *pathology_lsgpa_shape_series.shape[2:]
        )

    print("lsgpa_shapes", len(np.unique(lsgpa_shapes, axis=0)), lsgpa_shapes.shape)
    print(
        "lsgpa_shape_series",
        len(np.unique(lsgpa_shape_series, axis=0)),
        lsgpa_shape_series.shape,
    )

    shapes_list = [shapes, gpa_shapes, ls_shapes, lsgpa_shapes]
    shape_series_list = [
        shape_series,
        gpa_shape_series,
        ls_shape_series,
        lsgpa_shape_series,
    ]
    names = [
        "raw",
        f'gpa{"" if scale else "_noscale"}',
        f'linear_shifted{"" if scale else "_noscale"}',
        f'linear_shifted_gpa{"" if scale else "_noscale"}',
    ]

    out_path = result_dir.joinpath(f"{names[3]}").joinpath("fully_aligned_shapes.npy")
    out_path.parent.mkdir(exist_ok=True)
    np.save(out_path, lsgpa_shape_series)
    aligned_shape_series = np.load(out_path)
    aligned_shapes = aligned_shape_series.reshape(-1, *aligned_shape_series.shape[2:])
    print(aligned_shape_series.shape)
    bounds = np.array(
        [
            np.min(aligned_shape_series, axis=(0, 1, 2)),
            np.max(aligned_shape_series, axis=(0, 1, 2)),
        ]
    )
    # raise ValueError
    # out_dir = Path(result_dir.joinpath(f'{names[3]}/cycles/'))
    # out_dir.mkdir(parents=True, exist_ok=True)
    # for i, trajectory in enumerate(aligned_shape_series):
    #     name = contour_names[i]
    #     out_path = out_dir.joinpath(f'{contour_names[i]}_cycle.gif')
    #     plot_trajectory_animation(trajectory.transpose(0, 2, 1), out_path, name, bounds, 40)

    n_components = 8
    pca1 = PCA(svd_solver="full", n_components=n_components)
    norm_aligned_shape_series = lsgpa_shape_series[np.where(norm_check)]
    print(norm_aligned_shape_series.shape)
    norm_aligned_shapes = norm_aligned_shape_series.reshape(
        -1, *norm_aligned_shape_series.shape[2:]
    )
    print(
        "norm_aligned_shapes",
        len(np.unique(norm_aligned_shapes, axis=0)),
        norm_aligned_shapes.shape,
    )

    pca1.fit(aligned_shapes.reshape(aligned_shapes.shape[0], -1))
    np.save(
        result_dir.joinpath(f"{names[3]}/pca1_explained_variance_ratio.npy"),
        pca1.explained_variance_ratio_,
    )

    trajectories = pca1.transform(
        aligned_shape_series.reshape(np.product(aligned_shape_series.shape[:2]), -1)
    ).reshape(*lsgpa_shape_series.shape[:2], -1)
    print("trajectories", len(np.unique(trajectories, axis=0)), trajectories.shape)
    np.save(result_dir.joinpath(f"{names[3]}/shape_space.npy"), trajectories)
    plt.plot(pca1.explained_variance_ratio_, marker="o")
    plt.ylim(0, 1)
    plt.xlabel("n_components")
    plt.ylabel("Explained variance ratio")
    plt.title("Shape space components explained variance ratio")
    plt.savefig(result_dir.joinpath(f"{names[3]}/shape_space_elbow_plot.png"))
    plt.close()
    # last_scale = False
    last_scale = True

    if start_with_norm:
        (
            mean_trajectory,
            norm_trajectories_gpa,
        ) = shape_analysis.generalized_procrustes_analysis(
            trajectories[np.where(norm_check)], scale=last_scale
        )
        pathology_trajectories_gpa = [
            shape_analysis.procrustes_analysis(
                trajectory, mean_trajectory, scale=last_scale
            )[1]
            for trajectory in trajectories[np.where(np.logical_not(norm_check))]
        ]
    else:
        (
            mean_trajectory,
            trajectories_gpa,
        ) = shape_analysis.generalized_procrustes_analysis(
            trajectories, scale=last_scale
        )
        norm_trajectories_gpa = trajectories_gpa[np.where(norm_check)]
        pathology_trajectories_gpa = trajectories_gpa[
            np.where(np.logical_not(norm_check))
        ]
    trajectories_gpa = np.zeros_like(trajectories)
    trajectories_gpa[np.where(norm_check)] = norm_trajectories_gpa
    trajectories_gpa[np.where(np.logical_not(norm_check))] = pathology_trajectories_gpa
    np.save(
        result_dir.joinpath(f"{names[3]}/shape_space_after_gpa.npy"), trajectories_gpa
    )
    print("trajectories_gpa", len(np.unique(trajectories_gpa, axis=0)))
    n_components2 = 8
    pca2 = PCA(svd_solver="full", n_components=n_components2)
    if start_with_norm:
        pca2.fit(
            trajectories_gpa.reshape(trajectories_gpa.shape[0], -1)[
                np.where(norm_check)
            ]
        )
    else:
        pca2.fit(trajectories_gpa.reshape(trajectories_gpa.shape[0], -1))
    print("pca2", pca2.explained_variance_ratio_)
    np.save(
        result_dir.joinpath(f"{names[3]}/pca2_explained_variance_ratio.npy"),
        pca2.explained_variance_ratio_,
    )
    final_space = pca2.transform(
        trajectories_gpa.reshape(trajectories_gpa.shape[0], -1)
    )
    np.save(result_dir.joinpath(f"{names[3]}/trajectory_space.npy"), final_space)
    plt.plot(pca2.explained_variance_ratio_, marker="o")
    plt.ylim(0, 1)
    plt.xlabel("n_components")
    plt.ylabel("Explained variance ratio")
    plt.title("Trajectory space components explained variance ratio")
    plt.savefig(result_dir.joinpath(f"{names[3]}/trajectory_space_elbow_plot.png"))
    plt.close()

    np.save(result_dir.joinpath("space_shape_components.npy"), pca1.components_)
    np.save(
        result_dir.joinpath("space_shape_variance.npy"), pca1.explained_variance_ratio_
    )
    np.save(result_dir.joinpath("space_traj_components.npy"), pca2.components_)
    np.save(
        result_dir.joinpath("space_traj_variance.npy"), pca2.explained_variance_ratio_
    )
    # raise ValueError
    raise ValueError
    from sklearn.cluster import KMeans
    from collections import defaultdict

    pairs = defaultdict(dict)
    for i, name in enumerate(contour_names):
        patient_id, operation_status = name.split("_")
        if operation_status == "np":
            pairs[patient_id]["before"] = i
        elif operation_status == "0m":
            pairs[patient_id]["after"] = i
    bad_pairs = []
    pairs = {
        key: pairs[key]
        for key in pairs
        if ("after" in pairs[key]) and ("before" in pairs[key])
    }

    before_operation_index = [
        i for i, name in enumerate(contour_names) if name.endswith("np")
    ]
    after_operation_index = [
        i for i, name in enumerate(contour_names) if name.endswith("0m")
    ]
    print(before_operation_index)
    print(after_operation_index)
    kmeans = KMeans(n_clusters=2, random_state=123)
    clusters = kmeans.fit_predict(final_space)
    contour_names = np.array(contour_names)
    plt.figure(figsize=(10, 10))
    for operation_status in ("norm", "np", "0m"):
        plt.scatter(
            *final_space[:, :2][
                np.where(np.char.endswith(contour_names, operation_status))
            ].T,
            label=operation_status,
        )
    # for cluster in np.unique(clusters):
    #     ax = plt.scatter(*final_space[:, :2][np.where(clusters == cluster)].T, label={cluster})
    for i, name in enumerate(contour_names):
        plt.gca().annotate(name.split("_")[0], final_space[i, :2])
    for pair in pairs.values():
        tail = final_space[:, :2][pair["before"]]
        arrow = final_space[:, :2][pair["after"]]
        dv = arrow - tail
        plt.arrow(
            *tail,
            *dv,
            length_includes_head=True,
            linestyle="--",
            color="black",
            alpha=0.3,
            head_width=0.01,
            head_length=0.03,
        )
    plt.xlabel(f"PCA 1 {pca2.explained_variance_ratio_[0]:.2f}%")
    plt.ylabel(f"PCA 2 {pca2.explained_variance_ratio_[1]:.2f}%")
    plt.legend()
    plt.gca().set_aspect("equal")
    plt.savefig(result_dir.joinpath(f"{names[3]}/trajectory_space_scatterplot.png"))
    plt.close()

    for pc_coord1, pc_coord2 in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        plt.figure(figsize=(10, 10))
        for operation_status in ("norm", "np", "0m"):
            plt.scatter(
                *final_space[:, [pc_coord1, pc_coord2]][
                    np.where(np.char.endswith(contour_names, operation_status))
                ].T,
                label=operation_status,
            )
        # for cluster in np.unique(clusters):
        #     ax = plt.scatter(*final_space[:, :2][np.where(clusters == cluster)].T, label={cluster})
        for i, name in enumerate(contour_names):
            plt.gca().annotate(
                name.split("_")[0], final_space[i, [pc_coord1, pc_coord2]]
            )
        for pair in pairs.values():
            tail = final_space[:, [pc_coord1, pc_coord2]][pair["before"]]
            arrow = final_space[:, [pc_coord1, pc_coord2]][pair["after"]]
            dv = arrow - tail
            plt.arrow(
                *tail,
                *dv,
                length_includes_head=True,
                linestyle="--",
                color="black",
                alpha=0.3,
                head_width=0.01,
                head_length=0.03,
            )
        plt.xlabel(
            f"PCA {pc_coord1 + 1} {pca2.explained_variance_ratio_[pc_coord1]:.2f}%"
        )
        plt.ylabel(
            f"PCA {pc_coord2 + 1} {pca2.explained_variance_ratio_[pc_coord2]:.2f}%"
        )
        plt.legend()
        plt.gca().set_aspect("equal")
        plt.savefig(
            result_dir.joinpath(
                f"{names[3]}/trajectory_space_scatterplot{pc_coord1 + 1}_{pc_coord2 + 1}.png"
            )
        )
        plt.close()

    # out_dir = Path(result_dir.joinpath(f'{names[3]}/cycles_after_pca{n_components}_and_pca{n_components2}_and_gpa/'))
    # out_dir.mkdir(parents=True, exist_ok=True)
    # for i, pca_trajectory in enumerate(final_space):
    #     print(final_space.shape, pca_trajectory.shape)
    #     trajectory = get_original_space_from_trajectory_space(pca_trajectory,
    #                                                           pca2, trajectories_gpa.shape[1:],
    #                                                           pca1, aligned_shape_series.shape[1:])
    #     name = f'{contour_names[i]}_pca{n_components}_gpa_pca{n_components2}'
    #     out_path = out_dir.joinpath(f'{name}_cycle.gif')
    #     plot_trajectory_animation(trajectory.transpose(0, 2, 1), out_path, name, bounds, 40)

    mean_trajectory_in_trajectory_space = np.zeros_like(final_space[0])
    print("trajectory_space shape", mean_trajectory_in_trajectory_space.shape)
    mean_trajectory_in_shape_space = get_shape_space_from_trajectory_space(
        mean_trajectory_in_trajectory_space, pca2, trajectories_gpa.shape[1:]
    )
    print("shape space shape", mean_trajectory_in_shape_space.shape)
    mean_trajectory_in_original_space = get_original_space_from_shape_space(
        mean_trajectory_in_shape_space, pca1, aligned_shape_series.shape[1:]
    )
    print("original space shape", mean_trajectory_in_original_space.shape)
    mean_trajectory_name = "trajectory_space_mean"
    #
    trajectory_pca_space_dir = result_dir.joinpath(f"{names[3]}/trajectory_space")
    trajectory_pca_space_dir.mkdir(parents=True, exist_ok=True)

    from matplotlib.cm import ScalarMappable
    import matplotlib.colors as mcolors

    mean_trajectory = np.zeros_like(final_space[0])
    final_space_bounds = np.array(
        [np.min(final_space, axis=0), np.max(final_space, axis=0)]
    )
    n_points_per_component = 2
    for component in range(final_space.shape[1]):
        component_movement = np.concatenate(
            [
                np.linspace(
                    final_space_bounds[0, component],
                    0,
                    n_points_per_component,
                    endpoint=False,
                ),
                np.linspace(
                    0, final_space_bounds[1, component], n_points_per_component + 1
                ),
            ]
        )
        trajectories_ = np.array(
            [mean_trajectory for i in range(len(component_movement))]
        )
        for i in range(len(component_movement)):
            trajectories_[i, component] = component_movement[i]
        print(trajectories_.shape)
        trajectories_shapes = get_shape_space_from_trajectory_space(
            trajectories_, pca2, trajectories_gpa.shape[1:]
        )
        scalar_mappable = ScalarMappable(
            mcolors.Normalize(vmin=component_movement[0], vmax=component_movement[-1]),
            "coolwarm",
        )
        print(trajectories_shapes.shape)
        for i, shape in enumerate(trajectories_shapes):
            x, y = shape.T[:2]
            plt.plot(
                x, y, marker="o", color=scalar_mappable.to_rgba(component_movement[i])
            )
            plt.scatter(x[0], y[0], color="black", zorder=2, marker="x")
        plt.scatter(
            *trajectories_shapes[n_points_per_component + 1, 0, :2].T,
            color="black",
            label="First point in cycle",
            zorder=3,
            marker="x",
        )
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.xlabel("Shape Space component 1")
        plt.ylabel("Shape Space component 2")
        plt.title(
            f"Changes in {component + 1} component of Trajectories Space in Shape Space"
        )
        plt.legend()
        plt.colorbar(scalar_mappable, orientation="horizontal", label="Some Units")
        plt.gca().set_aspect("equal")
        plt.savefig(
            result_dir.joinpath(
                f"{names[3]}/trajectory_space_component_{component + 1}_in_shape_space.png"
            )
        )
        plt.close()

    # plot_trajectory_animation(mean_trajectory_in_original_space.transpose(0, 2, 1),
    #                           trajectory_pca_space_dir.joinpath(f'{mean_trajectory_name}.gif'),
    #                           mean_trajectory_name, bounds, 40
    #                           )
    final_space_bounds = (
        np.array([np.min(final_space, axis=0), np.max(final_space, axis=0)]) * 2
    )
    print(final_space_bounds.shape)
    mean_trajectory = np.zeros_like(final_space[0])
    mean_trajectory_shape = get_original_space_from_trajectory_space(
        mean_trajectory,
        pca2,
        trajectories_gpa.shape[1:],
        pca1,
        aligned_shape_series.shape[1:],
    )
    for component in range(final_space.shape[1]):
        min_trajectory = np.copy(mean_trajectory)
        min_trajectory[component] = final_space_bounds[0, component]
        min_trajectory = get_original_space_from_trajectory_space(
            min_trajectory,
            pca2,
            trajectories_gpa.shape[1:],
            pca1,
            aligned_shape_series.shape[1:],
        )
        min_name = f"PCA{component + 1} min"
        plot_trajectory_animation_diff(
            min_trajectory.transpose(0, 2, 1),
            mean_trajectory_shape.transpose(0, 2, 1),
            trajectory_pca_space_dir.joinpath(f"{min_name}.gif"),
            min_name,
            bounds,
            100,
        )
        max_trajectory = np.copy(mean_trajectory)
        max_trajectory[component] = final_space_bounds[1, component]
        max_trajectory = get_original_space_from_trajectory_space(
            max_trajectory,
            pca2,
            trajectories_gpa.shape[1:],
            pca1,
            aligned_shape_series.shape[1:],
        )
        max_name = f"PCA{component + 1} max"
        # plot_trajectory_animation_diff(max_trajectory.transpose(0, 2, 1),
        #                                mean_trajectory_shape.transpose(0, 2, 1),
        #                                trajectory_pca_space_dir.joinpath(f'{max_name}.gif'),
        #                                max_name, bounds, 100
        #                                )
    for i in (3,):
        shape_preprocessing_dir = result_dir.joinpath(names[i])
        shape_preprocessing_dir.mkdir(parents=True, exist_ok=True)
        pca_1_dir = shape_preprocessing_dir.joinpath("pca_1_dir")
        pca_1_dir.mkdir(parents=True, exist_ok=True)
        pca_animation1(
            shapes_list[i].shape,
            pca=pca1,
            pca_space=trajectories.reshape(-1, *trajectories.shape[2:]),
            out_dir=pca_1_dir,
            n_frames=10,
            interval=50,
        )
    bounds_list = np.array(
        [[np.min(temp, axis=(0, 1)), np.max(temp, axis=(0, 1))] for temp in shapes_list]
    )

    for i in (3,):
        shape_preprocessing_dir = result_dir.joinpath(names[i])
        pca_1_dir = shape_preprocessing_dir.joinpath("pca_1_dir")
        explained_variance = pca1.explained_variance_ratio_
        mean_trajectory = np.mean(trajectories[np.where(norm_check)], axis=0)
        print(mean_trajectory.shape)
        mean_trajectory_shapes = pca1.inverse_transform(mean_trajectory)
        print(shapes.shape)
        mean_trajectory_shapes = mean_trajectory_shapes.reshape(
            -1, *shapes[0].shape
        ).transpose(0, 2, 1)

        mean_path_trajectory = np.mean(
            trajectories[np.where(np.logical_not(norm_check))], axis=0
        )
        mean_path_trajectory_shapes = pca1.inverse_transform(mean_path_trajectory)
        mean_path_trajectory_shapes = mean_path_trajectory_shapes.reshape(
            -1, *shapes[0].shape
        ).transpose(0, 2, 1)

        # plot_trajectory_animation(mean_trajectory_shapes, bounds=bounds_list[i],
        #                           out_path=pca_1_dir.joinpath('mean trajectory animation.gif'),
        #                           name=names[i] + ' mean',
        #                           interval=40
        #                           )
        # plot_trajectory_animation_diff(mean_path_trajectory_shapes, mean_trajectory_shapes,
        #                                out_path=pca_1_dir.joinpath('animation of mean norm and path trajectories.gif'),
        #                                name=names[i] + ' norm and path names,',
        #                                bounds=bounds, interval=100
        #                                )
        for pc_coord1, pc_coord2 in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
            plt.plot(
                np.mean(norm_trajectories_gpa, axis=0)[:, pc_coord1].T,
                np.mean(norm_trajectories_gpa, axis=0)[:, pc_coord2].T,
                color="black",
                marker="o",
                label="mean trajectory (norm",
            )
            plt.plot(
                np.mean(pathology_trajectories_gpa, axis=0)[:, pc_coord1].T,
                np.mean(pathology_trajectories_gpa, axis=0)[:, pc_coord2].T,
                color="red",
                marker="o",
                label="mean trajectory (pathology)",
            )
            plt.legend()
            plt.xlabel(f"PC{pc_coord1 + 1} {explained_variance[pc_coord1]:.3f}")
            plt.ylabel(f"PC{pc_coord2 + 1} {explained_variance[pc_coord2]:.3f}")
            plt.gca().set_aspect("equal")
            plt.title(f"{names[i]} mean trajectories in PCA space")
            plt.savefig(
                pca_1_dir.joinpath(
                    f"mean trajectories_{pc_coord1 + 1}_{pc_coord2 + 1}.png"
                )
            )
            plt.close()

        for pc_coord1, pc_coord2 in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
            plt.plot(
                np.mean(trajectories[np.where(norm_check)], axis=0)[:, pc_coord1].T,
                np.mean(trajectories[np.where(norm_check)], axis=0)[:, pc_coord2].T,
                color="black",
                marker="o",
                label="mean trajectory (norm",
            )
            plt.plot(
                np.mean(trajectories[np.where(np.logical_not(norm_check))], axis=0)[
                    :, pc_coord1
                ].T,
                np.mean(trajectories[np.where(np.logical_not(norm_check))], axis=0)[
                    :, pc_coord2
                ].T,
                color="red",
                marker="o",
                label="mean trajectory (pathology)",
            )
            plt.legend()
            plt.xlabel(f"PC{pc_coord1 + 1} {explained_variance[0]:.3f}")
            plt.ylabel(f"PC{pc_coord2 + 1} {explained_variance[1]:.3f}")
            plt.gca().set_aspect("equal")
            plt.title(f"{names[i]} mean trajectories in PCA space before GPA")
            plt.savefig(
                pca_1_dir.joinpath(
                    f"before GPA mean trajectories_{pc_coord1 + 1}_{pc_coord2 + 1}.png"
                )
            )
            plt.close()

for traj in norm_trajectories_gpa:
    plt.plot(*traj[:, :2].T, color="black", marker="o")
for traj in pathology_trajectories_gpa:
    plt.plot(*traj[:, :2].T, color="red", marker="o")
plt.xlabel(f"PC1 {explained_variance[0]:.3f}")
plt.ylabel(f"PC2 {explained_variance[1]:.3f}")
plt.title(f"{names[i]} trajectories in shape space")
plt.gca().set_aspect("equal")
plt.savefig(pca_1_dir.joinpath("trajectories_in_shape_space.png"))
plt.close()
