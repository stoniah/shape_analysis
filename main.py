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
    np.save("results/data/contours", contours)
    np.save("results/data/contours_time_interpolated", contours_time_interpolated)
    np.save("results/data/contours_fourier_interpolated", contours_fourier_interpolated)

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
    np.save(Path("results/contour_names.npy"), contour_names)
    shapes = shape_series.reshape(-1, *shape_series.shape[2:])

    scale = True
    global_mean, gpa_shapes = shape_analysis.generalized_procrustes_analysis(
        shapes, scale=scale
    )

    gpa_shape_series = gpa_shapes.reshape(*shape_series.shape)
    print(global_mean.shape, gpa_shapes.shape, gpa_shape_series.shape)
    ls_shape_series = shape_analysis.linear_shift(
        shape_series, global_mean, scale=scale
    )
    print(ls_shape_series.shape)
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
    names = [
        "raw",
        f'gpa{"" if scale else "_noscale"}',
        f'linear_shifted{"" if scale else "_noscale"}',
        f'linear_shifted_gpa{"" if scale else "_noscale"}',
    ]

    out_path = Path(f"results/{names[3]}").joinpath("fully_aligned_shapes.npy")
    out_path.parent.mkdir(exist_ok=True)
    np.save(out_path, lsgpa_shape_series)
    aligned_shapes = np.load(out_path)
    print(aligned_shapes.shape)
    bounds = np.array(
        [np.min(aligned_shapes, axis=(0, 1, 2)), np.max(aligned_shapes, axis=(0, 1, 2))]
    )
    # out_dir = Path(f'results/{names[3]}/cycles/')
    # out_dir.mkdir(parents=True, exist_ok=True)
    # for i, trajectory in enumerate(aligned_shapes):
    #     name = contour_names[i]
    #     out_path = out_dir.joinpath(f'{contour_names[i]}_cycle.gif')
    #     plot_trajectory_animation(trajectory.transpose(0, 2, 1), out_path, name, bounds, 40)

    n_components = 4
    pca1 = PCA(svd_solver="full", n_components=n_components)
    trajectories = pca1.fit_transform(
        aligned_shapes.reshape(np.product(aligned_shapes.shape[:2]), -1)
    ).reshape(*aligned_shapes.shape[:2], -1)
    np.save(Path(f"results/{names[3]}/shape_space.npy"), trajectories)
    plt.plot(pca1.explained_variance_ratio_, marker="o")
    plt.ylim(0, 1)
    plt.xlabel("n_components")
    plt.ylabel("Explained variance ratio")
    plt.title("Shape space components explained variance ratio")
    plt.savefig(f"results/{names[3]}/shape_space_elbow_plot.png")
    plt.close()
    print(trajectories.shape)
    mean_trajectory, trajectories_gpa = shape_analysis.generalized_procrustes_analysis(
        trajectories, scale=scale
    )
    n_components2 = 6
    pca2 = PCA(svd_solver="full", n_components=n_components2)
    final_space = pca2.fit_transform(
        trajectories_gpa.reshape(trajectories_gpa.shape[0], -1)
    )
    np.save(Path(f"results/{names[3]}/trajectory_space.npy"), final_space)
    plt.plot(pca2.explained_variance_ratio_, marker="o")
    plt.ylim(0, 1)
    plt.xlabel("n_components")
    plt.ylabel("Explained variance ratio")
    plt.title("Trajectory space components explained variance ratio")
    plt.savefig(f"results/{names[3]}/trajectory_space_elbow_plot.png")
    plt.close()

    np.save("space_shape_components.npy", pca1.components_)
    np.save("space_shape_variance.npy", pca1.explained_variance_ratio_)
    np.save("space_traj_components.npy", pca2.components_)
    np.save("space_traj_variance.npy", pca2.explained_variance_ratio_)

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
    plt.figure(figsize=(10, 10))
    contour_names = np.array(contour_names)
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
    plt.savefig(Path(f"results/{names[3]}/trajectory_space_scatterplot.png"))
    plt.close()

    # out_dir = Path(f'results/{names[3]}/cycles_after_pca{n_components}_and_pca{n_components2}_and_gpa/')
    # out_dir.mkdir(parents=True, exist_ok=True)
    # for i, pca_trajectory in enumerate(final_space):
    #     trajectory = get_original_space_from_trajectory_space(pca_trajectory,
    #                                                           pca2, trajectories_gpa.shape[1:],
    #                                                           pca1, aligned_shapes.shape[1:])
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
        mean_trajectory_in_shape_space, pca1, aligned_shapes.shape[1:]
    )
    print("original space shape", mean_trajectory_in_original_space.shape)
    mean_trajectory_name = "trajectory_space_mean"
    #
    trajectory_pca_space_dir = Path(f"results/{names[3]}/trajectory_space")
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
        trajectories = np.array(
            [mean_trajectory for i in range(len(component_movement))]
        )
        for i in range(len(component_movement)):
            trajectories[i, component] = component_movement[i]
        print(trajectories.shape)
        trajectories_shapes = get_shape_space_from_trajectory_space(
            trajectories, pca2, trajectories_gpa.shape[1:]
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
            Path(
                f"results/{names[3]}/trajectory_space_component_{component + 1}_in_shape_space.png"
            )
        )
        plt.close()

    plot_trajectory_animation(
        mean_trajectory_in_original_space.transpose(0, 2, 1),
        trajectory_pca_space_dir.joinpath(f"{mean_trajectory_name}.gif"),
        mean_trajectory_name,
        bounds,
        40,
    )
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
        aligned_shapes.shape[1:],
    )
    for component in range(final_space.shape[1]):
        min_trajectory = np.copy(mean_trajectory)
        min_trajectory[component] = final_space_bounds[0, component]
        min_trajectory = get_original_space_from_trajectory_space(
            min_trajectory,
            pca2,
            trajectories_gpa.shape[1:],
            pca1,
            aligned_shapes.shape[1:],
        )
        min_name = f"PCA{component + 1} min"
        # plot_trajectory_animation_diff(min_trajectory.transpose(0, 2, 1),
        #                                mean_trajectory_shape.transpose(0, 2, 1),
        #                                trajectory_pca_space_dir.joinpath(f'{min_name}.gif'),
        #                                min_name, bounds, 100
        #                                )
        max_trajectory = np.copy(mean_trajectory)
        max_trajectory[component] = final_space_bounds[1, component]
        max_trajectory = get_original_space_from_trajectory_space(
            max_trajectory,
            pca2,
            trajectories_gpa.shape[1:],
            pca1,
            aligned_shapes.shape[1:],
        )
        max_name = f"PCA{component + 1} max"
        # plot_trajectory_animation_diff(max_trajectory.transpose(0, 2, 1),
        #                                mean_trajectory_shape.transpose(0, 2, 1),
        #                                trajectory_pca_space_dir.joinpath(f'{max_name}.gif'),
        #                                max_name, bounds, 100
        #                                )
    for i in (3,):
        shape_preprocessing_dir = Path("results").joinpath(names[i])
        shape_preprocessing_dir.mkdir(parents=True, exist_ok=True)
        pca_1_dir = shape_preprocessing_dir.joinpath("pca_1_dir")
        pca_1_dir.mkdir(parents=True, exist_ok=True)
        # pca_animation1(shapes_list[i], n_components=8, out_dir=pca_1_dir, n_frames=10, interval=50)
        pca_animation1(
            shapes_list[i].shape,
            pca1,
            trajectories,
            out_dir=pca_1_dir,
            n_frames=10,
            interval=50,
        )
    bounds_list = np.array(
        [[np.min(temp, axis=(0, 1)), np.max(temp, axis=(0, 1))] for temp in shapes_list]
    )

    for i in (3,):
        shape_preprocessing_dir = Path("results").joinpath(names[i])
        pca_1_dir = shape_preprocessing_dir.joinpath("pca_1_dir")
        pca = PCA(svd_solver="full")
        shapes = shapes_list[i]
        shape_series = shape_series_list[i]
        pca_space = pca.fit_transform(shapes.reshape(-1, np.product(shapes.shape[1:])))
        pca_space = pca_space.reshape(*shape_series.shape[:2], -1)
        explained_variance = pca.explained_variance_ratio_
        mean_trajectory = np.mean(pca_space, axis=0)
        print(mean_trajectory.shape)
        mean_trajectory_shapes = pca.inverse_transform(mean_trajectory)
        print(shapes.shape)
        mean_trajectory_shapes = mean_trajectory_shapes.reshape(
            -1, *shapes[0].shape
        ).transpose(0, 2, 1)

        plot_trajectory_animation(
            mean_trajectory_shapes,
            bounds=bounds_list[i],
            out_path=pca_1_dir.joinpath("mean trajectory animation.gif"),
            name=names[i] + " mean",
            interval=40,
        )
        plt.plot(*mean_trajectory[:, :2].T, color="black", marker="o")
        plt.xlabel(f"PC1 {explained_variance[0]:.3f}")
        plt.ylabel(f"PC2 {explained_variance[1]:.3f}")
        plt.title(f"{names[i]} mean trajectory in PCA space")
        plt.savefig(pca_1_dir.joinpath("mean trajectory.png"))
        plt.close()
