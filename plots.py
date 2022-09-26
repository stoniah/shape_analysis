from reading_utils import get_contour_title
from time_interpolation import time_warp

import functools
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.animation import FuncAnimation
import numpy as np
from sklearn.decomposition import PCA


def plot_contours(output_dir, contours, contours_time_interpolated, contours_fourier_interpolated, n_harmonics):
    for i in range(len(contours)):
        coords_orig = contours[i]['coordinates']
        coords_time = contours_time_interpolated[i]['coordinates']
        coords_fourier = contours_fourier_interpolated[i]['coordinates']
        plt.figure(figsize=(40, 40))
        for point in range(coords_orig.shape[1]):
            curve_orig = coords_orig[:, point]
            curve_time = coords_time[:, point]
            curve_fourier = coords_fourier[:, point]
            plt.plot(*curve_orig.T, marker='o', markersize=2, linewidth=1, fillstyle='none', color='black', zorder=1)
            plt.scatter(*curve_time.T, marker='o', s=9, facecolors='none', color='blue', zorder=2, linewidths=0.5)
            plt.scatter(*curve_fourier.T, marker='^', facecolors='none', color='red', zorder=3, linewidths=0.5)
        name = get_contour_title(contours[i])
        plt.title(name + f'_n_harmonics_{n_harmonics}')
        plt.gca().set_aspect('equal')
        plt.xlim(-6, 6)
        plt.ylim(-6, 6)
        plt.savefig(output_dir.joinpath(name + '.png'))
        plt.close()


def plot_areas(area_graph_dir, contours, areas, areas_time_interpolated, areas_fourier_interpolated):
    area_lists = [areas, areas_time_interpolated, areas_fourier_interpolated]
    markers = ['o', 's', '^']
    colors = ['black', 'blue', 'red']
    min_ = np.min(np.min(area_lists))
    max_ = np.max(np.max(area_lists))

    area_graph_dir.mkdir(parents=True, exist_ok=True)
    for j in range(len(contours)):
        for i in range(3):
            area_list = area_lists[i][j]
            if i == 0:
                time = time_warp(len(area_list), contours[j]['end_systolic_id'])
            else:
                time = np.linspace(0, 1, len(area_list))
            plt.plot(time, area_list, color=colors[i], marker=markers[i], fillstyle='none')
        plt.ylim(min_ - 1, max_ + 1)
        plt.xlabel('warped time')
        plt.ylabel('LV projection area')
        title = get_contour_title(contours[j])
        plt.title(title)
        plt.savefig(area_graph_dir.joinpath(title + '.png'))
        plt.close()


def pca_animation1(shapes_shape, pca, pca_space, out_dir, n_frames=60, interval=50):
    """
    :param shapes_shape: ndarray of shape (nshapes, ndims, 2npoints)
        2npoints means that the first npoints are endo and second are epi
    :param pca: int
        number of components in decomposition to visualize
    :param out_dir: string, pathlike
        where to save all results_norm
    """
    name = out_dir.parent.name

    def update(frame, line_endo, line_epi, scalar_mappable, ax):
        shape, ref_shape = frame
        n_points = shape.shape[1] // 2

        line_endo.set_data(*shape[:, :n_points])
        line_epi.set_data(*shape[:, n_points:])

        patches = []
        for patch in ax.patches:
            patch.remove()
            # patch.set_visible(False)
        for i in range(shape.shape[1]):
            x, y = shape[:, i]
            dx, dy = ref_shape[0, i] - x, ref_shape[1, i] - y
            distance = np.linalg.norm([dx, dy])
            color = scalar_mappable.to_rgba(distance)
            patch = mpatches.Arrow(x, y, dx, dy, color=color)
            patches.append(ax.add_patch(patch))

        return line_endo, line_epi, *patches

    def init(ax, line_endo, line_epi, bounds):
        # ax.set_yaxis_direction('down')
        ax.set_xlim(bounds[:, 0] * 1.1)
        ax.set_ylim(bounds[:, 1] * 1.1)
        ax.set_xlabel('centimeters')
        ax.set_ylabel('centimeters')
        ax.set_aspect('equal')
        return line_endo, line_epi,

    n_components = len(pca.explained_variance_ratio_)
    elbow_plot_outfile = out_dir.joinpath('elbow.png')
    variance = pca.explained_variance_ratio_
    component_ids = np.arange(1, len(variance) + 1)
    plt.plot(component_ids, variance, marker='o')
    plt.xlabel('n_components')
    plt.ylabel('explained_variance')
    plt.title(f'PCA on f{name}. Explained variance')
    plt.ylim(0, 1)
    plt.savefig(elbow_plot_outfile)
    plt.close()

    pca_bounds = np.array([np.min(pca_space, axis=0), np.max(pca_space, axis=0)])
    pca_mean_shape = np.zeros_like(pca_space[0])
    for component in range(n_components):
        comp_min, comp_max = pca_bounds[:, component]
        component_increase = np.linspace(0, comp_max, n_frames + 1)
        component_decrease = np.linspace(comp_min, 0, n_frames, endpoint=False)
        component_change = np.concatenate(
            [component_decrease, component_increase, component_increase[::-1][1:], component_decrease[::-1][:-1]])
        pca_series = np.repeat(pca_mean_shape[np.newaxis, :], len(component_change), axis=0)
        pca_series[:, component] = component_change
        shape_variation_series = pca.inverse_transform(pca_series).reshape(-1, *shapes_shape[1:]).transpose(0, 2, 1)
        distances = np.linalg.norm(shape_variation_series - shape_variation_series[n_frames + 1], axis=1)
        shape_bounds = np.array(
            [np.min(shape_variation_series, axis=(0, 2)), np.max(shape_variation_series, axis=(0, 2))])
        fig, ax = plt.subplots()
        ax.set_title(f'{name}\nPCA {component + 1} {variance[component]:.3f}%')
        line_endo, = plt.plot([], [], marker='o', color='black')
        line_epi, = plt.plot([], [], marker='o', color='black')
        scalar_mappable = ScalarMappable(mcolors.Normalize(vmin=0, vmax=np.max(distances)), 'YlOrRd')

        init_func = functools.partial(init, ax=ax, line_endo=line_endo, line_epi=line_epi, bounds=shape_bounds)
        update_func = functools.partial(update, line_endo=line_endo, line_epi=line_epi, scalar_mappable=scalar_mappable,
                                        ax=ax)
        frames = [(shape_variation_series[i], shape_variation_series[n_frames + 1]) for i in
                  range(len(shape_variation_series))]
        ani = FuncAnimation(fig, func=update_func, frames=frames, interval=interval,
                            init_func=init_func, blit=True)
        ani.save(out_dir.joinpath(f'pca_component{component + 1}.gif'))
        plt.close(fig)


def plot_trajectory_animation(trajectory, out_path, name, bounds, interval):
    def update(frame, line_epi, line_endo):
        n_points = frame.shape[1] // 2

        line_endo.set_data(*frame[:, :n_points])
        line_epi.set_data(*frame[:, n_points:])
        return line_endo, line_epi

    fig, ax = plt.subplots(figsize=(6, 6))
    square_bounds = np.min(bounds), np.max(bounds)
    # ax.set_xlim(bounds[:, 0])
    # ax.set_ylim(bounds[:, 1])
    ax.set_xlim(square_bounds)
    ax.set_ylim(square_bounds)
    ax.set_xlabel('centimeters')
    ax.set_ylabel('centimeters')
    ax.set_title(f'{name} trajectory')
    ax.set_aspect('equal')
    # fig.subplots_adjust(left=0, bottom=0.09, right=0.9, top=0.95, wspace=0, hspace=0)
    fig.tight_layout(pad=0)

    line_endo, = plt.plot([], [], marker='o', color='black')
    line_epi, = plt.plot([], [], marker='o', color='black')

    update_func = functools.partial(update, line_epi=line_epi, line_endo=line_endo)
    ani = FuncAnimation(fig, func=update_func, frames=trajectory, interval=interval, blit=True)
    ani.save(out_path)
    plt.close(fig)


def plot_trajectory_animation_diff(trajectory, mean_trajectory, out_path, name, bounds, interval):
    def update(frame, line_epi_, line_endo_, line_epi_ref_, line_endo_ref_, connecting_lines_, scatter_,
               scalar_mappable_):
        shape, ref_shape = frame
        n_points = shape.shape[1] // 2

        line_endo_.set_data(*shape[:, :n_points])
        line_epi_.set_data(*shape[:, n_points:])
        line_endo_ref_.set_data(*ref_shape[:, :n_points])
        line_epi_ref_.set_data(*ref_shape[:, n_points:])

        dist = np.linalg.norm(shape - ref_shape, axis=0)
        scatter_.set_offsets(shape.T)
        colors = scalar_mappable_.to_rgba(dist)
        scatter_.set_fc(colors)
        scatter_.set_ec('face')

        for i in range(shape.shape[1]):
            x1, y1 = shape[:, i]
            x2, y2 = ref_shape[:, i]
            distance = np.linalg.norm([x2 - x1, y2 - y1])
            connecting_lines_[i].set_data([x1, x2], [y1, y2])
            connecting_lines_[i].set_color(scalar_mappable_.to_rgba(distance))

        return line_endo_, line_epi_, scatter_

    fig, ax = plt.subplots(figsize=(6, 6))
    square_bounds = np.min(bounds), np.max(bounds)
    # ax.set_xlim(bounds[:, 0])
    # ax.set_ylim(bounds[:, 1])
    ax.set_xlim(square_bounds)
    ax.set_ylim(square_bounds)
    ax.set_xlabel('centimeters')
    ax.set_ylabel('centimeters')
    ax.set_title(f'{name} trajectory')
    ax.set_aspect('equal')
    # fig.subplots_adjust(left=0, bottom=0.09, right=0.9, top=0.95, wspace=0, hspace=0)
    fig.tight_layout(pad=0)

    distances = np.linalg.norm(trajectory - mean_trajectory, axis=1)
    scalar_mappable = ScalarMappable(mcolors.Normalize(vmin=0, vmax=np.max(distances)), 'YlOrRd')

    plt.plot(*mean_trajectory[0], color='blue', marker='o', alpha=0.7)
    line_endo, = plt.plot([], [], color='black')
    line_epi, = plt.plot([], [], color='black')
    line_endo_ref, = plt.plot([], [], color='black', linestyle='dashed', alpha=0.7)
    line_epi_ref, = plt.plot([], [], color='black', linestyle='dashed', alpha=0.7)
    connecting_lines = [plt.plot([], [])[0] for _ in range(trajectory.shape[2])]

    scatter = plt.scatter(trajectory[0][0], trajectory[0][1], c=scalar_mappable.to_rgba(distances[0]), cmap='YlOrRd',
                          s=100,
                          vmin=0, vmax=np.max(distances))

    update_func = functools.partial(update, line_epi_=line_epi, line_endo_=line_endo, line_endo_ref_=line_endo_ref,
                                    line_epi_ref_=line_epi_ref, connecting_lines_=connecting_lines,
                                    scatter_=scatter,
                                    scalar_mappable_=scalar_mappable)
    frames = np.array([trajectory, mean_trajectory]).transpose(1, 0, 2, 3)
    ani = FuncAnimation(fig, func=update_func, frames=frames, interval=interval,
                        # blit=True
                        )
    ani.save(out_path)
    plt.close(fig)
