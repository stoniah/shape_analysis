import numpy as np
import scipy.interpolate


def time_warp(n_frames, end_systolic_id, systolic_fraction=0.35):
    """
    Performs a piecewise-linear time warp. The frames up to the end-systole
    comprise systolic_fraction of the [0,1] interval, following frames comprise the rest.
    :param n_frames: int
        number of frames
    :param end_systolic_id: int
        end-systolic frame number
    :param systolic_fraction: float
        fraction of the [0,1] interval reserved for the systole
    :return: (n_frames,) float array
        position of each frame in the [0,1] interval after the warp
    """
    frame_ids = np.arange(n_frames)
    time = np.zeros(n_frames)
    systole_mask = np.where(frame_ids <= end_systolic_id)
    time[systole_mask] = frame_ids[systole_mask] * systolic_fraction / end_systolic_id
    diastole_mask = np.where(frame_ids > end_systolic_id)
    time[diastole_mask] = systolic_fraction + (1 - systolic_fraction) * (
        frame_ids[diastole_mask] - end_systolic_id
    ) / (n_frames - end_systolic_id - 1)
    return time


def interpolate_curve(curve, dist, n_points=20):
    """
    Interpolates in equidistant points a 2d curve
    defined by curve = f(dist).
    Interpolation is based on cubic splines.
    :param curve: (t, 2) array-like
        curve in 2d space
    :param dist: t array-like
    :param n_points: int
        number of points along the curve
    :return: (n_points, 2) array-like
    """
    alpha = np.linspace(0, 1, n_points)
    interpolator = scipy.interpolate.interp1d(
        dist, np.transpose(curve), kind="cubic", axis=0
    )
    return np.array(interpolator(alpha))


def interpolate_time(
    coordinates, end_systolic_id, systolic_fraction=0.35, n_time_points=21
):
    """
    Performs interpolation on the cardiac cycle
    :param coordinates: (t, p, 2) numpy array
        Coordinates of the heart contour points throughout the cardiac cycle
        t - number of frames in the cardiac cycle
        p - number of points along the heart contour
    :param end_systolic_id: int
        end-systolic frame number
    :param systolic_fraction: float
        fraction of the [0,1] interval reserved for the systole
    :param n_time_points: int
        desired number of timepoints for the cardiac cycle
    :return: (n_time_points, p, 2) numpy array
        interpolated cardiac cycle. First systolic_fraction * n_time_points
        timepoints are taken equidistant from the systolic interval,
        other timepoints equidistantly comprise the diastolic interval.
    """
    time = time_warp(coordinates.shape[0], end_systolic_id, systolic_fraction)
    interpolated_coords = []
    for point_id in range(coordinates.shape[1]):
        point_trajectory = coordinates[:, point_id]
        spline_trajectory = interpolate_curve(point_trajectory.T, time, n_time_points)
        interpolated_coords.append(spline_trajectory)
    interpolated_coords_np = np.array(interpolated_coords).transpose(1, 0, 2)
    return interpolated_coords_np
