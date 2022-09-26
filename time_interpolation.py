import numpy as np
import scipy.interpolate


def time_warp(n_frames, end_systolic_id, systolic_fraction=0.35):
    frame_ids = np.arange(n_frames)
    time = np.zeros(n_frames)
    systole_mask = np.where(frame_ids <= end_systolic_id)
    time[systole_mask] = frame_ids[systole_mask] * systolic_fraction / end_systolic_id
    diastole_mask = np.where(frame_ids > end_systolic_id)
    time[diastole_mask] = systolic_fraction + (1 - systolic_fraction) * \
                          (frame_ids[diastole_mask] - end_systolic_id) / (n_frames - end_systolic_id - 1)
    return time


def interpolate_curve(curve, dist, n_points=20):
    alpha = np.linspace(0, 1, n_points)
    interpolator = scipy.interpolate.interp1d(dist, np.transpose(curve), kind='cubic', axis=0)
    return np.array(interpolator(alpha))


def interpolate_time(coordinates, end_systolic_id, systolic_fraction=0.35, n_time_points=21):
    time = time_warp(coordinates.shape[0], end_systolic_id, systolic_fraction)
    interpolated_coords = []
    for point_id in range(coordinates.shape[1]):
        point_trajectory = coordinates[:, point_id]
        spline_trajectory = interpolate_curve(point_trajectory.T, time, n_time_points)
        interpolated_coords.append(spline_trajectory)
    interpolated_coords_np = np.array(interpolated_coords).transpose(1, 0, 2)
    return interpolated_coords_np
