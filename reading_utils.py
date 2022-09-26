import numpy as np
from time_interpolation import interpolate_time
from collections import defaultdict


def is_norm(tokens):
    return len(tokens) == 2


def is_crt(tokens):
    return len(tokens) == 4


def parse_filename(filename):
    filename_tokens = filename.split('_')
    if is_norm(filename_tokens):
        patient_code, wall_type = filename_tokens
        operation_status = 'norm'
    elif is_crt(filename_tokens):
        patient_code, operation_status, wall_type = filename.split('_')[-3:]
    else:
        raise ValueError(f'{filename} is from unknown patient group')
    return patient_code, operation_status, wall_type


def get_length_from_segment_coordinates(segment_coordinates):
    """
    coords: listlike of 4 numbers x1,y1,x2,y2
    returns: squared distance between segment endpoints
    """
    p1, p2 = np.reshape(segment_coordinates, (2, 2))
    diff = p2 - p1
    dot = np.dot(diff, diff)
    return np.sqrt(dot)


def parse_header(string):
    numbers = list(map(int, string.split()))
    end_systolic_id = numbers[0]
    calibration_segment_coordinates = numbers[1:]
    pixels_per_cm = get_length_from_segment_coordinates(calibration_segment_coordinates)
    header_info = {
        'end_systolic_id': end_systolic_id,
        'pixels_per_cm': pixels_per_cm
    }
    return header_info


def read_contour_file(path):
    patient_code, operation_status, wall_type = parse_filename(path.stem)
    with open(path, 'r') as f:
        header = f.readline()
        header_info = parse_header(header)
    coordinates = np.loadtxt(path, skiprows=1).astype(int)
    # print(coordinates.shape)
    coordinates = coordinates.reshape(coordinates.shape[0], -1, 2)[:, :, ::-1]
    # import matplotlib.pyplot as plt
    # print(path)
    # plt.plot(*coordinates[0].T, color='blue', marker='o')
    # plt.plot(*coordinates[0,:,::-1].T, color='red', marker='o')
    # plt.gca().set_aspect('equal')
    # plt.show()
    # print(coordinates.shape)
    # coordinates = coordinates.transpose(0, 2, 1)
    contour_info = header_info.copy()
    contour_info['coordinates'] = coordinates
    contour_info['patient_code'] = patient_code
    contour_info['operation_status'] = operation_status
    contour_info['wall_type'] = wall_type
    return contour_info


def load_data(path, only_norm=False, only_endo=False):
    if only_norm:
        group_filter = lambda x: x.name.startswith('N')
    else:
        group_filter = None
    return [read_contour_file(p) for p in filter(group_filter, sorted(path.iterdir()))]


def fourier_interpolate_coords(coords, n_harmonics=10, n_points_out=None):
    """
    :param coords: ndarray of shape n_time,n_points,space_dim
    :param n_harmonics: number of harmonics to use
    :param n_points_out: number of points to get in output
    :return: array of same shape n_time,n_points_out,space dim interpolated by n_harmonics harmonic functions
    """
    coords_center = np.mean(coords, axis=0, keepdims=True)
    coords_centered = coords - coords_center
    coords_f = np.fft.rfft(coords_centered, axis=0)
    coords_f[n_harmonics:] = 0
    coords_if = np.fft.irfft(coords_f, n=len(coords), axis=0)
    if n_points_out:
        skip = len(coords_if) // n_points_out
        coords_if = coords_if[::skip]
    coords_if += coords_center
    return coords_if


def prepare_contour(contour,
                    scale=True,
                    center=True,
                    interpolate_time_points=100,
                    fourier_interpolate_kwargs=True,
                    close_contours=True):
    coords = contour['coordinates']
    coords_new = np.copy(coords)
    if center:
        coords_new = coords_new - np.mean(coords_new[0], axis=0, keepdims=True)
    if scale:
        scale = contour['pixels_per_cm']
        coords_new = coords_new / scale
    if interpolate_time_points:
        es_id = contour['end_systolic_id']
        coords_new = interpolate_time(coords_new, end_systolic_id=es_id, n_time_points=interpolate_time_points)
    if fourier_interpolate_kwargs is not None:
        coords_new = fourier_interpolate_coords(coords_new, **fourier_interpolate_kwargs)
    if close_contours:
        coords_new = np.insert(coords_new, -1, coords_new[0], axis=0)
    new_contour = contour.copy()
    new_contour['coordinates'] = coords_new
    return new_contour


def prepare_contours(contours,
                     scale=True,
                     center=True,
                     interpolate_time_points=100,
                     fourier_interpolate_kwargs=True,
                     close_contours=True):
    prepared_contours = [
        prepare_contour(contour,
                        scale=scale,
                        center=center,
                        interpolate_time_points=interpolate_time_points,
                        fourier_interpolate_kwargs=fourier_interpolate_kwargs,
                        close_contours=close_contours)
        for contour in contours
    ]
    return prepared_contours


def get_contour_title(contour):
    return '_'.join([contour['patient_code'], contour['operation_status'], contour['wall_type']])


def get_contour_tensor(contours, only_endo=False):
    contours_dict = defaultdict(dict)
    for contour in contours:
        contour_id = contour['patient_code'] + '_' + contour['operation_status']
        contours_dict[contour_id][contour['wall_type']] = contour['coordinates']
    contour_names = np.array(list(contours_dict.keys()))
    shape_series = []
    for val in contours_dict.values():
        endo, epi = val['endo'], val['epi']
        if only_endo:
            full_contour = endo
        else:
            full_contour = np.hstack([endo, epi])
        shape_series.append(full_contour)
    shape_series = np.array(shape_series).transpose(0, 1, 2, 3)  # we need to have 2 before else

    return contour_names, shape_series
