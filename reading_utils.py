from collections import defaultdict
import numpy as np
import time_interpolation


def _is_norm(tokens):
    """

    :param tokens:
    :return:
    """
    # return len(tokens) == 2
    return len(tokens) == 4


def _is_crt(tokens):
    """

    :param tokens:
    :return:
    """
    return len(tokens) == 3


def parse_filename(filename):
    """
    Filename parser. Accepts filenames of format {code}_{wall} and {code}_{status}_{wall}
    where code - is some string without underscores,
    wall is 'endo' or 'epi',
    and status is the string allowing to differentiate different records of the same patient.
    :param filename: string
    :return: tuple with three strings
    """
    filename_tokens = filename.split("_")
    if _is_norm(filename_tokens):
        # patient_code, wall_type = filename_tokens
        patient_code = filename_tokens[1]
        wall_type = filename_tokens[-1]
        operation_status = "norm"
    elif _is_crt(filename_tokens):
        patient_code, operation_status, wall_type = filename.split("_")[-3:]
    else:
        raise ValueError(f"{filename} is from unknown patient group")
    return patient_code, operation_status, wall_type


def _get_length_from_segment_coordinates(segment_coordinates):
    """
    Finds distance between two points on a plane
    coords: listlike of 4 numbers x1,y1,x2,y2
    returns: Euclidean distance between segment endpoints
    """
    point1, point2 = np.reshape(segment_coordinates, (2, 2))
    diff = point2 - point1
    dot = np.dot(diff, diff)
    return np.sqrt(dot)


def parse_header(header_string):
    """
    Extracts end-systolic frame index and pixels to centimeters image caliber.
    :param header_string:
    :return: dict
    """
    numbers = list(map(int, header_string.split()))
    end_systolic_id = numbers[0]
    calibration_segment_coordinates = numbers[1:]
    pixels_per_cm = _get_length_from_segment_coordinates(calibration_segment_coordinates)
    header_info = {"end_systolic_id": end_systolic_id, "pixels_per_cm": pixels_per_cm}
    return header_info


def read_contour_file(path):
    """
    Reads information from the contour file
    :param path: path to the contour file
    :return: dict
    """
    patient_code, operation_status, wall_type = parse_filename(path.stem)
    with open(path, mode="r", encoding='UTF-8') as contour_file:
        header = contour_file.readline()
        header_info = parse_header(header)
    coordinates = np.loadtxt(path, skiprows=1).astype(int)
    coordinates = coordinates.reshape(coordinates.shape[0], -1, 2)[:, :, ::-1]
    contour_info = header_info.copy()
    contour_info["coordinates"] = coordinates
    contour_info["patient_code"] = patient_code
    contour_info["operation_status"] = operation_status
    contour_info["wall_type"] = wall_type
    return contour_info


def load_data(path, only_norm=False, only_endo=False):
    """
    Reads all the files in the path
    :param path: directory containing contour files
    :param only_norm: read only control group files
    :param only_endo: ready only endocardial files
    :return: list of dicts
    """
    def control_group_filter(filepath):
        return filepath.name.startswith("N")
    def endo_filter(filepath):
        return 'endo' in filepath.name
    if only_norm:
        group_filter = control_group_filter
    else:
        group_filter = None

    if only_endo:
        endo_filter = endo_filter
    else:
        endo_filter = None
    return [read_contour_file(p) for p in filter(endo_filter, filter(group_filter, sorted(path.iterdir())))]


def fourier_interpolate_coords(coords, n_harmonics=10, n_points_out=None):
    """
    :param coords: ndarray of shape n_time,n_points,space_dim
    :param n_harmonics: number of harmonics to use
    :param n_points_out: number of points to get in output
    :return: array of same shape n_time,n_points_out,space dim
        interpolated by n_harmonics harmonic functions
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


def prepare_contour(
        contour,
        scale=True,
        center=True,
        interpolate_time_points=100,
        fourier_interpolate_kwargs=True,
        close_contours=True,
):
    """
    Applies default normalization procedures to the contour record
    :param contour: contour dict
    :param scale: bool
        If True contour coordinates are scaled to the centimeter scale
        Else they are represented in pixel scale, which varies from record to record
    :param center: bool
        If True
    :param interpolate_time_points: int number of timepoints
        If nonzero time interpolation is performed and returned contour dict contains
         interpolate_time_points timepoints
    :param fourier_interpolate_kwargs: dict
        If not None fourier interpolation is performed which ensures continuity in time
    :param close_contours: bool
        If True adds the first point to the end of the contour array
        so that the array represents a closed shape
    :return: contour dict
    """
    coords = contour["coordinates"]
    coords_new = np.copy(coords)
    es_id = contour["end_systolic_id"]
    if center:
        coords_new = coords_new - np.mean(coords_new[0], axis=0, keepdims=True)
    if scale:
        scale = contour["pixels_per_cm"]
        coords_new = coords_new / scale
    if interpolate_time_points:
        coords_new = time_interpolation.interpolate_time(
            coords_new, end_systolic_id=es_id, n_time_points=interpolate_time_points
        )
        es_id = int(np.floor(interpolate_time_points * 0.35))
    if fourier_interpolate_kwargs is not None:
        coords_new = fourier_interpolate_coords(
            coords_new, **fourier_interpolate_kwargs
        )
    if close_contours:
        coords_new = np.append(coords_new, coords_new[:, [0]], axis=1)

    new_contour = contour.copy()
    new_contour["coordinates"] = coords_new
    new_contour["end_systolic_id"] = es_id
    return new_contour


def prepare_contours(
        contours,
        scale=True,
        center=True,
        interpolate_time_points=100,
        fourier_interpolate_kwargs=True,
        close_contours=True,
):
    """
    Applies default normalization procedures to the contour records
    :param contours: list of contour dicts
    :param scale: bool
        If True contour coordinates are scaled to the centimeter scale
        Else they are represented in pixel scale, which varies from record to record
    :param center: bool
        If True
    :param interpolate_time_points: int number of timepoints
        If nonzero time interpolation is performed and returned contour dict contains
         interpolate_time_points timepoints
    :param fourier_interpolate_kwargs: dict
        If not None fourier interpolation is performed which ensures continuity in time
    :param close_contours: bool
        If True adds the first point to the end of the contour array
        so that the array represents a closed shape
    :return: list of contour dicts
    """
    prepared_contours = [
        prepare_contour(
            contour,
            scale=scale,
            center=center,
            interpolate_time_points=interpolate_time_points,
            fourier_interpolate_kwargs=fourier_interpolate_kwargs,
            close_contours=close_contours,
        )
        for contour in contours
    ]
    return prepared_contours


def get_contour_title(contour):
    """
    Combines contour dict string values into one string
    :param contour: dict
    :return: string
    """
    return "_".join(
        [contour["patient_code"], contour["operation_status"], contour["wall_type"]]
    )


def get_contour_tensor(contours, only_endo=False):
    """
    Creates a tensor from a list of contour dicts
    :param contours: list of contour dicts
    :param only_endo: bool
        If true skips epicardial records
    :return: list of record names, tensor of contours
    """
    contours_dict = defaultdict(dict)
    for contour in contours:
        contour_id = contour["patient_code"] + "_" + contour["operation_status"]
        contours_dict[contour_id][contour["wall_type"]] = contour["coordinates"]
    contour_names = np.array(list(contours_dict.keys()))
    shape_series = []
    for val in contours_dict.values():
        endo, epi = val["endo"], val["epi"]
        if only_endo:
            full_contour = endo
        else:
            full_contour = np.hstack([endo, epi])
        shape_series.append(full_contour)
    shape_series = np.array(shape_series).transpose(
        0, 1, 2, 3
    )  # we need to have 2 before else

    return contour_names, shape_series
