from pathlib import Path
from shapely.geometry import Polygon, LineString
import numpy as np
from scipy.interpolate import interp1d

from reading_utils import prepare_contours, load_data, get_contour_tensor


def segment_to_vector(segment):
    return np.subtract(segment[1], segment[0])


def get_segment_len(segment):
    return np.linalg.norm(segment_to_vector(segment))


def normal_to_vector(vector, scale=True):
    normal = np.array([1, -np.divide(vector[0], vector[1])])
    if scale:
        return normal / np.linalg.norm(normal)
    return normal


def get_basal_midpoint(contour_points):
    return np.mean([contour_points[0], contour_points[-1]], axis=0)


def get_apex_id(contour_points):
    return (len(contour_points) - 1) // 2


def get_long_axis(contour_points):
    basal_midpoint = get_basal_midpoint(contour_points)
    return np.array([contour_points[len(contour_points) // 2], basal_midpoint])


def get_apex_id(contour_points):
    return len(contour_points) // 2


def get_apex_point(contour_points):
    apex_id = get_apex_id(contour_points)
    return contour_points[apex_id]


def get_short_axis(contour_points, long_axis=None, la_intersection=2 / 3):
    if long_axis is None:
        long_axis = get_long_axis(contour_points)
    long_axis_vector = segment_to_vector(long_axis)
    long_axis_normal = normal_to_vector(long_axis_vector)
    long_axis_len = np.linalg.norm(long_axis_vector)

    sa_la_intersection_point = np.add(long_axis[0], long_axis_vector * la_intersection)
    la_normal_linestring = LineString([
        sa_la_intersection_point - long_axis_normal * long_axis_len,
        sa_la_intersection_point + long_axis_normal * long_axis_len
    ])
    contour_poly = Polygon(contour_points)
    sa_linestring = la_normal_linestring.intersection(contour_poly)
    short_axis = np.array(sa_linestring.xy).T[:, :2]
    return short_axis


def get_sphericity_index(contour_points):
    la = get_long_axis(contour_points)
    sa = get_short_axis(contour_points, la)
    return get_segment_len(sa) / get_segment_len(la)


def get_gibson_sphericity_index(contour_points):
    contour_poly = Polygon(contour_points)
    contour_perimeter = contour_poly.length
    contour_area = contour_poly.area
    circle_radius = contour_perimeter / 2 / np.pi
    circle_area = np.pi * (circle_radius ** 2)
    return contour_area / circle_area


def get_conicity_index():
    pass


def car2pol(coords, center=None):
    if center is None:
        center = np.mean(coords, axis=0)
    coords_c = np.subtract(coords, center)
    r = np.linalg.norm(coords_c, axis=1)
    angle = np.arctan2(coords_c[:, 1], coords_c[:, 0])
    return r, angle


def make_contour_angle_increasing(angle):
    angle = np.array(angle)
    angle[np.where(angle < angle[0])] += 2 * np.pi
    return angle


def convert_to_polar(contour_points_time):
    ed_contour = contour_points_time[0]
    _, phi_ed0 = car2pol([ed_contour[0]], [0, 0])

    rs = []
    phis = []
    for frame in contour_points_time:
        r, phi = car2pol(frame, [0, 0])
        rs.append(r)
        phi_increasing_normalized = make_contour_angle_increasing(phi) - phi_ed0
        phis.append(phi_increasing_normalized)
    contour_points_time_polar = np.array([rs, phis]).transpose(1, 2, 0)
    return contour_points_time_polar


def get_triangle_areas(contour_points_polar):
    r, phi = contour_points_polar.T
    areas_tr = r[:-1] * r[1:] * np.sin(np.diff(phi)) / 2
    areas_tr = np.concatenate([[0], areas_tr])
    return areas_tr


def get_cumulative_areas(contour_points_polar):
    return np.cumsum(get_triangle_areas(contour_points_polar))


def get_region_angles(contour_points_polar, n_regions):
    r, phi = contour_points_polar.T
    cum_ar = get_cumulative_areas(contour_points_polar)
    area2angle_function = interp1d(cum_ar, phi)
    sample_areas = np.linspace(0, cum_ar[-1], n_regions, endpoint=False)
    sample_angles = area2angle_function(sample_areas)
    region_angles = np.concatenate([sample_angles[1:], [phi[-1]]])
    return region_angles


def get_regional_areas(contour_points_polar, region_angles):
    r, phi = contour_points_polar.T
    cum_ar = get_cumulative_areas(contour_points_polar)
    angle2area_function = interp1d(phi, cum_ar, bounds_error=False, fill_value=(0, cum_ar[-1]))
    regional_cum_areas = angle2area_function(region_angles)
    regional_areas = np.concatenate([[regional_cum_areas[0]], np.diff(regional_cum_areas)])
    return regional_areas


def get_regional_areas_time(contour_points_time_polar, n_regions):
    region_angles = get_region_angles(contour_points_time_polar[0], n_regions)
    regional_areas_time = []
    for frame in contour_points_time_polar:
        regional_areas = get_regional_areas(frame, region_angles)
        regional_areas_time.append(regional_areas)
    regional_areas_time = np.array(regional_areas_time)
    return regional_areas_time


def get_local_ES(regional_areas_time):
    return np.argmin(regional_areas_time, axis=0)


def get_local_EF(regional_areas_time):
    local_es = get_local_ES(regional_areas_time)
    local_esv = regional_areas_time[(local_es, np.arange(len(local_es)))]
    local_edv = regional_areas_time[0]
    local_sv = local_edv - local_esv
    local_ef = local_sv / local_edv
    return local_ef


def get_spacial_heterogeneity_index(contour_points_time, n_regions=20):
    contour_points_time_polar = convert_to_polar(contour_points_time)
    regional_areas_time = get_regional_areas_time(contour_points_time_polar, n_regions)
    local_ef = get_local_EF(regional_areas_time)
    return np.std(local_ef) / np.mean(local_ef)


def get_temporal_heterogeneity_index(contour_points_time, end_systolic_id, n_regions=20):
    contour_points_time_polar = convert_to_polar(contour_points_time)
    regional_areas_time = get_regional_areas_time(contour_points_time_polar, n_regions)
    local_es = get_local_ES(regional_areas_time)
    time_ratio = local_es / end_systolic_id
    return np.std(time_ratio) / np.mean(time_ratio)


if __name__ == '__main__':
    contour_dir = Path('../../raw_data/Data_strain')

    contours = prepare_contours(
        load_data(contour_dir, only_norm=False),
        scale=True, center=True,
        interpolate_time_points=None,
        fourier_interpolate_kwargs=None,
        close_contours=False
    )
