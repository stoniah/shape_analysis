from shapely.geometry import Polygon, LineString, Point
from shapely import ops
import numpy as np
from scipy.interpolate import interp1d


def segment_to_vector(segment):
    """
    :param segment: numpy array [[x1, y1], [x2, y2]]
    :return: numpy array [x2 - x1, y2 - y1]
    """
    return np.subtract(segment[1], segment[0])


def get_segment_len(segment):
    """

    :param segment: numpy array [[x1, y1], [x2, y2]]
    :return: float
    """
    return np.linalg.norm(segment_to_vector(segment))


def normal_to_vector(vector, scale=True):
    """
    Calculates a normal vector to the given vector
    :param vector: numpy array [x1, y1]
    :param scale: bool
    :return: numpy array [x1, y1]
        normal to the vector
        if scale is True, the normal is of unit length
    """
    normal = np.array([1, -np.divide(vector[0], vector[1])])
    if scale:
        return normal / np.linalg.norm(normal)
    return normal


def get_basal_midpoint(contour_points):
    """
    Finds a point in the middle of the LV base
    :param contour_points: (n_points, 2) array, LV shape
    :return: numpy array [x1, y1]
    """
    return np.mean([contour_points[0], contour_points[-1]], axis=0)


def get_apex_id(contour_points):
    """
    Returns the index of apex point in shape points array
    The convention is that shapes are given by 2*n + 1 point arrays,
    and the apex is in the middle of the array
    :param contour_points: (n_points, 2) array, LV shape
    :return: int
    """
    return (len(contour_points) - 1) // 2


def get_long_axis(contour_points):
    """
    Finds the long axis of the LV shape defined as
    a segment between the apex and the basal midpoint
    :param contour_points: (n_points, 2) array, LV shape
    :return: numpy array [[x1, y1], [x2, y2]]
    """
    basal_midpoint = get_basal_midpoint(contour_points)
    apex_id = get_apex_id(contour_points)
    apex = contour_points[apex_id]
    return np.array([apex, basal_midpoint])


def get_apex_point(contour_points):
    """
    Returns the coordinates of the LV apex
    :param contour_points: (n_points, 2) array, LV shape
    :return: numpy array [x1, y1]
    """
    apex_id = get_apex_id(contour_points)
    return contour_points[apex_id]


def get_short_axis(contour_points, long_axis=None, la_intersection=2 / 3):
    """
    Calculates the short axis of the LV. Defined as the segment between LV walls
    perpendicular and intersecting the long axis at the la_intersection %.
    :param contour_points: (n_points, 2) array, LV shape
    :param long_axis: numpy array ([x1, y1], [x2, y2])
        the long axis of the LV
        Calculated from contour_points if None
    :param la_intersection: float in [0, 1]
        the point of intersection with the long axis taken from the apex
    :return: numpy array ([x1, y1], [x2, y2])
    """
    if long_axis is None:
        long_axis = get_long_axis(contour_points)
    long_axis_vector = segment_to_vector(long_axis)
    long_axis_normal = normal_to_vector(long_axis_vector)
    long_axis_len = np.linalg.norm(long_axis_vector)

    sa_la_intersection_point = np.add(long_axis[0], long_axis_vector * la_intersection)
    la_normal_linestring = LineString(
        [
            sa_la_intersection_point - long_axis_normal * long_axis_len,
            sa_la_intersection_point + long_axis_normal * long_axis_len,
        ]
    )
    contour_poly = Polygon(contour_points)
    sa_linestring = la_normal_linestring.intersection(contour_poly)
    short_axis = np.array(sa_linestring.xy).T[:, :2]
    return short_axis


def get_sphericity_index(contour_points):
    """
    Calculates the sphericity index of the LV,
    short to long axis ratio. Typical values range from 0 to 1,
    closer to 0 means less spherical, closer to 1 more spherical.
    If the short axis is larger than the long axis, the index will be larger than 1.
    :param contour_points: (n_points, 2) array, LV shape
    :return: float
        the sphericity index of the LV
    """
    la = get_long_axis(contour_points)
    sa = get_short_axis(contour_points, la)
    return get_segment_len(sa) / get_segment_len(la)


def get_gibson_sphericity_index(contour_points):
    """
    Calculates Gibson sphericity index. It is the ratio of the shape area
    to the area of the circle with the same perimeter.
    Values range from 1 for a circle to 0 if the shape is a straight line.
    :param contour_points: (n_points, 2) array, LV shape
    :return: float
        the Gibson sphericity index of the LV
    """
    contour_poly = Polygon(contour_points)
    contour_perimeter = contour_poly.length
    contour_area = contour_poly.area
    circle_radius = contour_perimeter / 2 / np.pi
    circle_area = np.pi * (circle_radius ** 2)
    return contour_area / circle_area


def get_apical_region(contour_points, bound=1 / 3, short_segment=None):
    """
    Returns the points from the apical region outline. If we take a point on the LV long axis
    and slice the LV shape perpendicular to the long axis at this point, then the part containing
    the apex point is defined as the apical region.
    :param contour_points: (n_points, 2) array, LV shape
    :param bound: float in (0,1),
        where to slice the long axis
    :param short_segment: numpy array ([x1, y1], [x2, y2])
        the short axis of the LV
        Calculated from contour_points and bound if None
    :return: (m_points, 2) array
        LV apical region points
    """
    if short_segment is None:
        short_segment = get_short_axis(contour_points, la_intersection=bound)
    short_segment_center = np.mean(short_segment, axis=0)
    short_segment_vector = segment_to_vector(short_segment)
    bound_segment = np.array(
        [
            short_segment_center - short_segment_vector,
            short_segment_center + short_segment_vector,
        ]
    )
    shape_poly = Polygon(contour_points)
    bound_segment_linestring = LineString(bound_segment)
    polys = ops.split(shape_poly, bound_segment_linestring)
    bound_center = np.mean(bound_segment, axis=0)
    centers = np.array([np.mean(poly.boundary.xy, axis=1) for poly in polys.geoms])
    lowest_poly_id = np.where(bound_center[1] > centers[:, 1])[0][0]
    lowest_poly = polys.geoms[lowest_poly_id]
    return np.array(lowest_poly.boundary.xy).T[:-1]


def get_conicity_index(contour_points):
    """
    Calculates the LV conicity index. It is defined as the ratio of two areas.
    First area is the area of the region from the apex to the half of the long axis.
    The second area is the area of a triangle formed by the apex and the short axis at the half of the long axis.
    Values closer to 1 are more conical, larger is less conical.
    :param contour_points: (n_points, 2) array, LV shape
    :return: float
        LV conicity index
    """
    long_axis = get_long_axis(contour_points)
    long_axis_len = get_segment_len(long_axis)
    long_axis_bound = .5
    short_axis = get_short_axis(contour_points, long_axis=long_axis, la_intersection=long_axis_bound)
    short_axis_len = get_segment_len(short_axis)
    apical_region = get_apical_region(contour_points, short_segment=short_axis)
    apical_region_area = Polygon(apical_region).area

    triangle_area = long_axis_bound * long_axis_len * short_axis_len * .5

    return apical_region_area / triangle_area


def get_eccentricity_index(contour_points):
    """
    Calculates the LV eccentricity index. It is defined by an elliptical model
    where the LV long axis represents the ellipse major axis, and the LV short axis
    represents the ellipse minor axis. Then we take the standard eccentricity of this ellipse.
    Values range from 0 for a circle to 1 for a line.
    :param contour_points: (n_points, 2) array, LV shape
    :return: float
        LV eccentricity index
    """
    long_axis = get_long_axis(contour_points)
    long_axis_len = get_segment_len(long_axis)
    long_axis_bound = .5
    short_axis = get_short_axis(contour_points, long_axis=long_axis, la_intersection=long_axis_bound)
    short_axis_len = get_segment_len(short_axis)

    return np.sqrt(long_axis_len ** 2 - short_axis_len ** 2) / long_axis_len


def car2pol(coords, center=None):
    """
    Converts cartesian coordinate array to polar coordinates (radius and angle)
    :param coords: (n_points, 2) coordinate array
    :param center: [x1, y1] array
        Is subtracted from the input coordinate array before conversion,
        If None taken as a center of mass.
    :return: (n_points,) (n_points,) arrays
        radius and angle polar coordinates of the input points
    """
    if center is None:
        center = np.mean(coords, axis=0)
    coords_c = np.subtract(coords, center)
    r = np.linalg.norm(coords_c, axis=1)
    angle = np.arctan2(coords_c[:, 1], coords_c[:, 0])
    return r, angle


def make_contour_angle_increasing(angle):
    """
    Shifts the angle array, so it becomes an increasing function
    :param angle: (n_points,) array
    :return: (n_points,) array
    """
    angle = np.array(angle)
    angle[np.where(angle < angle[0])] += 2 * np.pi
    return angle


def convert_to_polar(contour_points_time, center=True):
    """
    Converts the LV heart cycle array to polar coordinates.
    We need to do it simultaneously for the whole cycle
    to be able to compare frames in the cycle by angle.
    :param contour_points_time: (time, n_points, 2)
    :param center: bool (default is True)
        If True every contour in the cycle is centered at the origin
        otherwise, the translation differences between frames are left untouched
    :return: (time, n_points, 2) float array
    """
    if center:
        contour_points_time_centered = contour_points_time - np.mean(contour_points_time, axis=1, keepdims=True)
    ed_contour = contour_points_time_centered[0]
    _, phi_ed0 = car2pol([ed_contour[0]], [0, 0])

    rs = []
    phis = []
    for frame in contour_points_time_centered:
        r, phi = car2pol(frame, [0, 0])
        rs.append(r)
        phi_increasing_normalized = make_contour_angle_increasing(phi) - phi_ed0
        phis.append(phi_increasing_normalized)
    contour_points_time_polar = np.array([rs, phis]).transpose(1, 2, 0)
    return contour_points_time_polar


def get_triangle_areas(contour_points_polar):
    """
    Calculates areas of the triangles formed by the center and neighbouring points of the contour
    :param contour_points_polar: (n_points, 2)
    :return: (n_points,) float array
        Areas
    """
    r, phi = contour_points_polar.T
    areas_tr = r[:-1] * r[1:] * np.sin(np.diff(phi)) / 2
    areas_tr = np.concatenate([[0], areas_tr])
    return areas_tr


def get_cumulative_areas(contour_points_polar):
    """
    Calculates cumulative areas of the triangles formed by the center and neighbouring points of the contour
    :param contour_points_polar: (n_points, 2)
        LV shape in polar coordinates
    :return: (n_points,) float array
        Cumulative areas
    """
    return np.cumsum(get_triangle_areas(contour_points_polar))


def get_region_angles(contour_points_polar, n_regions):
    """
    Divides the LV contour into n_regions sectors of equal area
    :param contour_points_polar: (n_points, 2)
        LV shape in polar coordinates
    :param n_regions: int
        Number of regions to divide into
    :return: (n_regions,) float array,
        Array of angles corresponding to the regions
    """
    r, phi = contour_points_polar.T
    cum_ar = get_cumulative_areas(contour_points_polar)
    area2angle_function = interp1d(cum_ar, phi)
    sample_areas = np.linspace(0, cum_ar[-1], n_regions, endpoint=False)
    sample_angles = area2angle_function(sample_areas)
    region_angles = np.concatenate([sample_angles[1:], [phi[-1]]])
    return region_angles


def get_regional_areas(contour_points_polar, region_angles):
    """
    Calculates the areas in the LV regions defined by the angles
    :param contour_points_polar: (n_points, 2)
        LV shape in polar coordinates
    :param region_angles: (n_regions,) float array,
        Array of angles corresponding to the regions
    :return: (n_regions,) flaot array
        Areas in the regions
    """
    r, phi = contour_points_polar.T
    cum_ar = get_cumulative_areas(contour_points_polar)
    angle2area_function = interp1d(
        phi, cum_ar, bounds_error=False, fill_value=(0, cum_ar[-1])
    )
    regional_cum_areas = angle2area_function(region_angles)
    regional_areas = np.concatenate(
        [[regional_cum_areas[0]], np.diff(regional_cum_areas)]
    )
    return regional_areas


def get_regional_areas_time(contour_points_time_polar, n_regions):
    """
    Divides the lv shape into n_regions sectors of equal areas and calculates the areas
    throughout the cardiac cycle
    :param contour_points_time_polar: (time, n_points, 2)
        LV shape dynamics in polar coordinates
    :param n_regions: int
        number of regions to divide the shape into
    :return: (time, n_regions,) float array
        Area change in the LV regions
    """
    region_angles = get_region_angles(contour_points_time_polar[0], n_regions + 2)
    regional_areas_time = []
    for frame in contour_points_time_polar:
        regional_areas = get_regional_areas(frame, region_angles)
        regional_areas_time.append(regional_areas)
    regional_areas_time = np.array(regional_areas_time)[:,1:-1]
    return regional_areas_time


def _get_local_ES(regional_areas_time):
    """
    Finds the local end-systoles (ES) in the LV regions.
    Local ES is defined as the id of the frame with minimal area for the particular region.
    :param regional_areas_time: (time, n_regions) float array
        Area change in the LV regions
    :return: (n_regions) int array
        local end-systole ids
    """
    return np.argmin(regional_areas_time, axis=0)


def get_local_ES(contour_points_time, n_regions=20, center=True):
    """
    Finds the local end-systoles (ES) in the LV regions.
    Local ES is defined as the id of the frame with minimal area for the particular region.
    :param contour_points_time: (time, n_points, 2) float array
      LV shape dynamics
    :param n_regions: int
      number of regions to divide into
    :param center: bool (default is True)
        If True every contour in the cycle is centered at the origin
        otherwise, the translation differences between frames are left untouched
    :return: (n_regions) float array
    """
    contour_points_time_polar = convert_to_polar(contour_points_time, center)
    regional_areas_time = get_regional_areas_time(contour_points_time_polar, n_regions)
    local_es = _get_local_ES(regional_areas_time)
    return local_es


def _get_local_EF(regional_areas_time):
    """
    Finds the local ejection fraction (EF) in the LV regions.
    EF is defined as the ratio (end-diastolic volume - end-systolic volume) / (end-diastolic volume).
    Here we take local areas instead of volumes, end-diastolic frame is the first frame in the cycle,
    and end-systolic frame is the frame with the lowest area for the region.
    :param regional_areas_time: (time, n_regions) float array
        Area change in the LV regions
    :return: (n_regions) float array
        Local ejection fractions
    """
    local_es = _get_local_ES(regional_areas_time)
    local_esv = regional_areas_time[(local_es, np.arange(len(local_es)))]
    local_edv = regional_areas_time[0]
    local_sv = local_edv - local_esv
    local_ef = local_sv / local_edv
    return local_ef


def get_local_EF(contour_points_time, n_regions=20, center=True):
    """
    Finds the local ejection fraction (EF) in the LV regions.
    EF is defined as the ratio (end-diastolic volume - end-systolic volume) / (end-diastolic volume).
    Here we take local areas instead of volumes, end-diastolic frame is the first frame in the cycle,
    and end-systolic frame is the frame with the lowest area for the region.
    :param contour_points_time: (time, n_points, 2) float array
      LV shape dynamics
    :param n_regions: int
      number of regions to divide into
    :param center: bool (default is True)
        If True every contour in the cycle is centered at the origin
        otherwise, the translation differences between frames are left untouched
    :return: (n_regions) float array
      Local ejection fractions
    """
    contour_points_time_polar = convert_to_polar(contour_points_time, center)
    regional_areas_time = get_regional_areas_time(contour_points_time_polar, n_regions)
    local_ef = _get_local_EF(regional_areas_time)
    return local_ef


def get_spacial_heterogeneity_index(contour_points_time, n_regions=20, center=True, local_ef=None):
    """
    Calculates spacial heterogeneity index (SHI) which is defined as the coefficient
    of variation of local ejection fractions in n_regions.
    :param contour_points_time: (time, n_points, 2) float array
        LV shape dynamics
    :param n_regions: int
        number of regions to divide into
    :param center: bool (default is True)
        If True every contour in the cycle is centered at the origin
        otherwise, the translation differences between frames are left untouched
    :param local_ef: (n_regions,) float array (default is None)
        If None it is calculated
        Can be supplied to avoid repeating computations
    :return: float
    """
    if local_ef is None:
        local_ef = get_local_EF(contour_points_time, n_regions, center)
    return np.std(local_ef, ddof=1) / np.mean(local_ef)


def get_local_asynchronism_indexes(contour_points_time, end_systolic_id, n_regions=20, center=True):
    """
    Calculates local asynchronism indexes for every region, defined as the ratio of local ES id to global ES id.
    :param contour_points_time: (time, n_points, 2) float array
        LV shape dynamics
    :param end_systolic_id: the global end-systolic frame
    :param n_regions: int,
        number of regions to divide the shape into
    :param center: bool (default is True)
        If True every contour in the cycle is centered at the origin
        otherwise, the translation differences between frames are left untouched
    :return: (n_regions,) float array
    """
    local_es = get_local_ES(contour_points_time, n_regions, center)
    time_ratio = local_es / end_systolic_id
    return time_ratio


def get_temporal_heterogeneity_index(
        contour_points_time, end_systolic_id, n_regions=20, center=True, time_ratio=None
):
    """
    Calculates temporal heterogeneity index which is defined as the coefficient of variation
    of local asynchronism indexes.
    :param contour_points_time: (time, n_points, 2) float array
        LV shape dynamics
    :param end_systolic_id:
        the global end-systolic frame
    :param n_regions: int,
        number of regions to divide the shape into
    :param center: bool (default is True)
        If True every contour in the cycle is centered at the origin
        otherwise, the translation differences between frames are left untouched
    :param time_ratio: (n_regions,) flaot array (default is None)
        if None it is calculated from other parameters
        can be supplied to avoid repeating computations
    :return: float
    """
    if time_ratio is None:
        time_ratio = get_local_asynchronism_indexes(contour_points_time, end_systolic_id, n_regions)
    return np.std(time_ratio, ddof=1) / np.mean(time_ratio)


def close_contour_flat(contour_points, n_points_cap=10):
    """
    Auxiliary function, adds a flat cap on the base of the LV, closing the shape
    :param contour_points: (n_points, 2) array, LV shape
    :param n_points_cap: int, number of points in the cap
    :return: (n_points + n_points_cap, 2) array, closed LV shape
    """
    closure_points = np.linspace(contour_points[-1], contour_points[0], n_points_cap + 1)[1:]
    contour_points_closed = np.append(contour_points, closure_points, axis=0)
    return contour_points_closed


def fourier_series_coeff_numpy(f, T, N, return_complex=False):
    """
    Calculates first 2*N+1 Fourier series coeff of the function
    :param f: periodic function, callable
    :param T: period of the function, so that f(0)==f(T)
    :param N: method will return the first N + 1 Fourier coeff
    :param return_complex: bool
    :return:
        if return_complex == False, the function returns:

        a0 : float
        a,b : numpy float arrays describing respectively the cosine and sine coeff.

        if return_complex == True, the function returns:

        c : numpy 1-dimensional complex-valued array of size N+1
    """

    # We must use a sampling freq larger than the maximum
    # frequency we want to catch in the signal (Shanon theorem)
    f_sample = 2 * N
    # we also need to use an integer sampling frequency, so the
    # points will be equispaced between 0 and 1. We then add +2 to f_sample
    t, dt = np.linspace(0, T, f_sample + 2, endpoint=False, retstep=True)

    y = np.fft.rfft(f(t)) / t.size

    if return_complex:
        return y
    else:
        y *= 2
        return y[0].real, y[1:-1].real, -y[1:-1].imag


def get_contour_fourier_coeff(contour_points, n_coeff):
    """
    Calculates the first 2 * n_coeff Fourier series coeff of the LV shape
    :param contour_points: (n_points, 2) array, LV shape
    :param n_coeff: int
    :return:
        a0 : float
        a,b : numpy float arrays describing respectively the cosine and sine coeff.
    """
    contour_points_closed = close_contour_flat(contour_points)
    contour_points_polar = car2pol(contour_points_closed)
    r, phi = contour_points_polar
    phi = make_contour_angle_increasing(phi) - phi[0]
    f = interp1d(phi[:-1], r[:-1], fill_value='extrapolate')
    a0, a, b = fourier_series_coeff_numpy(f, 2 * np.pi, n_coeff, return_complex=False)
    return a0, a, b


def get_cycle_fourier_coeff(contour_points_time, n_coeff):
    """
    Calculates the first 2 * n_coeff Fourier series coeff of the LV shapes in the cycle
    :param contour_points_time: (time, n_points, 2) array, LV shape cycle
    :param n_coeff: int
    :return:
        a0 : (time,) numpy float array
        a, b : (time, n_coeff) numpy float arrays
    """
    a0_list, a_list, b_list = [], [], []
    for contour_points in contour_points_time:
        a0, a, b = get_contour_fourier_coeff(contour_points, n_coeff)
        a0_list.append(a0)
        a_list.append(a)
        b_list.append(b)
    return np.array(a0_list), np.array(a_list), np.array(b_list)


def get_fourier_shape_power_index(contour_points, n_coeff=8):
    """
    Calculates Fourier Shape Power Index (Kass, 1988)
    :param contour_points: (n_points, 2) array, LV shape
    :param n_coeff: int
    :return: float
    """
    a0, a, b = get_contour_fourier_coeff(contour_points, n_coeff)
    c = np.sqrt(a ** 2 + b ** 2)
    fspi = np.sum((c[1:] / a0) ** 2)
    return fspi


def reconstruct_contour_from_fourier(a0, a, b, n_points=200):
    """
    Given fourier coefficients a0, a, b reconstructs n_points points of LV shape
    :param a0: float
    :param a: numpy float array
    :param b: numpy float array
    :param n_points: float
        number of points to reconstruct
    :return: (n_points, 2) float numpy array
        reconstructed LV shape
    """
    ks = np.arange(1, len(a) + 1)
    phi = np.linspace(0, 2 * np.pi, n_points)
    r_reconstructed = []
    for t in phi:
        series = [a0 / 2, *(a * np.cos(t * ks) + b * np.sin(t * ks))]
        r_t = np.sum(series)
        r_reconstructed.append(r_t)
    return phi, r_reconstructed


if __name__ == "__main__":
    pass
