import numpy as np
import skimage.transform


def procrustes_analysis(shape, reference_shape, scale=True, translate=True):
    if scale:
        ttype = 'similarity'
        transform = skimage.transform.SimilarityTransform()
    else:
        ttype = 'euclidean'
        transform = skimage.transform.EuclideanTransform()
    transform.estimate(shape, reference_shape)
    # transform = skimage.transform.estimate_transform(ttype, shape, reference_shape)
    if not translate:
        shape_translated = skimage.transform.EuclideanTransform(translation=transform.translation)(shape)
        # transform = skimage.transform.estimate_transform(ttype, shape_translated, reference_shape)
        transform.estimate(shape_translated, reference_shape)
    transformed_shape = transform(shape)
    error = procrustes_distance(transformed_shape, reference_shape)
    return transform, transform(shape)


def procrustes_distance(reference_shape, shape):
    """
    :param reference_shape:
    :param shape:
    :return:
    """
    dist = np.sum(np.linalg.norm(reference_shape - shape, axis=1))
    return dist


def generalized_procrustes_analysis(shapes, scale=True, translate=True, rtol=1e-5, atol=1e-8, max_iter=1000):
    """
    Performs superimposition on a set of
    shapes, calculates a mean shape.

    :param shapes: (a list of n_points,ndim Numpy arrays)
        Shapes to be aligned.
    :param max_iter: int
        Maximum number of iterations for finding global mean shape.
    :param atol: float
        Absolute tolerance for procrustes distance.
    :param rtol: float
        Relative tolerance for procrustes distance.
    :param translate: bool
        Apply translation in procrustes transformation.
    :param scale: bool
        Apply scaling in procrustes transformation.
    :return:
    mean: (2nx1 NumPy array)
        a new mean shape.
    aligned_shapes: (a list of n_points,ndim Numpy arrays)
        super-imposed shapes.
    """
    # initialize Procrustes distance
    current_distance = 0

    # initialize a mean shape
    mean_shape = np.array(shapes[0])

    num_shapes = len(shapes)

    # create array for new shapes, add
    new_shapes = np.zeros_like(shapes)

    for iter_id in range(max_iter):

        # add the mean shape as first element of array
        new_shapes[0] = mean_shape

        # superimpose all shapes to current mean
        for sh in range(1, num_shapes):
            try:
                new_sh = shapes[sh]
                _, new_sh = procrustes_analysis(new_sh, mean_shape, scale, translate)
                new_shapes[sh] = new_sh
            except ValueError as f:
                print(iter_id, sh)
                print(mean_shape, mean_shape.shape)
                print(shapes[sh], shapes[sh].shape)
                raise f

        # calculate new mean
        #         print(new_shapes.shape)
        #         return new_shapes
        new_mean = np.mean(new_shapes, axis=0)
        #         print(new_mean)

        new_distance = procrustes_distance(new_mean, mean_shape)

        # if the distance did not change, break the cycle
        if np.isclose(new_distance, current_distance, rtol=rtol, atol=atol):
            # print(new_distance)
            break

        # align the new_mean to old mean
        _, new_mean = procrustes_analysis(new_mean, mean_shape, scale, translate)

        # update mean and distance
        mean_shape = new_mean
        current_distance = new_distance

    return mean_shape, new_shapes


def linear_shift(shape_series, global_mean, scale=True, rtol=1e-5, atol=1e-8, max_iter=1000):
    local_gpas = [
        generalized_procrustes_analysis(shapes, scale=scale, rtol=rtol, atol=atol, max_iter=max_iter)[0]
        for shapes in shape_series
    ]
    local_gpas_aligned = [
        procrustes_analysis(lt, global_mean, scale=scale)[1] for lt in local_gpas
    ]
    linear_shifted = [
        [procrustes_analysis(frame, local_gpas_aligned[i], scale=scale)[1] - local_gpas_aligned[i] + global_mean
         for frame in shape_series[i]]
        for i in range(len(shape_series))
    ]
    return np.array(linear_shifted)


def linear_shift2(shape_series, global_mean, scale=True, rtol=1e-5, atol=1e-8, max_iter=1000):
    """
    difference is that here we will not be scaling inside the cycle
    :param shape_series:
    :param global_mean:
    :param scale:
    :param rtol:
    :param atol:
    :param max_iter:
    :return:
    """
    local_gpas = [
        generalized_procrustes_analysis(shapes, scale=False, rtol=rtol, atol=atol, max_iter=max_iter)[0]
        for shapes in shape_series
    ]
    alignments = []
    aligned_local_gpas = []
    for lt in local_gpas:
        alignment, aligned_lt = procrustes_analysis(lt, global_mean, scale=scale)
        alignments.append(alignment)
        aligned_local_gpas.append(aligned_lt)

    linear_shifted = [
        [alignments[i](procrustes_analysis(frame, local_gpas[i], scale=False)[1]) - aligned_local_gpas[i] + global_mean
         for frame in shape_series[i]]
        for i in range(len(shape_series))
    ]
    return np.array(linear_shifted)