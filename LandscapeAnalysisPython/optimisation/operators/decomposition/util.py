import numpy as np


def calc_distance_to_weights(obj_array, weights, utopian_point):
    norm = np.linalg.norm(weights, axis=1)
    obj_array = obj_array - utopian_point

    d1 = (obj_array*weights).sum(axis=1) / norm
    d2 = np.linalg.norm(obj_array - (d1[:, None] * weights / norm[:, None]), axis=1)

    return d1, d2

