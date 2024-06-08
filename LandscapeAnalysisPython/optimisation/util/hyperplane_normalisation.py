import numpy as np
import warnings


class HyperplaneNormalisation(object):

    def __init__(self, n_dim):

        self.ideal_point = np.full(n_dim, np.inf)
        self.worst_point = np.full(n_dim, -np.inf)

        self.nadir_point = None
        self.extreme_points = None

    def update(self, obj_array, nds=None):

        # Find or update the new ideal point from feasible solutions
        self.ideal_point = np.min(np.vstack((self.ideal_point, obj_array)), axis=0)
        self.worst_point = np.max(np.vstack((self.worst_point, obj_array)), axis=0)

        # Determine whether only non-dominated points or all points are used to determine the extreme points
        if nds is None:
            nds = np.arange(len(obj_array))

        # Find extreme points for normalisation
        self.extreme_points = get_extreme_points(obj_array[nds, :], self.ideal_point, extreme_points=self.extreme_points)

        # Find the intercepts for normalisation and calculate backup in case gaussian elimination fails
        worst_of_population = np.max(obj_array, axis=0)
        worst_of_front = np.max(obj_array[nds, :], axis=0)

        # Calculate nadir point
        self.nadir_point = get_nadir_point(self.extreme_points, self.ideal_point, self.worst_point,
                                           worst_of_population, worst_of_front)


def get_extreme_points(obj_array, ideal_point, extreme_points=None):

    # Calculate asf weights for extreme point decomposition
    weights = np.eye(obj_array.shape[1])
    weights[weights == 0.0] = 1e6

    # Add old extreme points for normalisation
    _obj_array = obj_array
    if extreme_points is not None:
        _obj_array = np.concatenate((extreme_points, _obj_array), axis=0)

    # Substitute small values with zero
    __obj_array = _obj_array - ideal_point
    __obj_array[__obj_array < 1e-3] = 0.0

    # Update the extreme points for the normalisation having the highest asf values
    obj_array_asf = np.max(__obj_array*weights[:, None, :], axis=2)

    # Extract extreme points
    indices = np.argmin(obj_array_asf, axis=1)
    extreme_points = _obj_array[indices, :]

    return extreme_points


def get_nadir_point(extreme_points, ideal_point, worst_point, worst_of_front, worst_of_population):

    try:
        # Find the intercepts using Gaussian elimination
        m = extreme_points - ideal_point
        b = np.ones(extreme_points.shape[1])
        plane = np.linalg.solve(m, b)
        warnings.simplefilter("ignore")
        intercepts = 1.0/plane

        # Calculate nadir point
        nadir_point = ideal_point + intercepts

        # Check if the hyperplane makes sense
        if not np.allclose(np.dot(m, plane), b) or np.any(intercepts <= 1e-6):
            raise np.linalg.LinAlgError()

        # If the nadir point should be larger than any value discovered, set it to the larger value
        mask = nadir_point > worst_point
        nadir_point[mask] = worst_point[mask]

    except np.linalg.LinAlgError:
        # Gaussian elimination failed, so fall back to worst of front
        nadir_point = worst_of_front

    # If the range is too small, replace with worst_of_population
    mask = nadir_point - ideal_point <= 1e-6
    nadir_point[mask] = worst_of_population[mask]

    return nadir_point




