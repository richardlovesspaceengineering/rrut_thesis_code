import numpy as np

from optimisation.util.misc import find_duplicates


def calculate_hypervolume(obj, filter_out_duplicates=True):

    # Extracting size of passed population and number of objective functions
    n_points, n_obj = obj.shape

    # Can only calculate hypervolume for > 2 points and >= 2 dimensions
    if n_points <= 2 or n_obj < 2:
        return np.full(n_points, np.inf)
    else:

        if filter_out_duplicates:
            # filter out solutions which are duplicates - duplicates get a zero finally
            is_unique = np.where(np.logical_not(find_duplicates(obj, epsilon=1e-24)))[0]
        else:
            # set every point to be unique without checking it
            is_unique = np.arange(n_points)

        # Extract unique points of object array
        _obj = obj[is_unique]

        # Calculate nadir point (the worst objective function value in each of the dimensions)
        nadir_pt = np.array([np.max(obj[:, i]) for i in range(n_obj)])
        # Calculate reference point (minimisation is assumed here) - shifted to a slightly worse location than nadir pt
        ref_pt = nadir_pt + np.ones(len(nadir_pt))

        # Delta S-metric (delta hypervolume measure)
        _delta_s = np.zeros(len(_obj))

        if n_obj == 2:

            # Sort solutions according to objective value 1 (they will then be in inverse order for objective 2, as
            # for each front, each point is non-dominated by another)
            index_arr = np.argsort(_obj, axis=0)
            _obj = _obj[index_arr[:, 0], :]

            # Calculate delta S-metric for each point on the front
            for i in range(len(_obj)):
                if i == 0:
                    _delta_s[i] = (_obj[i+1, 0] - _obj[i, 0])*(ref_pt[1] - _obj[i, 1])
                elif i == len(_obj) - 1:
                    _delta_s[i] = (ref_pt[0] - _obj[i, 0])*(_obj[i-1, 1] - _obj[i, 1])
                else:
                    _delta_s[i] = (_obj[i+1, 0] - _obj[i, 0])*(_obj[i-1, 1] - _obj[i, 1])
        else:
            # Todo: Need to implement method to calculate delta_s for each point in > 2 dimensions
            raise Exception('Need to implement method to calculate delta_s for each point in > 2 dimensions')

        # Output the final vector which sets the crowding distance for duplicates to zero to be eliminated
        delta_s = np.zeros(n_points)
        delta_s[is_unique] = _delta_s

    return delta_s

