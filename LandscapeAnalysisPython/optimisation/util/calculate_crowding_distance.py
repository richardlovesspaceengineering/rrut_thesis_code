import numpy as np

from optimisation.util.misc import find_duplicates


def calculate_crowding_distance(obj, filter_out_duplicates=True):

    # Extracting size of passed population and number of objective functions
    n_points, n_obj = obj.shape

    # Can only calculate crowding for > 2 points
    if n_points <= 2:
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

        # Sort each column and get index
        index_arr = np.argsort(_obj, axis=0, kind='mergesort')

        # Sort the objective space values for the whole array
        _obj = _obj[index_arr, np.arange(n_obj)]

        # Calculate the distance from each point to the last and next points
        dist = np.row_stack([_obj, np.full(n_obj, np.inf)]) - np.row_stack([np.full(n_obj, -np.inf), _obj])

        # Calculate the norm for each objective, set to NaN if all values are equal
        norm = np.max(_obj, axis=0) - np.min(_obj, axis=0)
        norm[norm == 0] = np.nan

        # Distance to last array
        dist_to_last = dist[:-1]/norm

        # Distance to next array
        dist_to_next = np.copy(dist)[1:]/norm

        # If we divided through by zero because all values in one column are equal, replace by zeros
        dist_to_last[np.isnan(dist_to_last)] = 0.0
        dist_to_next[np.isnan(dist_to_next)] = 0.0

        # Sort the list
        index_arr_sorted = np.argsort(index_arr, axis=0)

        # Calculate crowding distance
        crowding_distance = np.sum(dist_to_last[index_arr_sorted, np.arange(n_obj)] + dist_to_next[index_arr_sorted, np.arange(n_obj)], axis=1)/n_obj

        # Output the final vector which sets the crowding distance for duplicates to zero to be eliminated
        crowding = np.zeros(n_points)
        crowding[is_unique] = crowding_distance

    # Replace infinite crowding values with large number
    crowding[np.isinf(crowding)] = 1.0e14

    return crowding

