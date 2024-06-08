import numpy as np


def randomized_argsort(a, order='ascending'):

    permutations = np.random.permutation(len(a))
    idx_arr = np.argsort(a[permutations], kind='quicksort')
    idx_arr = permutations[idx_arr]

    if order == 'ascending':
        return idx_arr
    elif order == 'descending':
        return np.flip(idx_arr, axis=0)
    else:
        raise Exception('Unknown sorting order')
