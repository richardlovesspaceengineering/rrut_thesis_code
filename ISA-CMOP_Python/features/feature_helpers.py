import numpy as np
import copy


def remove_imag_rows(matrix):
    """
    Remove rows which have at least one imaginary value in them
    """

    matrix = copy.deepcopy(matrix)
    rmimg = np.nonzero(np.sum(np.imag(matrix) != 0, axis=1))
    matrix = np.delete(matrix, rmimg, axis=0)
    return matrix
