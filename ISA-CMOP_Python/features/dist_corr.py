from features.feature_helpers import remove_imag_rows, corr_coef
from scipy.spatial.distance import cdist
import numpy as np


def dist_corr(pop, NonDominated):
    """
    Distance correlation.

    Distance for each solution to nearest global solution in decision space. Correlation of distance and constraints norm.

    Fairly certain that NonDominated is an instance of the Population class containing the non-dominated solutions
    """

    obj = pop.extract_obj()
    var = pop.extract_var()
    cons = pop.extract_cons()  # constraint matrix.
    cv = pop.extract_cv()

    # Remove imaginary rows. Deep copies are created here.
    obj = remove_imag_rows(obj)
    var = remove_imag_rows(var)
    cons = remove_imag_rows(cons)

    # For each ND decision variable, find the smallest distance to the nearest population decision variable.
    dist_matrix = cdist(NonDominated.extract_var(), var, "euclidean")
    min_dist = np.min(dist_matrix, axis=0)

    # Then compute correlation coefficient between CV and . Assumed that all values in the correlation matrix are the same, meaning we only need one scalar. See f_corr for a similar implementation.
    dist_c_corr = corr_coef(cv, min_dist)
    return [dist_c_corr]
