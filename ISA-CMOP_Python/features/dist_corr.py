from feature_helpers import remove_imag_rows, corr_coef
from scipy.spatial.distance import cdist
import numpy as np


def dist_corr(pop, NonDominated, significance_level=0.05):
    """
    Distance correlation.

    Distance for each solution to nearest global solution in decision space. Correlation of distance and constraints norm.

    Fairly certain that NonDominated is an instance of the Population class containing the non-dominated solutions
    """

    objvar = pop.extract_obj()
    decvar = pop.extract_var()
    consvar = pop.extract_cons()  # constraint matrix.

    # Remove imaginary rows. Deep copies are created here.
    objvar = remove_imag_rows(objvar)
    decvar = remove_imag_rows(decvar)
    consvar = remove_imag_rows(consvar)

    # Get CV, a row vector containing the norm of the constraint violations. Assuming this can be standardised for any given problem setup.
    consvar[consvar <= 0] = 0
    cv = np.sum(consvar, axis=1)

    # For each ND decision variable, find the smallest distance to the nearest population decision variable.
    dist_matrix = cdist(NonDominated.extract_var(), decvar, "euclidean")
    min_dist = np.min(dist_matrix, axis=0)

    # Then compute correlation coefficient between CV and . Assumed that all values in the correlation matrix are the same, meaning we only need one scalar. See f_corr for a similar implementation.
    dist_c_corr = corr_coef(cv, min_dist)
    return [dist_c_corr]
