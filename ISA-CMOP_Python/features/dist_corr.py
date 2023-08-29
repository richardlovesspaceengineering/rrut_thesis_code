from feature_helpers import remove_imag_rows
from scipy.spatial.distance import cdist
import numpy as np
import warnings
import scipy


def dist_corr(pop, NonDominated, significance_level=0.05):
    """
    Distance correlation.

    Distance for each solution to nearest global solution in decision space. Correlation of distance and constraints norm.

    Fairly certain that NonDominated is an instance of the Population class containing the non-dominated solutions
    """

    objvar = pop.extract_obj()
    decvar = pop.extract_var()
    consvar = pop.extract_cons()

    # Remove imaginary rows.
    objvar = remove_imag_rows(objvar)
    decvar = remove_imag_rows(decvar)
    consvar = remove_imag_rows(consvar)

    # For each ND decision variable, find the smallest distance to the nearest population decision variable.
    dist_matrix = cdist(NonDominated.extract_var(), decvar, "euclidean")
    min_dist = np.min(dist_matrix, axis=0)

    # Then compute correlation coefficient. Assumed that all values in the correlation matrix are the same, meaning we only need one scalar. See f_corr for a similar implementation.
    with warnings.catch_warnings():
        # Suppress warnings where corr is NaN - will just set to 0 in this case.
        warnings.simplefilter("ignore", scipy.stats.ConstantInputWarning)
        result = scipy.stats.pearsonr(NORMCVS, min_dist)

    dist_c_corr = result.statistic
    pvalue = result.pvalue

    # Signficance test. Alsouly does p > alpha for some reason.
    if pvalue < significance_level:
        corr_obj = 0

    elif np.isnan(dist_c_corr):
        # Make correlation 0 if there is no change in one vector.
        dist_c_corr = 0

    return [dist_c_corr]
