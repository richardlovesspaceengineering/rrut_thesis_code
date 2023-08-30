import numpy as np
from feature_helpers import remove_imag_rows
from optimisation.util.non_dominated_sorting import NonDominatedSorting
from scipy.spatial.distance import cdist
from scipy.stats import iqr


def f_decdist(pop, n1, n2):
    """
    Properties of the Pareto-Set (PS). Includes global maximum, global mean and mean IQR of distances across the PS.

    For our application, we set n2 = n1 = 1 i.e we want the distances between all of the decision variables on the PS (corresponding to the PF in decision space)
    """
    objvar = pop.extract_obj()
    decvar = pop.extract_var()

    # Remove imaginary rows. Deep copies are created here.
    objvar = remove_imag_rows(objvar)
    decvar = remove_imag_rows(decvar)

    # Initialize metrics.
    PSdecdist_max = 0
    PSdecdist_mean = 0
    PSdecdist_iqr_mean = 0

    if objvar.size > 1:
        # NDSort. Need to make sure this outputs a NumPy array for conditional indexing to work.
        ranksort = NonDominatedSorting.fast_non_dominated_sort(objvar)

        # Distance across and between n1 and n2 rank fronts in decision space. Each argument of cdist should be arrays corresponding to the DVs on front n1 and front n2.
        dist_matrix = cdist(
            decvar[ranksort == n1, :], decvar[ranksort == n2, :], "euclidean"
        )

        # Compute statistics on this dist_matrix.
        PSdecdist_max = np.max(np.max(dist_matrix, axis=0))
        PSdecdist_mean = np.mean(dist_matrix, axis=None)

        # Take IQR of each column, then take mean of IQRs to get a scalar.
        PSdecdist_iqr_mean = np.mean(iqr(dist_matrix, axis=0), axis=None)

    return [PSdecdist_max, PSdecdist_mean, PSdecdist_iqr_mean]
