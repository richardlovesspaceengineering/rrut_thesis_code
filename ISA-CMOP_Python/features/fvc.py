from feature_helpers import remove_imag_rows, corr_coef
from optimisation.util.non_dominated_sorting import NonDominatedSorting
import numpy as np


def fvc(pop):
    """
    Compute correlation between objective values and norm violation values using Spearman's rank correlation coefficient.
    """

    objvar = pop.extract_obj()
    consvar = pop.extract_cons()

    # Remove imaginary rows. Deep copies are created here.
    objvar = remove_imag_rows(objvar)
    consvar = remove_imag_rows(consvar)  # may be able to remove this.

    # Get CV, a row vector containing the norm of the constraint violations. Assuming this can be standardised for any given problem setup.
    consvar[consvar <= 0] = 0
    cv = np.sum(consvar, axis=1)

    # Initialise correlation between objectives.
    corr_obj = np.zeros(1, objvar.shape[0])

    # Find correlations of each objective function with the CVs.
    for i in range(objvar.shape[1]):
        objx = objvar[:, i]
        corr_obj[i] = corr_coef(cv, objx)

    # Find Spearman's correlation between constrained and unconstrained fronts.
    # NDSort. Need to make sure this outputs a NumPy array for conditional indexing to work.
    ranksort = NonDominatedSorting.fast_non_dominated_sort(objvar)
    corr_f = corr_coef(cv, ranksort)

    return [corr_obj, corr_f]
