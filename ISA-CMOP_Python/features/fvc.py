from features.feature_helpers import remove_imag_rows, corr_coef
from optimisation.util.non_dominated_sorting import NonDominatedSorting
import numpy as np


def fvc(pop):
    """
    Compute correlation between objective values and norm violation values using Spearman's rank correlation coefficient.
    """

    obj = pop.extract_obj()
    cons = pop.extract_cons()
    cv = pop.extract_cv()

    # Remove imaginary rows. Deep copies are created here.
    obj = remove_imag_rows(obj)
    cons = remove_imag_rows(cons)  # may be able to remove this.

    # Initialise correlation between objectives.
    corr_obj = np.zeros(1, obj.shape[0])

    # Find correlations of each objective function with the CVs.
    for i in range(obj.shape[1]):
        objx = obj[:, i]
        corr_obj[i] = corr_coef(cv, objx)

    # Find Spearman's correlation between constrained and unconstrained fronts.
    # NDSort. Need to make sure this outputs a NumPy array for conditional indexing to work.
    ranksort = NonDominatedSorting().do(
        obj, cons_val=None, n_stop_if_ranked=obj.shape[0]
    )
    corr_f = corr_coef(cv, ranksort)

    return [corr_obj, corr_f]
