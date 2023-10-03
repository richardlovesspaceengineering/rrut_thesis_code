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
    corr_obj = np.zeros(obj.shape[0])

    # Find correlations of each objective function with the CVs.
    for i in range(obj.shape[1]):
        objx = obj[:, i]

        # Compute correlation.
        corr_obj[i] = corr_coef(cv, objx)

    # Find Spearman's correlation between CV and ranks of solutions.
    uncons_ranks = pop.extract_uncons_ranks()

    # TODO: check whether we need Spearman's or Pearson's
    corr_f = corr_coef(cv, uncons_ranks, spearman=False)

    return corr_obj, corr_f
