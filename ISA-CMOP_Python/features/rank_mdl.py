import numpy as np
from optimisation.util.non_dominated_sorting import NonDominatedSorting
from features.feature_helpers import fit_linear_mdl


def rank_mdl(pop):
    """
    Fit a linear model to decision variables-front location, then take the R2 and difference between the max and min of the absolute values of the linear model coefficients.

    Very similar to the cv_mdl function, except the model is fit to different parameters.
    """

    var = pop.extract_var()
    uncons_ranks = pop.extract_uncons_ranks()

    # Reshape data for compatibility. Assumes that y = mx + b where x is a matrix, y is a column vector
    uncons_ranks = uncons_ranks.reshape((-1, 1))

    # Fit linear model and compute adjusted R2 and difference between variable coefficients.
    mdl_r2, range_coeff = fit_linear_mdl(var, uncons_ranks)

    return mdl_r2, range_coeff
