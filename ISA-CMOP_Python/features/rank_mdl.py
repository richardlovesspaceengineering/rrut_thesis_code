import numpy as np
from sklearn.linear_model import LinearRegression
from optimisation.util.non_dominated_sorting import NonDominatedSorting
from features.cv_mdl import cv_mdl


def rank_mdl(pop):
    """
    Fit a linear model to decision variables-front location, then take the R2 and difference between the max and min of the absolute values of the linear model coefficients.

    Very similar to the cv_mdl function, except the model is fit to different parameters.
    """

    obj = pop.extract_obj()
    var = pop.extract_obj()

    # NDSort. Need to make sure this outputs a NumPy array for conditional indexing to work.
    ranksort = NonDominatedSorting().do(
        obj, cons_val=None, n_stop_if_ranked=obj.shape[0]
    )

    # Fit linear model and compute adjusted R2 and difference between variable coefficients.
    [mdl_r2, range_coeff] = cv_mdl(ranksort, var)

    return [mdl_r2, range_coeff]
