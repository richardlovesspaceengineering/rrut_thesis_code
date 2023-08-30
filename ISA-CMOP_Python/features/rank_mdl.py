import numpy as np
from sklearn.linear_model import LinearRegression
from optimisation.util.non_dominated_sorting import NonDominatedSorting
from cv_mdl import cv_mdl


def rank_mdl(objvar, decvar):
    """
    Fit a linear model to decision variables-front location, then take the R2 and difference between the max and min of the absolute values of the linear model coefficients.

    Very similar to the cv_mdl function, except the model is fit to different parameters.
    """

    ranksort = NonDominatedSorting.fast_non_dominated_sort(objvar)

    # Fit linear model and compute adjusted R2 and difference between variable coefficients.
    [mdl_r2, range_coeff] = cv_mdl(ranksort, decvar)

    return [mdl_r2, range_coeff]
