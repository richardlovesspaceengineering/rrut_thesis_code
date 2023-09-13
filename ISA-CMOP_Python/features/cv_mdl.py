import numpy as np
from features.feature_helpers import fit_linear_mdl


def cv_mdl(pop):
    """
    Fit a linear model to decision variables-constraint violation, then take the R2 and difference between the max and min of the absolute values of the linear model coefficients.
    """
    var = pop.extract_var()
    obj = pop.extract_obj()

    mdl_r2, range_coeff = fit_linear_mdl(var, obj)

    return [mdl_r2, range_coeff]
