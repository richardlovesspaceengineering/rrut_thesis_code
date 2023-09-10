import numpy as np
from sklearn.linear_model import LinearRegression


def cv_mdl(pop):
    """
    Fit a linear model to decision variables-constraint violation, then take the R2 and difference between the max and min of the absolute values of the linear model coefficients.
    """
    var = pop.extract_var()
    obj = pop.extract_obj()

    # Fit linear model to decision variable, CV data.
    mdl = LinearRegression().fit(var, obj)

    # R2 (adjusted) has to be computed from the unadjusted value.
    num_obs = obj.shape[0]
    num_coef = obj.shape[1]
    r2_unadj = mdl.score(var, obj)
    mdl_r2 = 1 - (1 - r2_unadj) * (num_obs - 1) / (num_obs - num_coef - 1)

    # Range. Ignore the intercepts.
    # Why isn't this taking individual absolute values as the paper suggests? Maybe because all coefficients are positive?
    range_coeff = np.max(mdl.coef[1:]) - np.min(mdl.coef[1:])

    return [mdl_r2, range_coeff]
