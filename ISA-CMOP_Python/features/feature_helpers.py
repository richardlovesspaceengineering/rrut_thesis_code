import numpy as np
import copy
import warnings
import scipy.stats
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def remove_imag_rows(matrix):
    """
    Remove rows which have at least one imaginary value in them
    """

    new_matrix = copy.deepcopy(matrix)
    rmimg = np.sum(np.imag(new_matrix) != 0, axis=1)
    new_matrix = new_matrix[np.logical_not(rmimg)]
    return new_matrix


def fit_linear_mdl(xdata, ydata):
    # Fit linear model to xdata and ydata.
    mdl = LinearRegression().fit(xdata, ydata)

    # R2 (adjusted) has to be computed from the unadjusted value.
    num_obs = ydata.shape[0]
    num_coef = xdata.shape[1]
    r2_unadj = mdl.score(xdata, ydata)
    mdl_r2 = 1 - (1 - r2_unadj) * (num_obs - 1) / (num_obs - num_coef - 1)

    # Range. Ignore the intercepts.
    # Why isn't this taking individual absolute values as the paper suggests? Maybe because all coefficients are positive?
    # range_coeff = np.abs(np.max(mdl.coef_)) - np.abs(np.min(mdl.coef_))
    range_coeff = np.max(mdl.coef_) - np.min(mdl.coef_)

    return mdl_r2, range_coeff


def corr_coef(xdata, ydata, spearman=False, significance_level=0.05):
    """
    Get correlation coefficient and pvalue, suppressing warnings when a constant vector is input.
    """
    with warnings.catch_warnings():
        # Suppress warnings where corr is NaN - will just set to 0 in this case.
        warnings.simplefilter("ignore", scipy.stats.ConstantInputWarning)

        # Method for computing correlation.
        if spearman:
            method = scipy.stats.spearmanr
        else:
            method = scipy.stats.pearsonr

        # Ensure shapes are compatible. Should be okay to squeeze because xdata and ydata will always be vectors.
        result = method(np.squeeze(xdata), np.squeeze(ydata))
        corr = result.statistic
        pvalue = result.pvalue

        # Signficance test. Null hypothesis is samples are uncorrelated.
        if pvalue > significance_level:
            corr = 0

        elif np.isnan(corr):
            # Make correlation 0 if there is no change in one vector.
            corr = 0

    return corr
