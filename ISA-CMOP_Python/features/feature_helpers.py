import numpy as np
import copy
import warnings
import scipy.stats
from sklearn.linear_model import LinearRegression


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

    # Range of magnitudes. Ignore the intercepts.
    range_coeff = np.abs(np.max(mdl.coef_)) - np.abs(np.min(mdl.coef_))

    return mdl_r2, range_coeff


def generate_bounds_from_problem(problem_instance):
    # Bounds of the decision variables.
    x_lower = problem_instance.xl
    x_upper = problem_instance.xu
    bounds = np.vstack((x_lower, x_upper))
    return bounds


def corr_coef(xdata, ydata, spearman=True, significance_level=0.05):
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


def autocorr(data, lag, spearman=True, significance_level=0.05):
    """
    Compute autocorrelation of data with applied lag.
    """
    return corr_coef(data[:-lag], data[lag:], spearman, significance_level)


import numpy as np
from scipy.stats import pearsonr, spearmanr


def compute_correlation_matrix(matrix, correlation_type, alpha=0.05):
    """
    Compute the correlation matrix of a given square matrix and trim based on significance.

    Note that computed p-values are only valid for > 500 observations - otherwise a parametric test for significance should be done.

    Parameters:
    - matrix: 2D numpy array, square matrix
    - correlation_type: str, either 'pearson' or 'spearman'
    - alpha: float, significance level

    Returns:
    - correlation_matrix: 2D numpy array, correlation matrix with trimmed values
    - significance_matrix: 2D numpy array, matrix indicating significance (True/False)
    """

    if correlation_type not in ["pearson", "spearman"]:
        raise ValueError("Invalid correlation type. Use 'pearson' or 'spearman'.")
    if correlation_type == "pearson":
        correlation_matrix, p_values = pearsonr(matrix.T)
    elif correlation_type == "spearman":
        correlation_matrix, p_values = spearmanr(matrix, axis=0)

    if correlation_matrix.ndim == 0:  # If the result is a scalar (2x2 matrix case)
        correlation_matrix = np.array(
            [[1, correlation_matrix], [correlation_matrix, 1]]
        )
        p_values = np.array([[1, p_values], [p_values, 1]])

    significance_matrix = p_values > alpha

    # Trim values based on significance
    correlation_matrix[significance_matrix] = 0

    return correlation_matrix, p_values
