import numpy as np
import copy
import warnings
import scipy.stats


def remove_imag_rows(matrix):
    """
    Remove rows which have at least one imaginary value in them
    """

    matrix = copy.deepcopy(matrix)
    rmimg = np.nonzero(np.sum(np.imag(matrix) != 0, axis=1))
    matrix = np.delete(matrix, rmimg, axis=0)
    return matrix


def corr_coef(xdata, ydata, significance_level=0.05):
    """
    Get correlation coefficient and pvalue, suppressing warnings when a constant vector is input.
    """
    with warnings.catch_warnings():
        # Suppress warnings where corr is NaN - will just set to 0 in this case.
        warnings.simplefilter("ignore", scipy.stats.ConstantInputWarning)
        result = scipy.stats.pearsonr(xdata, ydata)
        corr = result.statistic
        pvalue = result.pvalue

        # Signficance test. Null hypothesis is samples are uncorrelated.
        if pvalue > significance_level:
            corr = 0

        elif np.isnan(corr):
            # Make correlation 0 if there is no change in one vector.
            corr = 0

    return [corr, pvalue]
