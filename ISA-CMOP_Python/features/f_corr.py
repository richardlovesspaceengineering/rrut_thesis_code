import scipy
import numpy as np
import warnings


def f_corr(objvar, significance_level=0.05):
    """
    Significant correlation between objective values.

    Verel2013 provides a detailed derivation, but it seems as if we assume the objective correlation is the same metween all objectives. That is, for a correlation matrix C, C_np = rho for n != p, C_np = 1 for n = p. We want the value of rho.

    Since correlations are assumed to be equal across all objectives, we can just compute one pairwise correlation coefficient. Alsouly finds the same but computes the symmetric 2x2 correlation coefficient and pvalue matrices before extracting the upper-right value.
    """

    with warnings.catch_warnings():
        # Suppress warnings where corr is NaN - will just set to 0 in this case.
        warnings.simplefilter("ignore", scipy.stats.ConstantInputWarning)
        result = scipy.stats.pearsonr(objvar[:, 0], objvar[:, 1])

    corr_obj = result.statistic
    pvalue = result.pvalue

    # Signficance test. Alsouly does p > alpha for some reason.
    if pvalue < significance_level:
        corr_obj = 0
    elif np.isnan(corr_obj):
        corr_obj = 0

    # Make correlation 0 if there is no change in one vector.

    return corr_obj
