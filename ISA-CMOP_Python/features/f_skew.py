from scipy.stats import skew
from feature_helpers import remove_imag_rows
import numpy as np


def f_skew(objvar):
    """
    Checks the skewness of the population of objective values - only univariate avg/max/min/range. Removed rank skew computation since it was commented out in Alsouly's code.
    """
    objvar = remove_imag_rows(objvar)

    if objvar.size > 0:
        skew_avg = np.mean(skew(objvar, axis=0))
        skew_max = max(skew(objvar, axis=0))
        skew_min = min(skew(objvar, axis=0))
        skew_rnge = skew_max - skew_min
    return [skew_avg, skew_min, skew_max, skew_rnge]
