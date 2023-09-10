from scipy.stats import skew
from features.feature_helpers import remove_imag_rows
import numpy as np


def f_skew(pop):
    """
    Checks the skewness of the population of objective values - only univariate avg/max/min/range. Removed rank skew computation since it was commented out in Alsouly's code.
    """
    obj = pop.extract_obj()
    obj = remove_imag_rows(obj)

    if obj.size > 0:
        skew_avg = np.mean(skew(obj, axis=0))
        skew_max = np.max(skew(obj, axis=0))
        skew_min = np.min(skew(obj, axis=0))
        skew_rnge = skew_max - skew_min
    return [skew_avg, skew_min, skew_max, skew_rnge]
