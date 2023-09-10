import numpy as np
from scipy.stats import kurtosis, skew
from features.feature_helpers import remove_imag_rows


def cv_distr(pop):
    """
    Y-distribution of constraints violations. For now, since only min_f is needed to substitute into Eq. (13) in Alsouly2022, we only need min_f. @Richard has left in the remaining calculations since they are easy.

    """

    # Remove any rows with imaginary values.
    cv = pop.extract_cv()
    cv = remove_imag_rows(cv)

    # Setting axis = 0 ensures computation columnwise. Output is a row vector.
    # want mean of entire matrix? Surely not. Maybe this is okay because the CV matrix is really just a column vector.
    mean_f = np.mean(cv, axis=None)
    std_f = np.std(cv, axis=0)
    min_f = np.min(cv, axis=0)
    max_f = np.max(cv, axis=0)
    kurt_f = kurtosis(cv, axis=0)
    skew_f = skew(cv, axis=0)

    return [mean_f, std_f, min_f, max_f, skew_f, kurt_f]
