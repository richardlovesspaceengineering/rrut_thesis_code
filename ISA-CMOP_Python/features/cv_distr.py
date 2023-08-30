import numpy as np
from scipy.stats import kurtosis, skew
from feature_helpers import remove_imag_rows


def cv_distr(objvar):
    """
    Y-distribution of constraints violations. For now, since only min_f is needed to substitute into Eq. (13) in Alsouly2022, we only need min_f. @Richard has left in the remaining calculations since they are easy.

    Might want to change name of objvar to be more appropriate to constraints violations.
    """

    # Remove any rows with imaginary values.
    objvar = remove_imag_rows(objvar)

    # Setting axis = 0 ensures computation columnwise. Output is a row vector.
    # want mean of entire matrix? Surely not. Maybe this is okay because the CV matrix is really just a column vector.
    mean_f = np.mean(objvar, axis=None)
    std_f = np.std(objvar, axis=0)
    min_f = np.min(objvar, axis=0)
    max_f = np.max(objvar, axis=0)
    kurt_f = kurtosis(objvar, axis=0)
    skew_f = skew(objvar, axis=0)

    return [mean_f, std_f, min_f, max_f, skew_f, kurt_f]
