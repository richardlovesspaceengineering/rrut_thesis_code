import numpy as np
from scipy.stats import kurtosis, skew


def cv_distr(objvar):
    """
    Y-distribution of constraints violations. For now, since only min_f is needed to substitute into Eq. (13) in Alsouly2022, we only need min_f. @Richard has left in the remaining calculations since they are easy.
    """

    # Need to get Pythonic equivalent of this cleaning + figure out what it's doing.

    # Remove rows which have at least one imaginary value in them.
    rmimg = np.nonzero(np.sum(np.imag(objvar) != 0, axis=1))
    objvar = np.delete(objvar, rmimg, axis=0)

    # Setting axis = 0 ensures computation columnwise. Output is a row vector.
    mean_f = np.mean(objvar, axis=None)  # want mean of entire matrix? Surely not.
    std_f = np.std(objvar, axis=0)
    min_f = np.min(objvar, axis=0)
    max_f = np.max(objvar, axis=0)
    kurt_f = kurtosis(objvar, axis=0)
    skew_f = skew(objvar, axis=0)

    return [mean_f, std_f, min_f, max_f, skew_f, kurt_f]
