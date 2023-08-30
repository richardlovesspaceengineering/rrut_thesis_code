from feature_helpers import remove_imag_rows, corr_coef
from optimisation.util.non_dominated_sorting import NonDominatedSorting
import numpy as np


def PiIZ(pop):
    """
    Compute proportion of solutions in the ideal zone (the lower left quadrant of the fitness-violation scatterplot) for each objective and for unconstrained fronts-violation scatterplot.
    """

    # Extracting matrices.
    objvar = pop.extract_obj()
    decvar = pop.extract_var()
    consvar = pop.extract_cons()

    # Get CV, a column vector containing the norm of the constraint violations. Assuming this can be standardised for any given problem setup.
    consvar[consvar <= 0] = 0
    cv = np.norm(consvar, axis=1)

    # Remove imaginary rows. Deep copies are created here.
    objvar = remove_imag_rows(objvar)
    decvar = remove_imag_rows(decvar)

    # Defining the ideal zone.
    minobjs = np.min(objvar, axis=0)
    maxobjs = np.max(objvar, axis=0)
    mincv = np.min(cv, axis=0)
    maxcv = np.max(cv, axis=0)
    mconsIdealPoint = mincv + (0.25 * (maxcv - mincv))
    conZone = np.nonzero(np.all(cv <= mconsIdealPoint, axis=1))

    # Find PiZ for each objXcon
    piz_ob = np.zeros(1, objvar.shape[1])

    for i in range(objvar.shape[1]):
        objIdealPoint = minobjs(i)

    return [piz_ob, piz_f]
