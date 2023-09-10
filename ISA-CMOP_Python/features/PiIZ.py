from features.feature_helpers import remove_imag_rows, corr_coef
from optimisation.util.non_dominated_sorting import NonDominatedSorting
import numpy as np


def PiIZ(pop):
    """
    Compute proportion of solutions in the ideal zone (the lower left quadrant of the fitness-violation scatterplot) for each objective and for unconstrained fronts-violation scatterplot.

    May need to play around with the axis definitions while debugging.
    """

    # Extracting matrices.
    obj = pop.extract_obj()
    var = pop.extract_var()
    cons = pop.extract_cons()
    cv = pop.extract_cv()

    # Remove imaginary rows. Deep copies are created here.
    obj = remove_imag_rows(obj)
    var = remove_imag_rows(var)

    # Defining the ideal zone.
    minobjs = np.min(obj, axis=0)
    maxobjs = np.max(obj, axis=0)
    mincv = np.min(cv, axis=0)
    maxcv = np.max(cv, axis=0)
    mconsIdealPoint = mincv + (0.25 * (maxcv - mincv))
    conZone = np.nonzero(np.all(cv <= mconsIdealPoint, axis=1))

    # Find PiZ for each objXcon
    piz_ob = np.zeros(obj.shape[1])
    for i in range(obj.shape[1]):
        objIdealPoint = minobjs[i] + (0.25 * (maxobjs[i] - minobjs[i]))
        objx = obj[:, i]
        iz = np.nonzero(np.all(objx[conZone] <= objIdealPoint, axis=1))
        piz_ob[i] = iz.size / pop.extract_obj().size

    # Find PiZ for each frontsXcon

    # May need to transpose.
    ranksort = np.transpose(
        NonDominatedSorting().do(obj, cons_val=None, n_stop_if_ranked=obj.shape[0])
    )

    # Axes may need to change depending on the structure of ranksort. Right now we are taking the min of a column vector.
    minrank = np.min(ranksort, axis=0)
    maxrank = np.max(ranksort, axis=0)
    rankIdealPoint = minrank + (0.25 * (maxrank - minrank))
    iz = np.nonzero(np.all(ranksort[conZone] <= rankIdealPoint, axis=1))
    piz_f = iz.size / pop.extract_obj().size

    return [piz_ob, piz_f]
