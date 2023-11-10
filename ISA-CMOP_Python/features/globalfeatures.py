import numpy as np
from scipy.stats import kurtosis, skew, iqr
from features.feature_helpers import remove_imag_rows, corr_coef, fit_linear_mdl
from scipy.spatial.distance import cdist
from scipy.stats import iqr


def cv_distr(pop):
    """
    Distribution of constraints violations.
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

    return mean_f, std_f, min_f, max_f, skew_f, kurt_f


def cv_mdl(pop):
    """
    Fit a linear model to decision variables-constraint violation, then take the R2 and difference between the max and min of the absolute values of the linear model coefficients.
    """
    var = pop.extract_var()
    cv = pop.extract_cv()
    obj = pop.extract_obj()

    mdl_r2, range_coeff = fit_linear_mdl(var, cv)

    return mdl_r2, range_coeff


def dist_corr(pop, NonDominated):
    """
    Distance correlation.

    Distance for each solution to nearest global solution in decision space. Correlation of distance and constraints norm.

    """

    obj = pop.extract_obj()
    var = pop.extract_var()
    cons = pop.extract_cons()  # constraint matrix.
    cv = pop.extract_cv()

    # Remove imaginary rows. Deep copies are created here.
    obj = remove_imag_rows(obj)
    var = remove_imag_rows(var)
    cons = remove_imag_rows(cons)

    # Reshape nondominated variables array to be 2D if needed.
    nondominated_var = NonDominated.extract_var()
    if nondominated_var.ndim == 1:
        nondominated_var = np.reshape(nondominated_var, (1, -1))

    # For each ND decision variable, find the smallest distance to the nearest population decision variable.
    dist_matrix = cdist(nondominated_var, var, "euclidean")
    min_dist = np.min(dist_matrix, axis=0)

    # Then compute correlation coefficient between CV and . Assumed that all values in the correlation matrix are the same, meaning we only need one scalar. See f_corr for a similar implementation.
    dist_c_corr = corr_coef(cv, min_dist)
    return dist_c_corr


def f_corr(pop):
    """
    Significant correlation between objective values.

    Verel2013 provides a detailed derivation, but it seems as if we assume the objective correlation is the same metween all objectives. That is, for a correlation matrix C, C_np = rho for n != p, C_np = 1 for n = p. We want the value of rho.

    Since correlations are assumed to be equal across all objectives, we can just compute one pairwise correlation coefficient. Alsouly finds the same but computes the symmetric 2x2 correlation coefficient and pvalue matrices before extracting the upper-right value.
    """
    obj = pop.extract_obj()
    corr_obj = corr_coef(obj[:, 0], obj[:, 1])

    return corr_obj


def f_decdist(pop, n1, n2):
    """
    Properties of the Pareto-Set (PS). Includes global maximum, global mean and mean IQR of distances across the PS.

    For our application, we set n2 = n1 = 1 i.e we want the distances between all of the decision variables on the PS (corresponding to the PF in decision space)
    """
    obj = pop.extract_obj()
    var = pop.extract_var()

    # Remove imaginary rows. Deep copies are created here.
    obj = remove_imag_rows(obj)
    var = remove_imag_rows(var)

    # Initialize metrics.
    PSdecdist_max = 0
    PSdecdist_mean = 0
    PSdecdist_iqr_mean = 0

    if obj.size > 1:
        # Constrained ranks.
        ranks = pop.extract_rank()

        # Distance across and between n1 and n2 rank fronts in decision space. Each argument of cdist should be arrays corresponding to the DVs on front n1 and front n2.
        dist_matrix = cdist(var[ranks == n1, :], var[ranks == n2, :], "euclidean")

        # Compute statistics on this dist_matrix.
        PSdecdist_max = np.max(np.max(dist_matrix, axis=0))
        PSdecdist_mean = np.mean(dist_matrix, axis=None)

        # Take IQR of each column, then take mean of IQRs to get a scalar.
        PSdecdist_iqr_mean = np.mean(iqr(dist_matrix, axis=0), axis=None)

    return PSdecdist_max, PSdecdist_mean, PSdecdist_iqr_mean


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


def fvc(pop):
    """
    Compute correlation between objective values and norm violation values using Spearman's rank correlation coefficient.
    """

    obj = pop.extract_obj()
    cons = pop.extract_cons()
    cv = pop.extract_cv()

    # Remove imaginary rows. Deep copies are created here.
    obj = remove_imag_rows(obj)
    cons = remove_imag_rows(cons)  # may be able to remove this.

    # Initialise correlation between objectives.
    corr_obj = np.zeros(obj.shape[0])

    # Find correlations of each objective function with the CVs.
    for i in range(obj.shape[1]):
        objx = obj[:, i]

        # Compute correlation.
        corr_obj[i] = corr_coef(cv, objx)

    # Find Spearman's correlation between CV and ranks of solutions.
    uncons_ranks = pop.extract_uncons_rank()

    # TODO: check whether we need Spearman's or Pearson's
    corr_f = corr_coef(cv, uncons_ranks, spearman=False)

    return corr_obj, corr_f


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
    conZone = np.all(cv <= mconsIdealPoint, axis=1)

    # Find PiZ for each objXcon
    piz_ob = np.zeros(obj.shape[1])
    for i in range(obj.shape[1]):
        objIdealPoint = minobjs[i] + (0.25 * (maxobjs[i] - minobjs[i]))
        objx = obj[:, i]
        iz = np.asarray(objx[conZone] <= objIdealPoint)

        # TODO: check why denominator is num_objs x num_samples
        piz_ob[i] = np.count_nonzero(iz) / pop.extract_obj().size

    # Find PiZ for each frontsXcon

    # May need to transpose.
    uncons_ranks = pop.extract_uncons_rank()

    # Axes may need to change depending on the structure of ranks. Right now we are taking the min of a column vector.
    minrank = np.min(uncons_ranks)
    maxrank = np.max(uncons_ranks)
    rankIdealPoint = minrank + (0.25 * (maxrank - minrank))
    iz = np.asarray(uncons_ranks[conZone] <= rankIdealPoint)
    piz_f = np.count_nonzero(iz) / pop.extract_obj().size

    return piz_ob, piz_f


def rank_mdl(pop):
    """
    Fit a linear model to decision variables-front location, then take the R2 and difference between the max and min of the absolute values of the linear model coefficients.

    Very similar to the cv_mdl function, except the model is fit to different parameters.
    """

    var = pop.extract_var()
    uncons_ranks = pop.extract_uncons_rank()

    # Reshape data for compatibility. Assumes that y = mx + b where x is a matrix, y is a column vector
    uncons_ranks = uncons_ranks.reshape((-1, 1))

    # Fit linear model and compute adjusted R2 and difference between variable coefficients.
    mdl_r2, range_coeff = fit_linear_mdl(var, uncons_ranks)

    return mdl_r2, range_coeff
