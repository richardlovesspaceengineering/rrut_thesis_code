import numpy as np
from scipy.stats import kurtosis, skew, iqr
from features.feature_helpers import *
from scipy.spatial.distance import pdist, cdist
from scipy.stats import iqr
from optimisation.model.population import Population


def cv_distr(pop):
    """
    Distribution of constraints violations.
    """

    # Remove any rows with imaginary values.
    cv = pop.extract_cv()

    # Setting axis = 0 ensures computation columnwise. Output is a row vector.
    mean_cv = np.mean(cv)
    std_cv = np.std(cv)
    min_cv = np.min(cv)
    max_cv = np.max(cv)
    kurt_cv = kurtosis(cv, axis=None)
    skew_cv = skew(cv, axis=None)

    return mean_cv, std_cv, min_cv, max_cv, skew_cv, kurt_cv


def uc_rk_distr(pop):
    """
    Distribution of unconstrained ranks.
    """

    # TODO: handle nans when we have a constant.
    # TODO: check why these are all constant.

    # Remove any rows with imaginary values.
    rank_uncons = pop.extract_uncons_rank()

    # Setting axis = 0 ensures computation columnwise. Output is a row vector.
    mean_uc_rk = np.mean(rank_uncons)
    std_uc_rk = np.std(rank_uncons)
    min_uc_rk = np.min(rank_uncons)
    max_uc_rk = np.max(rank_uncons)
    kurt_uc_rk = kurtosis(rank_uncons)
    skew_uc_rk = skew(rank_uncons)

    return mean_uc_rk, std_uc_rk, min_uc_rk, max_uc_rk, skew_uc_rk, kurt_uc_rk


def cv_var_mdl(pop):
    """
    Fit a linear model to decision variables-constraint violation, then take the R2 and difference between the max and min of the absolute values of the linear model coefficients.
    """
    var = pop.extract_var()
    cv = pop.extract_cv()

    cv_mdl_r2, cv_range_coeff = fit_linear_mdl(var, cv)

    return cv_mdl_r2, cv_range_coeff


def dist_corr(pop, NonDominated):
    """
    Distance correlation.

    Distance for each solution to nearest global minimum in decision space. Correlation of distance and constraints norm.

    """

    var = pop.extract_var()
    cv = pop.extract_cv()

    # Reshape nondominated variables array to be 2D if needed.
    nondominated_var = NonDominated.extract_var()
    if nondominated_var.ndim == 1:
        nondominated_var = np.reshape(nondominated_var, (1, -1))

    # For each ND decision variable, find the smallest distance to the nearest population decision variable.
    dist_matrix = cdist(nondominated_var, var, "euclidean")
    min_dist = np.min(dist_matrix, axis=0)

    # Then compute correlation coefficient between CV and these minimum distances.
    dist_c_corr = corr_coef(cv, min_dist)

    return dist_c_corr


def corr_obj(pop):
    """
    Significant correlation between objective values.

    Verel2013 provides a detailed derivation, but it seems as if we assume the objective correlation is the same metween all objectives. That is, for a correlation matrix C, C_np = rho for n != p, C_np = 1 for n = p. We want the value of rho.

    Since correlations are assumed to be equal across all objectives, we can just compute one pairwise correlation coefficient. Alsouly finds the same but computes the symmetric 2x2 correlation coefficient and pvalue matrices before extracting the upper-right value.
    """

    # Determine the symmetric correlation matrix.
    obj = pop.extract_obj()
    corr_obj, _ = compute_correlation_matrix(
        obj, correlation_type="spearman", alpha=0.05
    )

    # So that we do not consider entries on the main diagonal = corr(x,x), fill these with nans.
    # Taken from: https://stackoverflow.com/questions/29394377/minimum-of-numpy-array-ignoring-diagonal
    mask = np.ones((corr_obj.shape[0], corr_obj.shape[0]))
    mask = (mask - np.diag(np.ones(corr_obj.shape[0]))).astype(bool)

    # Extract min, max, range
    corr_obj_min = np.amin(corr_obj[mask])
    corr_obj_max = np.amax(corr_obj[mask])
    corr_obj_rnge = corr_obj_min - corr_obj_max

    return corr_obj_min, corr_obj_max, corr_obj_rnge


def compute_ps_pf_distances(pop):
    """
    Properties of the estimated Pareto-Set (PS) and Pareto-Front (PF). Includes global maximum, global mean and mean IQR of distances across the PS/PF.
    """
    obj = pop.extract_obj()
    var = pop.extract_var()

    # Remove imaginary rows. Deep copies are created here.
    obj = remove_imag_rows(obj)
    var = remove_imag_rows(var)

    # Initialize metrics.
    PS_dist_max = 0
    PS_dist_mean = 0
    PS_dist_iqr = 0
    PF_dist_max = 0
    PF_dist_mean = 0
    PF_dist_iqr = 0

    if obj.size > 1:
        # Constrained ranks.
        ranks = pop.extract_rank()

        # Distance across and between n1 and n2 rank fronts in decision space. Each argument of cdist should be arrays corresponding to the DVs on front n1 and front n2.
        var_dist_matrix = pdist(var[ranks == 1, :], "euclidean")
        obj_dist_matrix = pdist(obj[ranks == 1, :], "euclidean")

        # Compute statistics on this var_dist_matrix, presuming there is more than one point on the front.
        if var_dist_matrix.size != 0:
            PS_dist_max = np.max(var_dist_matrix)
            PS_dist_mean = np.mean(var_dist_matrix)
            PF_dist_max = np.max(obj_dist_matrix)
            PF_dist_mean = np.mean(obj_dist_matrix)

            # Find average of 25th and 75th percentile values.
            PS_dist_iqr = iqr(var_dist_matrix)
            PF_dist_iqr = iqr(obj_dist_matrix)

    return (
        PS_dist_max,
        PS_dist_mean,
        PS_dist_iqr,
        PF_dist_max,
        PF_dist_mean,
        PF_dist_iqr,
    )


def obj_skew(pop):
    """
    Checks the skewness of the objective values for this population - only univariate avg/max/min/range.
    """
    obj = pop.extract_obj()
    skew_avg = np.mean(skew(obj, axis=0))
    skew_max = np.max(skew(obj, axis=0))
    skew_min = np.min(skew(obj, axis=0))
    skew_rnge = skew_max - skew_min
    return skew_avg, skew_min, skew_max, skew_rnge


def obj_kurt(pop):
    """
    Checks the kurtosis of the objective values for this population - only univariate avg/max/min/range.
    """
    obj = pop.extract_obj()
    kurt_avg = np.mean(kurtosis(obj, axis=0))
    kurt_max = np.max(kurtosis(obj, axis=0))
    kurt_min = np.min(kurtosis(obj, axis=0))
    kurt_rnge = kurt_max - kurt_min
    return kurt_avg, kurt_min, kurt_max, kurt_rnge


def compute_ranks_cv_corr(pop):
    """
    Compute correlation between objective values and norm violation values using Spearman's rank correlation coefficient.
    """

    obj = pop.extract_obj()
    cv = pop.extract_cv()
    ranks_cons = pop.extract_rank()
    ranks_uncons = pop.extract_uncons_rank()

    # Initialise correlation between objectives.
    corr_obj_cv = np.zeros(obj.shape[0])
    corr_obj_uc_rk = np.zeros(obj.shape[0])

    # Find correlations of each objective function with the CVs.
    for i in range(obj.shape[1]):
        objx = obj[:, i]

        # Compute correlation.
        corr_obj_cv[i] = corr_coef(cv, objx, spearman=True)
        corr_obj_uc_rk[i] = corr_coef(ranks_uncons, objx, spearman=True)

    # Compute max and min.
    corr_obj_cv_min = np.min(corr_obj_cv)
    corr_obj_cv_max = np.max(corr_obj_cv)
    corr_obj_uc_rk_min = np.min(corr_obj_uc_rk)
    corr_obj_uc_rk_max = np.max(corr_obj_uc_rk)

    # Find Spearman's correlation between CV and ranks of solutions.
    corr_cv_ranks = corr_coef(cv, ranks_cons, spearman=True)

    return (
        corr_obj_cv_min,
        corr_obj_cv_max,
        corr_obj_uc_rk_min,
        corr_obj_uc_rk_max,
        corr_cv_ranks,
    )


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

    # Return summary statistics.
    piz_ob_min = np.min(piz_ob)
    piz_ob_max = np.max(piz_ob)

    return piz_ob_min, piz_ob_max, piz_f


def rk_uc_var_mdl(pop):
    """
    Fit a linear model to decision variables-front location, then take the R2 and difference between the max and min of the absolute values of the linear model coefficients.

    Very similar to the cv_mdl function, except the model is fit to different parameters.
    """

    var = pop.extract_var()
    uncons_ranks = pop.extract_uncons_rank()

    # Reshape data for compatibility. Assumes that y = mx + b where x is a matrix, y is a column vector
    uncons_ranks = uncons_ranks.reshape((-1, 1))

    # Fit linear model and compute adjusted R2 and difference between variable coefficients.
    rk_uc_mdl_r2, rk_uc_range_coeff = fit_linear_mdl(var, uncons_ranks)

    return rk_uc_mdl_r2, rk_uc_range_coeff


def compute_fsr(pop):
    feasible = pop.extract_feasible()
    return len(feasible) / len(pop)


def compute_PF_UPF_features(pop):
    nondominated_cons = pop.extract_nondominated(constrained=True)
    nondominated_uncons = pop.extract_nondominated(constrained=False)

    # Proportion of sizes of PF and UPFs.
    cpo_upo_n = len(nondominated_cons) / len(nondominated_uncons)

    # Proportion of PO solutions.
    po_n = len(nondominated_cons) / len(pop)

    # Proportion of UPF covered by PF.

    # Make a merged population and evaluate.
    # TODO: time consuming step - could just use parallel NDSort here.
    merged_dec = np.vstack(
        (nondominated_cons.extract_var(), nondominated_uncons.extract_var())
    )
    new_pop = Population(pop[0].problem, n_individuals=merged_dec.shape[0])
    new_pop.evaluate(merged_dec, eval_fronts=True)

    cons_combined_ranks = new_pop.extract_rank()[: len(nondominated_cons)]
    uncons_combined_ranks = new_pop.extract_rank()[len(nondominated_cons) :]

    # Count elements in 'a' that are less than or equal to at least one value in 'y'
    count_upf_dominates_pf = np.sum(
        np.any(
            uncons_combined_ranks <= cons_combined_ranks[:, np.newaxis],
            axis=0,
        )
    )

    cover_cpo_upo_n = count_upf_dominates_pf / len(uncons_combined_ranks)

    return po_n, cpo_upo_n, cover_cpo_upo_n
