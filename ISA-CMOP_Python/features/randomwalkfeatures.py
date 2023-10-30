import numpy as np
from scipy.spatial.distance import cdist
from optimisation.util.non_dominated_sorting import NonDominatedSorting
from optimisation.util.calculate_hypervolume import calculate_hypervolume_pygmo


def compute_neighbourhood_features(pop_walk, pop_neighbours, PF):
    """
    Calculate all the features that could be generated from a single random walk.

    pop is a matrix representing a CMOP evaluated over a random walk, each row is a solution (step) and its neighbours.

    Instances is the problem name, PF is the known Pareto Front.

    Currently only returns [dist_f_dist_x_avg_rws, dist_c_dist_x_avg_rws, bhv_avg_rws] since these are the only features required in Eq.(13) of Alsouly.
    """

    # Initialize arrays.
    # dist_x_avg = np.zeros(len(Populations))
    # dist_f_avg = np.zeros(len(Populations))
    # dist_c_avg = np.zeros(len(Populations))
    # bhv = np.zeros(len(Populations))

    # for i, pop in enumerate(Populations):

    var = pop.extract_var()
    obj = pop.extract_obj()
    cv = pop.extract_cv()

    # Average distance between neighbours in variable space.
    distdec = cdist(var, var, "euclidean")
    dist_x_avg = np.mean(distdec, axis=None)  # why global mean?

    # Average distance between neighbours in objective space.
    distobj = cdist(obj, obj, "euclidean")
    dist_f_avg = np.mean(distobj, axis=None)  # why global mean?

    # Average distance between neighbours in constraint-norm space.
    distcons = cdist(cv, cv, "euclidean")
    dist_c_avg = np.mean(distcons, axis=None)  # why global mean?

    # Calculate hypervolume between the best ranked objectives and the known Pareto front.
    ranks = pop.extract_rank()
    bestrankobjs = obj[ranks == 1, :]

    nadir = np.ones(bestrankobjs.shape[1])
    bestrankobjs_normalised = normalise_for_hv(bestrankobjs, PF)

    if bestrankobjs_normalised.size != 0:
        bhv = calculate_hypervolume_pygmo(bestrankobjs_normalised, nadir)
    else:
        bhv = np.nan  # TODO: might need to handle better later.

    # Apply elementwise division to get ratios.
    dist_f_dist_x_avg = dist_f_avg / dist_x_avg
    dist_c_dist_x_avg = dist_c_avg / dist_x_avg

    return dist_f_dist_x_avg, dist_c_dist_x_avg, bhv


def normalise_for_hv(obj, PF):
    fmin = np.minimum(np.min(obj, axis=0), np.zeros((1, PF.shape[1])))
    fmax = np.max(PF, axis=0)
    obj_normalised = (obj - fmin) / ((fmax - fmin) * 1.1)

    # Remove any objectives larger than the nadir.
    obj_normalised = obj_normalised[~np.any(obj_normalised > 1, axis=1)]

    return obj_normalised

    # # Average of features across walk.
    # dist_f_dist_x_avg_rws = np.mean(dist_f_dist_x_avg)
    # dist_c_dist_x_avg_rws = np.mean(dist_c_dist_x_avg)
    # bhv_avg_rws = np.mean(bhv)

    # return dist_f_dist_x_avg_rws, dist_c_dist_x_avg_rws, bhv_avg_rws
