import numpy as np
from scipy.spatial.distance import cdist
from optimisation.util.non_dominated_sorting import NonDominatedSorting
from optimisation.util.calculate_hypervolume import calculate_hypervolume_pygmo


def randomwalkfeatures(pop, PF, Instances=None):
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
    fronts, ranks = NonDominatedSorting().do(
        obj, cons_val=cv, n_stop_if_ranked=obj.shape[0], return_rank=True
    )
    bestrankobjs = obj[ranks == 0, :]

    # Offset nadir from max objectives by 50% - can be arbitrary since we are taking the difference between hypervolumes.
    nadir = np.array([np.max(obj[:, i], axis=0) * 1.5 for i in range(obj.shape[1])])

    # Hypervolume we want is HV(PF, nadir) - HV(bestrankobjs, nadir)
    hv_nadir_pf = calculate_hypervolume_pygmo(PF, nadir)
    hv_nadir_bestrankobjs = calculate_hypervolume_pygmo(bestrankobjs, nadir)
    bhv = hv_nadir_pf - hv_nadir_bestrankobjs

    # Apply elementwise division to get ratios.
    dist_f_dist_x_avg = dist_f_avg / dist_x_avg
    dist_c_dist_x_avg = dist_c_avg / dist_x_avg

    return dist_f_dist_x_avg, dist_c_dist_x_avg, bhv

    # # Average of features across walk.
    # dist_f_dist_x_avg_rws = np.mean(dist_f_dist_x_avg)
    # dist_c_dist_x_avg_rws = np.mean(dist_c_dist_x_avg)
    # bhv_avg_rws = np.mean(bhv)

    # return dist_f_dist_x_avg_rws, dist_c_dist_x_avg_rws, bhv_avg_rws
