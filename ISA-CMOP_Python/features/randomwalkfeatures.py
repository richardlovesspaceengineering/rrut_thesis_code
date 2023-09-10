import numpy as np
from scipy.spatial.distance import cdist
from optimisation.util.non_dominated_sorting import NonDominatedSorting
from optimisation.util.calculate_hypervolume import calculate_hypervolume_pygmo


def randomwalkfeatures(Populations, Instances, PF):
    """
    Calculate all the features that could be generated from random walk samples.

    Populations is a matrix that represents a walk, each row is a solution (step) and its neighbours. Instances is the problem name, PF is the known Pareto Front.

    Currently only returns [dist_f_dist_x_avg_rws, dist_c_dist_x_avg_rws, bhv_avg_rws] since these are the only features required in Eq.(13) of Alsouly.
    """

    # Initialize arrays.
    dist_x_avg = np.zeros(1, len(Populations))
    dist_f_avg = np.zeros(1, len(Populations))
    dist_c_avg = np.zeros(1, len(Populations))
    bhv = np.zeros(1, len(Populations))

    for i, pop in enumerate(Populations):
        var = pop.extract_var()
        obj = pop.extract_obj()
        cons = pop.extract_cons()
        cv = pop.extract_cv()

        # Average distance between neighbours in variable space.
        distdec = cdist(var, var, "euclidean")
        dist_x_avg[i] = np.mean(distdec, axis=None)  # why global mean?

        # Average distance between neighbours in objective space.
        distobj = cdist(obj, obj, "euclidean")
        dist_f_avg[i] = np.mean(distobj, axis=None)  # why global mean?

        # Average distance between neighbours in constraint-norm space.
        distcons = cdist(cv, cv, "euclidean")
        dist_c_avg[i] = np.mean(distcons, axis=None)  # why global mean?

        # Calculate hypervolume between the best ranked objectives and the known Pareto front.
        ranksort = NonDominatedSorting().do(
            obj, cons_val=cv, n_stop_if_ranked=obj.shape[0]
        )
        bestrankobjs = obj[ranksort == 0, :]
        nadir = np.array([np.max(obj[:, i]) for i in range(obj.shape[1])])

        # Hypervolume we want is HV(PF, nadir) - HV(bestrankobjs, nadir)
        hv_nadir_pf = calculate_hypervolume_pygmo(PF, nadir)
        hv_nadir_bestrankobjs = calculate_hypervolume_pygmo(bestrankobjs, nadir)
        bhv[i] = hv_nadir_pf - hv_nadir_bestrankobjs

    # Apply elementwise division to get ratios.
    dist_f_dist_x_avg = dist_f_avg / dist_x_avg
    dist_c_dist_x_avg = dist_c_avg / dist_x_avg

    # Average of features across walk.
    dist_f_dist_x_avg_rws = np.mean(dist_f_dist_x_avg)
    dist_c_dist_x_avg_rws = np.mean(dist_c_dist_x_avg)
    bhv_avg_rws = np.mean(bhv)

    return [dist_f_dist_x_avg_rws, dist_c_dist_x_avg_rws, bhv_avg_rws]
