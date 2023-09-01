import numpy as np
from scipy.spatial.distance import cdist
from optimisation.util.non_dominated_sorting import NonDominatedSorting
from optimisation.util.calculate_hypervolume import calculate_hypervolume


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
        decvar = pop.extract_var()
        objvar = pop.extract_obj()
        consvar = pop.extract_cons()

        # Get CV, a column vector containing the norm of the constraint violations. Assuming this can be standardised for any given problem setup.
        consvar[consvar <= 0] = 0
        cv = np.norm(consvar, axis=1)

        # Average distance between neighbours in variable space.
        distdec = cdist(decvar, decvar, "euclidean")
        dist_x_avg[i] = np.mean(distdec, axis=None)  # why global mean?

        # Average distance between neighbours in objective space.
        distobj = cdist(objvar, objvar, "euclidean")
        dist_f_avg[i] = np.mean(distobj, axis=None)  # why global mean?

        # Average distance between neighbours in constraint-norm space.
        distcons = cdist(cv, cv, "euclidean")
        dist_c_avg[i] = np.mean(distcons, axis=None)  # why global mean?

        #
        ranksort = NonDominatedSorting.fast_non_dominated_sort(objvar, consvar)
        bestrankobjs = objvar[ranksort == 0, :]
        bhv[i] = calculate_hypervolume(bestrankobjs)

    # Apply elementwise division to get ratios.
    dist_f_dist_x_avg = dist_f_avg / dist_x_avg
    dist_c_dist_x_avg = dist_c_avg / dist_x_avg

    # Average of features across walk.
    dist_f_dist_x_avg_rws = np.mean(dist_f_dist_x_avg)
    dist_c_dist_x_avg_rws = np.mean(dist_c_dist_x_avg)
    bhv_avg_rws = np.mean(bhv)

    return [dist_f_dist_x_avg_rws, dist_c_dist_x_avg_rws, bhv_avg_rws]
