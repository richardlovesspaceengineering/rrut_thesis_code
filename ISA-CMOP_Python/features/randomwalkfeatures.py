import numpy as np
from scipy.spatial.distance import pdist
from optimisation.util.non_dominated_sorting import NonDominatedSorting
from optimisation.util.calculate_hypervolume import calculate_hypervolume_pygmo
from optimisation.model.population import Population
from optimisation.operators.sampling import RandomWalk
from features.feature_helpers import *
import copy
import time


def preprocess_nans_on_walks(pop_walk, pop_neighbours):
    # Remove any steps and corresponding neighbours if they contain infs or nans.
    pop_walk_new, num_rows_removed = pop_walk.remove_nan_inf_rows(
        "walk", re_evaluate=True
    )
    removal_idx = pop_walk.get_nan_inf_idx()
    pop_neighbours_new = [
        n for i, n in enumerate(pop_neighbours) if i not in removal_idx
    ]

    # Remove any neighbours if they contain infs or nans.
    var = pop_walk_new.extract_var()

    # Make new list of populations for neighbours.
    pop_neighbours_checked = []

    # Loop over each solution in the walk.
    for i in range(var.shape[0]):
        # Extract neighbours for this point and append.
        pop_neighbourhood = copy.deepcopy(pop_neighbours_new[i])
        pop_neighbourhood, num_rows_removed = pop_neighbourhood.remove_nan_inf_rows(
            "neig", re_evaluate=True
        )  # Don't think we need to revaluate fronts.

        # Save to list.
        pop_neighbours_checked.append(pop_neighbourhood)

    return pop_walk_new, pop_neighbours_new, pop_neighbours_checked


def compute_neighbourhood_crash_ratio(
    full_pop_neighbours_list, trimmed_pop_neighbours_list
):
    """
    Proportion of neighbourhood solutions that crash the solver.
    """

    ncr_array = np.zeros(len(full_pop_neighbours_list))

    for i in range(len(full_pop_neighbours_list)):
        full_neig = full_pop_neighbours_list[i]
        trimmed_neig = trimmed_pop_neighbours_list[i]

        # Compute ratio.
        ncr_array[i] = 1 - len(trimmed_neig) / len(full_neig)

    ncr_avg = np.mean(ncr_array)
    ncr_r1 = autocorr(ncr_array, lag=1)

    return ncr_avg, ncr_r1


def compute_neighbourhood_distance_features(
    pop_walk, pop_neighbours, normalisation_values, norm_method
):
    """
    Calculate neighbourhood_features.

    pop is a matrix representing a CMOP evaluated over a random walk, each row is a solution (step) and its neighbours.

    Instances is the problem name, PF is the known Pareto Front.

    Currently only returns [dist_f_dist_x_avg_rws, dist_c_dist_x_avg_rws, bhv_avg_rws] since these are the only features required in Eq.(13) of Alsouly.
    """

    # Extract normalisation values.
    var_lb, var_ub, obj_lb, obj_ub, cv_lb, cv_ub = extract_norm_values(
        normalisation_values, norm_method
    )

    # Initialise arrays.
    dist_x_array = np.zeros(len(pop_walk))
    dist_f_array = np.zeros(len(pop_walk))
    dist_c_array = np.zeros(len(pop_walk))
    dist_f_c_array = np.zeros(len(pop_walk))
    dist_f_dist_x_array = np.zeros(len(pop_walk))
    dist_c_dist_x_array = np.zeros(len(pop_walk))
    dist_f_c_dist_x_array = np.zeros(len(pop_walk))

    # Loop over each solution in the walk.
    for i in range(len(pop_walk)):
        # Extract neighbours for this point and append.
        pop_neighbourhood = copy.deepcopy(pop_neighbours[i])

        # Extract evaluated values for this neighbourhood and apply normalisation.
        neig_var = apply_normalisation(pop_neighbourhood.extract_var(), var_lb, var_ub)
        neig_obj = apply_normalisation(pop_neighbourhood.extract_obj(), obj_lb, obj_ub)
        neig_cv = apply_normalisation(pop_neighbourhood.extract_cv(), cv_lb, cv_ub)

        # Distance between neighbours in variable space.
        distdec = pdist(neig_var, "euclidean")
        dist_x_array[i] = np.mean(distdec)

        # Distance between neighbours in objective space.
        distobj = pdist(neig_obj, "euclidean")
        dist_f_array[i] = np.mean(distobj)

        # Distance between neighbours in constraint-norm space.
        distcons = pdist(neig_cv, "euclidean")
        dist_c_array[i] = np.mean(distcons)

        # Distance between neighbours in objective-violation space.

        # Construct objective-violation space by horizontally joining objectives and CV so that each solution forms a ((objectives), cv) tuple.
        neig_obj_violation = np.concatenate((neig_obj, neig_cv), axis=1)
        dist_obj_violation = pdist(neig_obj_violation)
        dist_f_c_array[i] = np.mean(dist_obj_violation)

        # Take ratios.
        dist_f_dist_x_array[i] = dist_f_array[i] / dist_x_array[i]
        dist_c_dist_x_array[i] = dist_c_array[i] / dist_x_array[i]
        dist_f_c_dist_x_array[i] = dist_f_c_array[i] / dist_x_array[i]

    # Aggregate for this walk.
    dist_x_avg = np.mean(dist_x_array)
    dist_f_avg = np.mean(dist_f_array)
    dist_c_avg = np.mean(dist_c_array)
    dist_f_c_avg = np.mean(dist_f_c_array)
    dist_f_dist_x_avg = np.mean(dist_f_dist_x_array)
    dist_c_dist_x_avg = np.mean(dist_c_dist_x_array)
    dist_f_c_dist_x_avg = np.mean(dist_f_c_dist_x_array)

    # Compute autocorrelations
    dist_x_r1 = autocorr(dist_x_array, lag=1)
    dist_f_r1 = autocorr(dist_f_array, lag=1)
    dist_c_r1 = autocorr(dist_c_array, lag=1)
    dist_f_c_r1 = autocorr(dist_f_c_array, lag=1)
    dist_f_dist_x_r1 = autocorr(dist_f_dist_x_array, lag=1)
    dist_c_dist_x_r1 = autocorr(dist_c_dist_x_array, lag=1)
    dist_f_c_dist_x_r1 = autocorr(dist_f_c_dist_x_array, lag=1)

    return (
        dist_x_avg,
        dist_x_r1,
        dist_f_avg,
        dist_f_r1,
        dist_c_avg,
        dist_c_r1,
        dist_f_c_avg,
        dist_f_c_r1,
        dist_f_dist_x_avg,
        dist_f_dist_x_r1,
        dist_c_dist_x_avg,
        dist_c_dist_x_r1,
        dist_f_c_dist_x_avg,
        dist_f_c_dist_x_r1,
    )


def trim_obj_using_nadir(obj, nadir):
    # Create a boolean mask
    mask = np.all(obj <= nadir, axis=1)

    # Use the mask to select the rows from obj
    trimmed_obj = obj[mask]

    return trimmed_obj, mask


def trim_neig_using_mask(lst, mask):
    # Use list comprehension and zip to create a new list with elements
    # corresponding to False in the mask
    filtered_list = [value for value, keep in zip(lst, mask) if keep]

    return filtered_list


def compute_neighbourhood_hv_features(
    pop_walk, pop_neighbours, normalisation_values, norm_method
):
    # Extract normalisation values.
    var_lb, var_ub, obj_lb, obj_ub, cv_lb, cv_ub = extract_norm_values(
        normalisation_values, norm_method
    )

    # Define the nadir
    nadir = 1.1 * np.ones(obj_lb.size)

    # Extract evaluated population values, normalise and trim any points larger than the nadir.
    obj, mask = trim_obj_using_nadir(
        apply_normalisation(pop_walk.extract_obj(), obj_lb, obj_ub), nadir
    )

    num_rows_trimmed = len(pop_walk) - np.count_nonzero(mask)
    if num_rows_trimmed > 0:
        print(
            "Had to remove {} rows that were further than the nadir from the origin.".format(
                num_rows_trimmed
            )
        )

    # Also remove corresponding neighbours for steps on the walk larger than the nadir.
    pop_neighbours = trim_neig_using_mask(pop_neighbours, mask)

    # Initialise arrays
    hv_single_soln_array = np.zeros(obj.shape[0])
    nhv_array = np.zeros(obj.shape[0])
    hvd_array = np.zeros(obj.shape[0])
    bhv_array = np.zeros(obj.shape[0])

    # Set nonsense value if obj.size == 0
    if obj.size == 0:
        hv_single_soln_array.fill(np.nan)
        nhv_array.fill(np.nan)
        hvd_array.fill(np.nan)
        bhv_array.fill(np.nan)
        print(
            "There are no individuals closer to the origin than the nadir. Setting all step HV metrics for this sample to NaN."
        )
    else:
        # Loop over each solution in the walk.
        for i in range(obj.shape[0]):
            # Extract neighbours for this point and normalise.
            pop_neighbourhood = pop_neighbours[i]
            neig_obj, _ = trim_obj_using_nadir(
                apply_normalisation(pop_neighbourhood.extract_obj(), obj_lb, obj_ub),
                nadir,
            )
            if neig_obj.size == 0:
                hv_single_soln_array[i] = np.nan
                nhv_array[i] = np.nan
                hvd_array[i] = np.nan
                bhv_array[i] = np.nan
                print(
                    "There are no neighbours closer to the origin than the nadir for step {} of {}. Setting all neighbourhood HV metrics for this step to NaN.".format(
                        i + 1, obj.shape[0]
                    )
                )
            else:
                # Compute HV of single solution at this step.
                hv_single_soln_array[i] = calculate_hypervolume_pygmo(
                    np.atleast_2d(obj[i, :]), nadir
                )

                # Compute HV of neighbourhood
                nhv_array[i] = calculate_hypervolume_pygmo(neig_obj, nadir)

                # Compute HV difference between neighbours and that covered by the current solution
                hvd_array[i] = nhv_array[i] - hv_single_soln_array[i]

                # Compute HV of non-dominated neighbours (trimmed).
                bestrankobjs, _ = trim_obj_using_nadir(
                    apply_normalisation(
                        pop_neighbourhood.extract_nondominated().extract_obj(),
                        obj_lb,
                        obj_ub,
                    ),
                    nadir,
                )
                try:
                    bhv_array[i] = calculate_hypervolume_pygmo(bestrankobjs, nadir)
                except:
                    # In case the NDFront is further from the origin than the nadir.
                    bhv_array[i] = np.nan
                    print(
                        "There are no non-dominated neighbours closer to the origin than the nadir for step {} of {}. Setting HV metric bhv_avg to NaN.".format(
                            i + 1, obj.shape[0]
                        )
                    )

    # Compute means (allowing for nans if need be)
    hv_single_soln_avg = np.nanmean(hv_single_soln_array)
    nhv_avg = np.nanmean(nhv_array)
    hvd_avg = np.nanmean(hvd_array)
    bhv_avg = np.nanmean(bhv_array)

    # Compute autocorrelations
    hv_single_soln_r1 = autocorr(hv_single_soln_array, 1)
    nhv_r1 = autocorr(nhv_array, 1)
    hvd_r1 = autocorr(hvd_array, 1)
    bhv_r1 = autocorr(bhv_array, 1)

    return (
        hv_single_soln_avg,
        hv_single_soln_r1,
        nhv_avg,
        nhv_r1,
        hvd_avg,
        hvd_r1,
        bhv_avg,
        bhv_r1,
    )


def compute_neighbourhood_violation_features(
    pop_walk, pop_neighbours, normalisation_values, norm_method
):
    # Extract normalisation values.
    var_lb, var_ub, obj_lb, obj_ub, cv_lb, cv_ub = extract_norm_values(
        normalisation_values, norm_method
    )

    # Extract evaluated population values.
    var = pop_walk.extract_var()
    obj = pop_walk.extract_obj()

    # Should only need to normalsie CV here.
    cv = apply_normalisation(pop_walk.extract_cv(), cv_lb, cv_ub)

    # Initialise arrays.
    cross_array = np.zeros(var.shape[0] - 1)
    nncv_array = np.zeros(var.shape[0])
    ncv_array = np.zeros(var.shape[0])
    bncv_array = np.zeros(var.shape[0])

    # Maximum ratio of feasible boundary crossing.
    num_steps = obj.shape[0]
    num_feasible_steps = pop_walk.extract_feasible().extract_obj().shape[0]
    rfb_max = (
        2
        / (num_steps - 1)
        * np.minimum(
            np.minimum(num_feasible_steps, num_steps - num_feasible_steps),
            (num_steps - 1) / 2,
        )
    )

    if rfb_max == 0:
        compute_rfb = False
    else:
        compute_rfb = True

    # Initialise boundary crossings index.
    cross_idx = -1

    # Loop over each solution in the walk.
    for i in range(var.shape[0]):
        # Extract neighbours for this point and normalise.
        pop_neighbourhood = pop_neighbours[i]
        neig_cv = apply_normalisation(pop_neighbourhood.extract_cv(), cv_lb, cv_ub)

        # Average neighbourhood violation value.
        nncv_array[i] = np.mean(neig_cv)

        # Single solution violation value.
        ncv_array[i] = cv[i]

        # Average violation value of neighbourhood's non-dominated solutions.
        bncv_array[i] = np.mean(
            apply_normalisation(
                pop_neighbourhood.extract_nondominated(constrained=True).extract_cv(),
                cv_lb,
                cv_ub,
            )
        )

        # Feasible boundary crossing
        if compute_rfb:
            if i > 0:
                cross_idx += 1
                if cv[i] > 0 and cv[i - 1] > 0:
                    # Stayed in infeasible region.
                    cross_array[cross_idx] = 0
                else:
                    if cv[i] <= 0 and cv[i - 1] <= 0:
                        # Stayed in feasible region.
                        cross_array[cross_idx] = 0
                    else:
                        # Crossed between regions.
                        cross_array[cross_idx] = 1

    # Aggregate total number of feasible boundary crossings.
    if compute_rfb:
        nrfbx = np.sum(cross_array) / (num_steps - 1) / rfb_max
    else:
        nrfbx = 0

    # Calculate means
    nncv_avg = np.mean(nncv_array)
    ncv_avg = np.mean(ncv_array)
    bncv_avg = np.mean(bncv_array)

    # Calculate autocorrelations
    nncv_r1 = autocorr(nncv_array, lag=1)
    ncv_r1 = autocorr(ncv_array, lag=1)
    bncv_r1 = autocorr(bncv_array, lag=1)

    return nrfbx, nncv_avg, nncv_r1, ncv_avg, ncv_r1, bncv_avg, bncv_r1


def compute_neighbourhood_dominance_features(pop_walk, pop_neighbours):
    # Extract evaluated population values.
    var = pop_walk.extract_var()
    obj = pop_walk.extract_obj()
    cv = pop_walk.extract_cv()
    PF = pop_walk.extract_pf()

    # Initialise arrays.
    sup_array = np.zeros(var.shape[0])
    inf_array = np.zeros(var.shape[0])
    inc_array = np.zeros(var.shape[0])
    lnd_array = np.zeros(var.shape[0])
    nfronts_array = np.zeros(var.shape[0])

    for i in range(var.shape[0]):
        # Extract neighbours for this point and append.
        pop_neighbourhood = pop_neighbours[i]

        # Compute proportion of locally non-dominated solutions.
        lnd_array[i] = np.atleast_2d(pop_neighbourhood.extract_nondominated()).shape[
            0
        ] / len(pop_neighbourhood)

        # Create merged matrix of solution and neighbours.
        merged_var = np.vstack((var[i, :], pop_neighbourhood.extract_var()))

        # Create a new population, find rank of step relative to neighbours.
        merged_pop = Population(pop_walk[0].problem, n_individuals=merged_var.shape[0])
        merged_pop.evaluate(merged_var, eval_fronts=True)

        ranks = merged_pop.extract_rank()
        step_rank = ranks[0]

        # Compute number of fronts in this neighbourhood relative to neighbourhood size.
        nfronts_array[i] = np.unique(ranks).size / len(pop_neighbourhood)

        # Compute proportion of neighbours dominated by current solution.
        dominated_neighbours = ranks[ranks > step_rank]
        sup_array[i] = dominated_neighbours.size / len(pop_neighbourhood)

        # Compute proportion of neighbours dominating the current solution.
        dominating_neighbours = ranks[ranks < step_rank]
        inf_array[i] = dominating_neighbours.size / len(pop_neighbourhood)

        # Compute proportion of neighbours incomparable to the current solution.
        incomparable_neighbours = ranks[ranks == step_rank] - 1
        inc_array[i] = incomparable_neighbours.size / len(pop_neighbourhood)

    # Calculate means
    sup_avg = np.mean(sup_array)
    inf_avg = np.mean(inf_array)
    inc_avg = np.mean(inc_array)
    lnd_avg = np.mean(lnd_array)
    nfronts_avg = np.mean(nfronts_array)

    # Calculate autocorrelations
    sup_r1 = autocorr(sup_array, lag=1)
    inf_r1 = autocorr(inf_array, lag=1)
    inc_r1 = autocorr(inc_array, lag=1)
    lnd_r1 = autocorr(lnd_array, lag=1)
    nfronts_r1 = autocorr(nfronts_array, lag=1)

    return (
        sup_avg,
        sup_r1,
        inf_avg,
        inf_r1,
        inc_avg,
        inc_r1,
        lnd_avg,
        lnd_r1,
        nfronts_avg,
        nfronts_r1,
    )
