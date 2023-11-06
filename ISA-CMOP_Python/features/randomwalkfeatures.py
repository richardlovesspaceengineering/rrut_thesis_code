import numpy as np
from scipy.spatial.distance import pdist
from optimisation.util.non_dominated_sorting import NonDominatedSorting
from optimisation.util.calculate_hypervolume import calculate_hypervolume_pygmo
from optimisation.model.population import Population
from optimisation.operators.sampling import RandomWalk
from features.feature_helpers import autocorr
import copy
import time


def preprocess_nans(pop_walk, pop_neighbours):
    
    # Remove any steps and corresponding neighbours if they contain infs or nans.
    pop_walk_new, num_rows_removed = pop_walk.remove_nan_inf_rows("walk", reeval_fronts=False)
    removal_idx = pop_walk.get_nan_inf_idx()
    pop_neighbours_new = [n for i, n in enumerate(pop_neighbours) if i not in removal_idx]
    
    # Remove any neighbours if they contain infs or nans.
    var = pop_walk_new.extract_var()
    
    # Make new list of populations for neighbours.
    pop_neighbours_checked = [] 
    
    # Loop over each solution in the walk.
    for i in range(var.shape[0]):
        # Extract neighbours for this point and append.
        pop_neighbourhood = copy.deepcopy(pop_neighbours_new[i])
        pop_neighbourhood, num_rows_removed = pop_neighbourhood.remove_nan_inf_rows("neig", reeval_fronts = True)
        
        # Save to list.
        pop_neighbours_checked.append(pop_neighbourhood)
        
    return pop_walk_new, pop_neighbours_new, pop_neighbours_checked

def compute_solver_crash_ratio(full_pop,trimmed_pop):
    
    obj_full = full_pop.extract_obj()
    obj_trimmed = trimmed_pop.extract_obj()
    
    scr = 1 - obj_trimmed.shape[0]/obj_full.shape[0]
    
    return scr

def compute_neighbourhood_crash_ratio(full_pop_neighbours_list,trimmed_pop_neighbours_list):
    """
    Proportion of neighbourhood solutions that crash the solver.
    """
    
    ncr_array = np.zeros(len(full_pop_neighbours_list))
    
    for i in range(len(full_pop_neighbours_list)):
        full_neig = full_pop_neighbours_list[i]
        trimmed_neig = trimmed_pop_neighbours_list[i]
        
        # Compute ratio.
        ncr_array[i] = 1-len(trimmed_neig)/len(full_neig)
    
    ncr_avg = np.mean(ncr_array)
    ncr_r1 = autocorr(ncr_array, lag=1)
    
    return ncr_avg, ncr_r1

def compute_neighbourhood_distance_features(pop_walk, pop_neighbours):
    """
    Calculate neighbourhood_features.

    pop is a matrix representing a CMOP evaluated over a random walk, each row is a solution (step) and its neighbours.

    Instances is the problem name, PF is the known Pareto Front.

    Currently only returns [dist_f_dist_x_avg_rws, dist_c_dist_x_avg_rws, bhv_avg_rws] since these are the only features required in Eq.(13) of Alsouly.
    """
    
    # TODO: ensure neighbourhood features are computed after removing any nans at each set of neighbourhoods.

    # Extract evaluated population values.
    var = pop_walk.extract_var()
    obj = pop_walk.extract_obj()
    cv = pop_walk.extract_cv()
    PF = pop_walk.extract_pf()

    # Initialise arrays.
    dist_x_array = np.zeros(var.shape[0])
    dist_f_array = np.zeros(var.shape[0])
    dist_c_array = np.zeros(var.shape[0])
    dist_f_c_array = np.zeros(var.shape[0])
    dist_f_dist_x_array = np.zeros(var.shape[0])
    dist_c_dist_x_array = np.zeros(var.shape[0])
    dist_f_c_dist_x_array = np.zeros(var.shape[0])

    # Loop over each solution in the walk.
    for i in range(var.shape[0]):
        # Extract neighbours for this point and append.
        pop_neighbourhood = copy.deepcopy(pop_neighbours[i])

        # Extract evaluated values for this neighbourhood.
        neig_var = pop_neighbourhood.extract_var()
        neig_obj = pop_neighbourhood.extract_obj()
        neig_cv = pop_neighbourhood.extract_cv()

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
        neig_obj_violation = np.concatenate((neig_obj, neig_cv), axis = 1)
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

    return dist_x_avg, dist_x_r1, dist_f_avg, dist_f_r1, dist_c_avg, dist_c_r1, dist_f_c_avg, dist_f_c_r1, dist_f_dist_x_avg, dist_f_dist_x_r1, dist_c_dist_x_avg, dist_c_dist_x_r1, dist_f_c_dist_x_avg, dist_f_c_dist_x_r1

def compute_neighbourhood_hv_features(pop_walk, pop_neighbours):
    
    PF = pop_walk.extract_pf()
        
    # Normalise objective space before computing any HVs.
    pop_walk_normalised, pop_neighbours_normalised, PF_normalised = normalise_objective_space(pop_walk, pop_neighbours, PF)
    
    # Extract evaluated population values.
    var = pop_walk_normalised.extract_var()
    obj = pop_walk_normalised.extract_obj()
    cv = pop_walk_normalised.extract_cv()

    # Initialise arrays
    hv_single_soln_array = np.zeros(var.shape[0])   
    nhv_array = np.zeros(var.shape[0])
    hvd_array = np.zeros(var.shape[0])
    bhv_array = np.zeros(var.shape[0])

    # Loop over each solution in the walk.
    nadir = np.ones(obj.shape[1])
    for i in range(var.shape[0]):
        # Extract neighbours for this point and append.
        pop_neighbourhood = pop_neighbours_normalised[i]
        
        # 
        
        # Compute HV of single solution at this step.
        hv_single_soln_array[i] = calculate_hypervolume_pygmo(np.atleast_2d(obj[i,:]), nadir)
        
        # Compute HV of neighbourhood
        nhv_array[i] = calc_nhv(pop_neighbourhood, nadir)
        
        # Compute HV difference between neighbours and that covered by the current solution
        hvd_array[i] = nhv_array[i] - hv_single_soln_array[i]
        
        # Compute HV of non-dominated neighbours.
        bhv_array[i] = calc_bhv(pop_neighbourhood, nadir)
    
    # Compute means
    hv_single_soln_avg = np.mean(hv_single_soln_array)
    nhv_avg = np.mean(nhv_array)
    hvd_avg = np.mean(hvd_array)
    bhv_avg = np.mean(bhv_array)
    
    # Compute autocorrelations
    hv_single_soln_r1 = autocorr(hv_single_soln_array, 1)
    nhv_r1 = autocorr(nhv_array, 1)
    hvd_r1 = autocorr(hvd_array, 1)
    bhv_r1 = autocorr(bhv_array, 1)
    
    return hv_single_soln_avg, hv_single_soln_r1, nhv_avg, nhv_r1, hvd_avg, hvd_r1, bhv_avg, bhv_r1
    
def compute_neighbourhood_violation_features(pop_walk, pop_neighbours):
    
    
    # Extract evaluated population values.
    var = pop_walk.extract_var()
    obj = pop_walk.extract_obj()
    cv = pop_walk.extract_cv()

    # Initialise arrays.
    cross_array = np.zeros(var.shape[0]-1)
    nncv_array = np.zeros(var.shape[0])
    ncv_array = np.zeros(var.shape[0])
    bncv_array = np.zeros(var.shape[0])
    
    # Maximum ratio of feasible boundary crossing.
    num_steps = obj.shape[0]
    num_feasible_steps = pop_walk.extract_feasible().extract_obj().shape[0]
    rfb_max = 2 / (num_steps - 1) * np.minimum(
    np.minimum(num_feasible_steps, num_steps - num_feasible_steps),
    (num_steps - 1) / 2
    )

    
    if rfb_max == 0:
        compute_rfb = False
    else:
        compute_rfb = True

    # Initialise boundary crossings index.
    cross_idx = -1
    
    # Loop over each solution in the walk.
    for i in range(var.shape[0]):
        # Extract neighbours for this point and append.
        pop_neighbourhood = pop_neighbours[i]
        neig_cv = pop_neighbourhood.extract_cv()
        
        # Average neighbourhood violation value.
        nncv_array[i] = np.mean(neig_cv)
        
        # Single solution violation value.
        ncv_array[i] = cv[i]
        
        # Average violation value of neighbourhood's non-dominated solutions.
        bncv_array[i] = np.mean(pop_neighbourhood.extract_nondominated(constrained = True).extract_cv())
        
        # Feasible boundary crossing
        if compute_rfb:
            if i > 0:
                cross_idx += 1
                if (cv[i] > 0 and cv[i-1] > 0):
                    
                    # Stayed in infeasible region.
                    cross_array[cross_idx] = 0
                else:
                    if (cv[i] <=0 and cv[i-1] <=0):
                        # Stayed in feasible region.
                        cross_array[cross_idx] = 0
                    else:
                        # Crossed between regions.
                        cross_array[cross_idx] = 1
            
        
    # Aggregate total number of feasible boundary crossings.
    if compute_rfb:
        nrfbx = np.sum(cross_array)/(num_steps-1)/rfb_max
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
        lnd_array[i] = np.atleast_2d(pop_neighbourhood.extract_nondominated()).shape[0]/len(pop_neighbourhood)
        
        # Create merged matrix of solution and neighbours.
        merged_var = np.vstack((var[i,:], pop_neighbourhood.extract_var()))
        
        # Create a new population, find rank of step relative to neighbours.
        merged_pop = Population(pop_walk[0].problem, n_individuals=merged_var.shape[0])
        merged_pop.evaluate(merged_var, eval_fronts=True)
        
        ranks = merged_pop.extract_rank()
        step_rank = ranks[0]
        
        # Compute number of fronts in this neighbourhood relative to neighbourhood size.
        nfronts_array[i] = np.unique(ranks).size/len(pop_neighbourhood)
        
        # Compute proportion of neighbours dominated by current solution.
        dominated_neighbours = ranks[ranks > step_rank]
        sup_array[i] = dominated_neighbours.size/len(pop_neighbourhood)
        
        # Compute proportion of neighbours dominating the current solution.
        dominating_neighbours = ranks[ranks < step_rank]
        inf_array[i] = dominating_neighbours.size/len(pop_neighbourhood)
        
        # Compute proportion of neighbours incomparable to the current solution.
        incomparable_neighbours = ranks[ranks == step_rank] - 1
        inc_array[i] = incomparable_neighbours.size/len(pop_neighbourhood)
        
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
    
    return sup_avg, sup_r1, inf_avg, inf_r1, inc_avg, inc_r1, lnd_avg, lnd_r1, nfronts_avg, nfronts_r1


def normalise_objective_space(pop_walk, pop_neighbours, PF, scale_offset = 1.1, region_of_interest = False):
    """
    Normalise all objectives for HV calculation.
    
    If computing population HV values, set region_of_interest to True to ensure objectives lie in the region of interest from Vodopija2023. 
    
    If computing neighbourhood HV values, set region_of_interest to False as neighbours generally do not fall in the region of interest.
    """
    
    # Merge walk objectives and neighbourhood objectives into one matrix.
    merged_obj = pop_walk.extract_obj()
    for pop_neighbourhood in pop_neighbours:
        merged_obj = np.vstack((merged_obj, pop_neighbourhood.extract_obj()))
    
    # fmin = np.minimum(np.min(obj, axis=0), np.zeros((1, PF.shape[1])))
    fmin = np.minimum(np.min(merged_obj, axis=0), np.min(PF, axis = 0))
    
    if region_of_interest:
        fmax = np.max(PF, axis=0)
    else:
        fmax = np.maximum(np.max(PF, axis = 0), np.max(merged_obj, axis = 0))
        
    # Create copies to save these new objectives to.
    pop_walk_normalised = copy.deepcopy(pop_walk)
    
    # Normalise walk objectives and update the population.
    obj_walk_normalised = apply_normalisation(pop_walk.extract_obj(), fmin, fmax, scale = scale_offset)
    pop_walk_normalised.set_obj(obj_walk_normalised)
    
    # Normalise neighbourhood objectives and update.
    pop_neighbours_normalised = []
    for pop_neighbourhood in pop_neighbours:
        obj_neighbourhood_normalised = apply_normalisation(pop_neighbourhood.extract_obj(), fmin, fmax, scale = scale_offset)
        pop_neighbourhood_normalised = copy.deepcopy(pop_neighbourhood)
        pop_neighbourhood_normalised.set_obj(obj_neighbourhood_normalised)
        pop_neighbours_normalised.append(pop_neighbourhood_normalised)
    
    # Normalise PF.
    PF_normalised = apply_normalisation(PF, fmin, fmax, scale = scale_offset)

    # To keep us in the region of interest, remove any objectives larger than the nadir.
    if region_of_interest:
        obj_normalised = obj_normalised[~np.any(obj_normalised > 1, axis=1)]

    return pop_walk_normalised, pop_neighbours_normalised, PF_normalised

def apply_normalisation(obj, fmin, fmax, scale = 1.1):
    return (obj - fmin) / ((fmax - fmin) * scale)

def calc_bhv(neighbourhood_normalised, nadir):
    """
    Calculate hypervolume of the (normalised) neighbourhood's non-dominated solutions.
    """
    bestrankobjs = neighbourhood_normalised.extract_nondominated(constrained = True).extract_obj()

    if bestrankobjs.size != 0:
        bhv = calculate_hypervolume_pygmo(bestrankobjs, nadir)
    else:
        bhv = np.nan  # TODO: might need to handle better later.
            
    return bhv

def calc_nhv(neighbourhood_normalised, nadir):
    """
    Calculate hypervolume of the (normalised) neighbourhood.
    """
    obj = neighbourhood_normalised.extract_obj()
    nhv = calculate_hypervolume_pygmo(obj, nadir)
            
    return nhv




