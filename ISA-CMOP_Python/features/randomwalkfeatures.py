import numpy as np
from scipy.spatial.distance import pdist
from optimisation.util.non_dominated_sorting import NonDominatedSorting
from optimisation.util.calculate_hypervolume import calculate_hypervolume_pygmo
from optimisation.model.population import Population


def compute_neighbourhood_features(pop_walk, pop_neighbours, PF):
    """
    Calculate neighbourhood_features.

    pop is a matrix representing a CMOP evaluated over a random walk, each row is a solution (step) and its neighbours.

    Instances is the problem name, PF is the known Pareto Front.

    Currently only returns [dist_f_dist_x_avg_rws, dist_c_dist_x_avg_rws, bhv_avg_rws] since these are the only features required in Eq.(13) of Alsouly.
    """
    
    # Extract evaluated population values. 
    var = pop_walk.extract_var()
    obj = pop_walk.extract_obj()
    cv = pop_walk.extract_cv()
    
    # Initialise arrays.
    dist_x_avg_array = np.zeros(var.shape[0])
    dist_f_avg_array = np.zeros(var.shape[0])
    dist_c_avg_array = np.zeros(var.shape[0])
    dist_f_dist_x_avg_array = np.zeros(var.shape[0])
    dist_c_dist_x_avg_array = np.zeros(var.shape[0])
    
    # Loop over each solution in the walk.
    for i in range(var.shape[0]):
        # Extract neighbours for this point and append.
        pop_neighbourhood = pop_neighbours[i]
        
        # Extract evaluated values for this neighbourhood.
        neig_var = pop_neighbourhood.extract_var()
        neig_obj = pop_neighbourhood.extract_obj()
        neig_cv = pop_neighbourhood.extract_cv()
        
        # Average distance between neighbours in variable space.
        distdec = pdist(neig_var, "euclidean")
        dist_x_avg_array[i] = np.mean(distdec)
        
        # Average distance between neighbours in objective space.
        distobj = pdist(neig_obj, "euclidean")
        dist_f_avg_array[i] = np.mean(distobj)
        
        # Average distance between neighbours in constraint-norm space.
        distcons = pdist(neig_cv, "euclidean")
        dist_c_avg_array[i] = np.mean(distcons)
        
        # Take ratios.
        dist_f_dist_x_avg_array[i] = dist_f_avg_array[i]/dist_x_avg_array[i]
        dist_c_dist_x_avg_array[i] = dist_c_avg_array[i]/dist_x_avg_array[i]
        
        
    # TODO: compute autocorrelations.
    
    # Aggregate for this walk.
    dist_x_avg = np.mean(dist_x_avg_array)
    dist_f_avg = np.mean(dist_f_avg_array)
    dist_c_avg = np.mean(dist_c_avg_array)
    dist_f_dist_x_avg = np.mean(dist_f_dist_x_avg_array)
    dist_c_dist_x_avg = np.mean(dist_c_dist_x_avg_array)

    # TODO: add bhv computation back
    # # Calculate hypervolume between the best ranked objectives and the known Pareto front.
    # ranks = pop.extract_rank()
    # bestrankobjs = obj[ranks == 1, :]

    # nadir = np.ones(bestrankobjs.shape[1])
    # bestrankobjs_normalised = normalise_for_hv(bestrankobjs, PF)

    # if bestrankobjs_normalised.size != 0:
    #     bhv = calculate_hypervolume_pygmo(bestrankobjs_normalised, nadir)
    # else:
    #     bhv = np.nan  # TODO: might need to handle better later.

    return dist_x_avg, dist_f_avg, dist_c_avg, dist_f_dist_x_avg, dist_c_dist_x_avg


def normalise_for_hv(obj, PF):
    fmin = np.minimum(np.min(obj, axis=0), np.zeros((1, PF.shape[1])))
    fmax = np.max(PF, axis=0)
    obj_normalised = (obj - fmin) / ((fmax - fmin) * 1.1)

    # Remove any objectives larger than the nadir.
    obj_normalised = obj_normalised[~np.any(obj_normalised > 1, axis=1)]

    return obj_normalised
