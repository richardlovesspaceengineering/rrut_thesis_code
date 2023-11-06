from ProblemEvaluator import ProblemEvaluator
import numpy as np
from pymoo.problems import get_problem
import matplotlib.pyplot as plt

import copy


from features.globalfeatures import (
    cv_distr,
    cv_mdl,
    rank_mdl,
    dist_corr,
    f_corr,
    f_decdist,
    f_skew,
    fvc,
    PiIZ,
)

from features.randomwalkfeatures import (
    compute_solver_crash_ratio,
    compute_neighbourhood_crash_ratio,
    preprocess_nans,
    compute_neighbourhood_distance_features,
    compute_neighbourhood_hv_features,
    compute_neighbourhood_violation_features,
    compute_neighbourhood_dominance_features,
    normalise_objective_space
    )


def plot_transformed_objective_space(pop_walk, pop_neighbours, pop_walk_normalised, pop_neighbours_normalised, PF_normalised, scale_offset):
    
    """
    Side by side plots of objective space showing non-normalised and normalised values respectively.
    """
    
    # Compare with two side-by-side plots.
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))  # Adjust figsize as needed

    # Non-normalised
    # Walk
    walk_obj = pop_walk.extract_obj()
    ax[0].plot(
        walk_obj[:, 0], walk_obj[:, 1], "b-"
    )

    # Neighbours
    for neighbour in pop_neighbours:
        neig_obj = neighbour.extract_obj()
        ax[0].plot(
            neig_obj[:, 0], neig_obj[:, 1], "go", markersize=2
        )

    # PF
    pf = pop_walk.extract_pf()
    ax[0].plot(
        pf[:, 0], pf[:, 1], "r-"
    )

    ax[0].set_aspect('equal')

    # Set the title for ax[0]
    ax[0].set_title("Objectives")

    # Normalised
    # Walk
    walk_obj = pop_walk_normalised.extract_obj()
    ax[1].plot(
        walk_obj[:, 0], walk_obj[:, 1], "b-"
    )

    # Neighbours
    for neighbour in pop_neighbours_normalised:
        neig_obj = neighbour.extract_obj()
        ax[1].plot(
            neig_obj[:, 0], neig_obj[:, 1], "go", markersize=2
        )

    # PF
    pf = PF_normalised
    ax[1].plot(
        pf[:, 0], pf[:, 1], "r-"
    )

    # Set equal aspect ratio and limits
    ax[1].set_aspect('equal')
    ax[1].set_xlim(0, 1.1)
    ax[1].set_ylim(0, 1.1)
    ax[1].axhline(1, color='gray', linestyle='--')
    ax[1].axvline(1, color='gray', linestyle='--')
    
    # Set the title for ax[1] with the scale offset
    ax[1].set_title(f"Normalised using Scale Factor {scale_offset}")

    plt.show()
    



if __name__ == "__main__":
    
    # Flags for which feature set we want to test.
    sample_global = False
    sample_rw = True
    
    # Example problem.
    n_var = 10
    problem_name = "MW11"
    problem = get_problem(problem_name, n_var)
    instance_string = f"{problem_name}_d{n_var}"

    # Generate evaluator which can generate RW samples and LHS samples.
    evaluator = ProblemEvaluator(problem, instance_string)
    
    if sample_global:
        ### GLOBAL FEATURES.

        # Generate distributed samples and evaluate populations on these samples.
        distributed_samples = evaluator.sample_for_global_features(
            problem, 
            num_samples = 2, 
            method="lhs.scipy"
        )
        pop_global = evaluator.evaluate_populations_for_global_features(
            problem, [distributed_samples[0]]
        )[0]

    if sample_rw:
        
        test_obj_normalisation = False
        test_neighbourhood_dist_features = False
        test_neighbourhood_hv_features = False
        test_neighbourhood_violation_features = False
        test_scr = True
        
        ### RW FEATURES
        # Repeat the above but for RW samples.
        walks_neighbours_list = evaluator.sample_for_rw_features(
            problem,
            num_steps=8,
            step_size = 0.01,
            neighbourhood_size=2*n_var + 1
            )
        
        pop_walk_neighbourhood = evaluator.evaluate_populations_for_rw_features(problem, [walks_neighbours_list[0]])[0]
        
        pop_walk = pop_walk_neighbourhood[0]
        pop_neighbours = pop_walk_neighbourhood[1]
        
        if test_obj_normalisation:
            # Compute normalised objectives.
            scale_offset = 1
            pop_walk_normalised, pop_neighbours_normalised, PF_normalised = normalise_objective_space(pop_walk, pop_neighbours, pop_walk.extract_pf(), scale_offset=scale_offset, region_of_interest=False)

            plot_transformed_objective_space(pop_walk, pop_neighbours, pop_walk_normalised, pop_neighbours_normalised, PF_normalised, scale_offset)

        # Neighbourhood distance features
        if test_neighbourhood_dist_features:
            dist_x_avg, dist_x_r1, dist_f_avg, dist_f_r1, dist_c_avg, dist_c_r1, dist_f_c_avg, dist_f_c_r1, dist_f_dist_x_avg, dist_f_dist_x_r1, dist_c_dist_x_avg, dist_c_dist_x_r1, dist_f_c_dist_x_avg, dist_f_c_dist_x_r1 = compute_neighbourhood_distance_features(pop_walk, pop_neighbours)
            
        # Neighbourhood HV features.
        if test_neighbourhood_hv_features:
            hv_single_soln_avg, hv_single_soln_r1, nhv_avg, nhv_r1, hvd_avg, hvd_r1, bhv_avg, bhv_r1 = compute_neighbourhood_hv_features(pop_walk, pop_neighbours)
            
        # Neighbourhood violation features.
        if test_neighbourhood_violation_features:
            nrfbx, nncv_avg, nncv_r1, ncv_avg, ncv_r1, bncv_avg, bncv_r1 = compute_neighbourhood_violation_features(pop_walk, pop_neighbours)
            
        # Solver crash ratio.
        if test_scr:
            walk_with_issue = copy.deepcopy(walks_neighbours_list[0][0])
            
            # Use different step to the walk issue.
            neighbourhood_with_issue = copy.deepcopy(walks_neighbours_list[0][1])
            
            # Put an infeasible value in the walk (step 1) and in the neighbours (step 2) for MW11.
            walk_with_issue[0,:] = 5*np.ones((1, walk_with_issue.shape[1]))
            neighbourhood_with_issue[1][0,:] = 5*np.ones((1, walk_with_issue.shape[1]))
            
            walks_neighbours_list_new = [(walk_with_issue, neighbourhood_with_issue)]
            
            pop_walk_neighbourhood = evaluator.evaluate_populations_for_rw_features(problem, walks_neighbours_list_new)[0]
            
            pop_walk = pop_walk_neighbourhood[0]
            pop_neighbours = pop_walk_neighbourhood[1]
            
            # Preprocess nans and infinities, compute solver crash ratio and update attributes.
            pop_new, pop_neighbours_new, pop_neighbours_checked = preprocess_nans(pop_walk, pop_neighbours)
            scr = compute_solver_crash_ratio(pop_walk,pop_new)
            ncr = compute_neighbourhood_crash_ratio(pop_neighbours_new, pop_neighbours_checked)
            
        
            
            
