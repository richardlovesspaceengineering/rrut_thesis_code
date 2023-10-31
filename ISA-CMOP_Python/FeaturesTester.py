from ProblemEvaluator import ProblemEvaluator
import numpy as np
from pymoo.problems import get_problem

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

from features.randomwalkfeatures import compute_neighbourhood_features


if __name__ == "__main__":
    
    # Flags for which feature set we want to test.
    run_global = False
    run_rw = True
    
    # Example problem.
    n_var = 5
    problem_name = "MW3"
    problem = get_problem(problem_name, n_var)
    instance_string = f"{problem_name}_d{n_var}"

    # Generate evaluator which can generate RW samples and LHS samples.
    evaluator = ProblemEvaluator([(instance_string, problem)])
    
    
    if run_global:
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

    if run_rw:
        ### RW FEATURES
        # Repeat the above but for RW samples.
        walks_neighbours_list = evaluator.sample_for_rw_features(
            problem,
            num_steps=100,
            step_size = 0.01,
            neighbourhood_size=2*n_var + 1
            )
        
        pop_walk_neighbourhood = evaluator.evaluate_populations_for_rw_features(problem, [walks_neighbours_list[0]])[0]
        
        pop_walk = pop_walk_neighbourhood[0]
        pop_neighbourhood = pop_walk_neighbourhood[1]

        # Neighbourhood features
        dist_x_avg, dist_f_avg, dist_c_avg, dist_f_dist_x_avg, dist_c_dist_x_avg, bhv_avg = compute_neighbourhood_features(pop_walk, pop_neighbourhood)