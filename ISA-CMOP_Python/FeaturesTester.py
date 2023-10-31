from ProblemEvaluator import ProblemEvaluator
import numpy as np
from pymoo.problems import get_problem


if __name__ == "__main__":
    # Example problem.
    n_var = 5
    problem_name = "MW3"
    problem = get_problem(problem_name, n_var)
    instance_string = f"{problem_name}_d{n_var}"

    # Generate evaluator which can generate RW samples and LHS samples.
    evaluator = ProblemEvaluator([(instance_string, problem)])

    ### GLOBAL FEATURES.

    # Generate distributed samples and evaluate populations on these samples.
    distributed_samples = evaluator.sample_for_global_features(
        problem, 2, method="lhs.scipy"
    )
    pops_global = evaluator.evaluate_populations_for_global_features(
        problem, distributed_samples
    )

    # Now we can extract individual samples for computing features.
    pop_global_test = pops_global[0]

    ### RW FEATURES
    # Repeat the above but for RW samples.
    # walks_neighbours_list = evaluator.sample_for_rw_features(problem)
