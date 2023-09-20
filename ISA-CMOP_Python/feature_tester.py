from cases.MW_setup import MW3
import numpy as np
from optimisation.model.individual import Individual
from optimisation.model.population import Population
from features.FitnessAnalysis import MultipleFitnessAnalysis
from features.RandomWalkAnalysis import MultipleRandomWalkAnalysis
from sampling.RandomSample import RandomSample
from sampling.RandomWalk import RandomWalk
from features.LandscapeAnalysis import LandscapeAnalysis


if __name__ == "__main__":
    problem = MW3()  # use default dimensionality.
    n_variables = problem.dim

    # Experimental setup of Alsouly
    # n_points = n_variables * 10**3
    neighbourhood_size = 2 * n_variables + 1
    # num_steps = n_points / neighbourhood_size * 10**3
    # step_size = 0.02  # 2% of the range of the instance domain

    n_points = n_variables
    num_steps = 5
    step_size = 0.02

    # Bounds of the decision variables.
    x_lower = problem.lb
    x_upper = problem.ub
    bounds = np.vstack((x_lower, x_upper))

    num_samples = 2

    # Run feature eval multiple times.
    pops_global = []
    pops_rw = []
    for ctr in range(num_samples):
        ## Populations for global features.
        sample = RandomSample(bounds, n_points)._do(seed=ctr)
        pop_global = Population(problem, n_individuals=n_points)
        pop_global.evaluate(sample)
        pops_global.append(pop_global)

        ## Populations for random walks.
        walk = RandomWalk(bounds, num_steps, step_size, neighbourhood_size)._do(
            seed=ctr
        )
        pop_rw = Population(problem, n_individuals=num_steps)
        pop_rw.evaluate(walk)
        pops_rw.append(pop_rw)

    # Now evaluate metrics for all populations.

    # Global.
    global_features = MultipleFitnessAnalysis(pops_global)
    global_features.eval_features_for_all_populations()
    global_features.aggregate_features()

    # Random walk.
    rw_features = MultipleRandomWalkAnalysis(pops_rw)
    rw_features.eval_features_for_all_populations()
    rw_features.aggregate_features()

    # Combine all features.
    landscape = LandscapeAnalysis(global_features, rw_features)
    landscape.combine_features()
