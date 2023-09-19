from features.cv_distr import cv_distr
from features.cv_mdl import cv_mdl
from features.dist_corr import dist_corr
from features.f_corr import f_corr
from features.f_decdist import f_decdist
from features.f_skew import f_skew
from cases.MW_setup import MW3
import numpy as np
from optimisation.model.individual import Individual
from optimisation.model.population import Population
from features.FitnessAnalysis import FitnessAnalysis
from features.RandomWalkAnalysis import RandomWalkAnalysis, MultipleRandomWalkAnalysis
from sampling.RandomSample import RandomSample
from sampling.RandomWalk import RandomWalk


if __name__ == "__main__":
    problem = MW3()  # use default dimensionality.
    n_variables = problem.dim

    # Experimental setup of Alsouly
    # n_points = n_variables * 10**3
    neighbourhood_size = 2 * n_variables + 1
    # num_steps = n_points / neighbourhood_size * 10**3
    # step_size = 0.02  # 2% of the range of the instance domain

    n_points = 5
    num_steps = 5
    step_size = 0.02

    # Decision variables - randomly generated in a basic way for now. Will need to consult Alsouly paper later to mimic their method.
    x_lower = problem.lb
    x_upper = problem.ub
    bounds = np.vstack((x_lower, x_upper))

    # Run random sampling and random walk.
    sample = RandomSample(bounds, n_points)._do()

    # Create the population and evaluate.
    pop_global = Population(problem, n_individuals=n_points)
    pop_global.evaluate(sample)

    # Now evaluate metrics.
    global_features = FitnessAnalysis(pop_global)
    global_features.eval_fitness_features()

    # Random walk
    walk = RandomWalk(bounds, num_steps, step_size, neighbourhood_size)._do(seed=123)
    pop_rw = Population(problem, n_individuals=num_steps)

    # Evaluate along the RW.
    pop_rw.evaluate(walk)

    # Compute RW features.
    pops = [pop_rw]
    rw_features = MultipleRandomWalkAnalysis(pops)
    rw_features.eval_features_for_all_populations()
    rw_features.aggregate_features()
