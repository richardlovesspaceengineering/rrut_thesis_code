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


if __name__ == "__main__":
    problem = MW3()  # use default dimensionality.
    n_variables = problem.dim
    n_points = 5

    # Decision variables - randomly generated in a basic way for now. Will need to consult Alsouly paper later to mimic their method.
    x_lower = problem.lb
    x_upper = problem.ub
    np.random.seed(1)
    x = np.random.uniform(x_lower, x_upper, size=(n_points, n_variables))

    # Create the population and evalute.
    pop = Population(problem, n_individuals=n_points)
    pop.evaluate(x)

    # Now evaluate metrics.
    features = FitnessAnalysis(pop)
    features.eval_fitness_features()
