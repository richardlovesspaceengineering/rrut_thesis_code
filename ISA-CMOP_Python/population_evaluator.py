# %%
from cases.MW_setup import MW3, MW7
import numpy as np
from optimisation.model.population import Population
from features.FitnessAnalysis import MultipleFitnessAnalysis
from features.RandomWalkAnalysis import MultipleRandomWalkAnalysis
from sampling.RandomSample import RandomSample
from sampling.RandomWalk import RandomWalk
from features.LandscapeAnalysis import LandscapeAnalysis
import pickle
import matplotlib.pyplot as plt

if __name__ == "__main__":
    problem = MW7(n_dim=2)  # use default dimensionality.
    n_variables = problem.dim

    # Experimental setup of Alsouly
    n_points = n_variables * 10**3
    # n_points = 3000
    # n_points = 5
    neighbourhood_size = 2 * n_variables + 1
    num_steps = int(n_variables / neighbourhood_size * 10**3)
    # num_steps = 10
    step_size_prop = 0.02  # 2% of the range of the instance domain

    # Bounds of the decision variables.
    x_lower = problem.lb
    x_upper = problem.ub
    bounds = np.vstack((x_lower, x_upper))

    num_samples = 30

    # Run feature eval multiple times.
    pops_global = []
    pops_rw = []
    for ctr in range(num_samples):
        ## Populations for global features.
        sample = RandomSample(bounds, n_points)._do(seed=ctr)
        pop_global = Population(problem, n_individuals=n_points)
        pop_global.evaluate(sample)
        print(
            "Evaluated rank and crowding for global population {} of {}".format(
                ctr + 1, num_samples
            )
        )
        pops_global.append(pop_global)

        ## Populations for random walks.
        walk = RandomWalk(bounds, num_steps, step_size_prop, neighbourhood_size)._do(
            seed=ctr
        )
        pop_rw = Population(problem, n_individuals=num_steps)
        pop_rw.evaluate(walk)
        print(
            "Evaluated rank and crowding for RW population {} of {}".format(
                ctr + 1, num_samples
            )
        )
        pops_rw.append(pop_rw)

    # Saving results to pickle file.
    with open("data/{}_d2_pop_data.pkl".format(problem.problem_name), "wb") as outp:
        pops = [pops_global, pops_rw]
        pickle.dump(pops, outp, -1)

# %%
