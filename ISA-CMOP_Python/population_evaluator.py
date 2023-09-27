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
from sampling.PolynomialMutation import PolynomialMutation


if __name__ == "__main__":
    problem = MW7(n_dim=10)  # use default dimensionality.
    n_variables = problem.dim

    # Experimental setup of Alsouly
    # n_points = n_variables * 10**3
    # n_points = n_variables * 2 * 10**3
    # n_points = 3000
    # n_points = 5
    neighbourhood_size = 2 * n_variables + 1
    num_steps = int(n_variables / neighbourhood_size * 10**3 * 1.5)
    # num_steps = 10
    step_size = 0.2  # 2% of the range of the instance domain

    # Bounds of the decision variables.
    x_lower = problem.lb
    x_upper = problem.ub
    bounds = np.vstack((x_lower, x_upper))

    num_samples = 150

    # Run feature eval multiple times.
    pops = []
    rw = RandomWalk(bounds, num_steps, step_size)
    for ctr in range(num_samples):
        # Simulate RW
        walk = rw._do(seed=ctr)

        # Generate neighbours.
        new_walk = rw.generate_neighbours_for_walk(walk, neighbourhood_size)

        pop = Population(problem, n_individuals=num_steps)
        pop.evaluate(walk)
        print(
            "Evaluated rank and crowding for RW population {} of {}".format(
                ctr + 1, num_samples
            )
        )
        pops.append(pop)

    # Saving results to pickle file.
    with open("data/{}_d2_pop_data.pkl".format(problem.problem_name), "wb") as outp:
        pickle.dump(pops, outp, -1)

# %%
