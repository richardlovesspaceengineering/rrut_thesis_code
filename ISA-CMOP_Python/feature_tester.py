from cases.MW_setup import MW3
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
    problem = MW3()  # use default dimensionality.
    n_variables = problem.dim

    # Experimental setup of Alsouly
    # n_points = n_variables * 10**3
    # n_points = 1000
    n_points = 5
    neighbourhood_size = 2 * n_variables + 1
    # num_steps = int(n_variables / neighbourhood_size * 10**3)
    num_steps = 10
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

    # Now evaluate metrics for all populations.
    PF = pops_global[0].extract_pf()

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
    landscape.map_features_to_instance_space()

    # Saving results to pickle file.
    with open("data/MW3_landscape_data.pkl", "wb") as outp:
        pickle.dump(landscape, outp, -1)

    # del landscape

    # with open("data/MW3_landscape_data.pkl", "rb") as inp:
    #     landscape = pickle.load(inp)
