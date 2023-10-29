# Other package imports
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# User packages.
from features.GlobalAnalysis import MultipleGlobalAnalysis
from features.RandomWalkAnalysis import MultipleRandomWalkAnalysis
from features.LandscapeAnalysis import LandscapeAnalysis

import numpy as np
from optimisation.model.population import Population
from sampling.RandomWalk import RandomWalk
import pickle


class ProblemEvaluator:
    def __init__(self, instances):
        self.instances = instances
        self.features_table = pd.DataFrame()
        

    def generate_binary_patterns(self, n):
        """
        Generate starting zones (represented as binary arrays) at every 2^n/n-th vertex.
        """

        num_patterns = 2**n
        patterns = []

        if n % 2 == 0:
            # Even n -> count from 2^n/n
            step = num_patterns // n
            for i in range(0, num_patterns, step):
                binary_pattern = np.binary_repr(i, width=n)
                patterns.append([int(bit) for bit in binary_pattern])
        else:
            # Odd n -> count from 0
            step = math.ceil(num_patterns / n)
            for i in range(0, num_patterns, step):
                binary_pattern = np.binary_repr(i, width=n)
                patterns.append([int(bit) for bit in binary_pattern])
        return patterns


    def do(self, num_samples):
        for instance_name, problem_instance in self.instances:
            self.num_samples = num_samples

            print(
                " ---------------- Evaluating instance: "
                + instance_name
                + " ----------------"
            )

            pops = self.evaluate_population(problem_instance, num_samples)

            # TODO: save results to numpy binary format using savez. Will need to write functions that do so, and ones that can create a population by reading these in.

            # Evaluate features and put into landscape.
            landscape = self.evaluate_features(pops)

            # Append metrics to features dataframe.
            aggregated_table = landscape.make_aggregated_feature_table(instance_name)

            # Append the aggregated_table to features_table
            self.features_table = pd.concat(
                [self.features_table, aggregated_table], ignore_index=True
            )

            # Log success.
            print("Success!")

        # Save to a csv
        self.features_table.to_csv("features.csv", index=False)  # Save to a CSV file

    def evaluate_population(self, problem, num_samples):
        n_var = problem.n_var
        
        # Generate starting zones binary patterns for random walk sampling.
        starting_zones = self.generate_binary_patterns(n_var)

        # Experimental setup of Alsouly
        neighbourhood_size = 2 * n_var + 1
        num_steps = int(n_var / neighbourhood_size * 10**3)
        step_size = 0.2  # 2% of the range of the instance domain

        # Bounds of the decision variables.
        x_lower = problem.xl
        x_upper = problem.xu
        bounds = np.vstack((x_lower, x_upper))

        # Run feature eval multiple times.
        pops = []
        rw = RandomWalk(bounds, num_steps, step_size)
        for ctr in range(num_samples):
            # TODO: Simulate RW (from every 2^n/n-th starting zone.)
            # for i in range(n_var):
                # walk = rw.do_progressive_walk(seed=ctr, starting_zone = )
            
            # TODO: generate neighbours from each walk.

            # Generate neighbours.
            new_walk = rw.generate_neighbours_for_walk(walk, neighbourhood_size)

            pop = Population(problem, n_individuals=num_steps)
            pop.evaluate(new_walk)
            

            pops.append(pop)
        print("Evaluated rank and crowding")
        return pops

    def evaluate_features(self, pops):
        # Evaluate landscape features.
        # Global features.
        global_features = MultipleGlobalAnalysis(pops)
        global_features.eval_features_for_all_populations()

        # Random walk features.
        rw_features = MultipleRandomWalkAnalysis(pops)
        rw_features.eval_features_for_all_populations()

        # Combine all features
        landscape = LandscapeAnalysis(global_features, rw_features)
        landscape.extract_feature_arrays()
        landscape.aggregate_features(YJ_transform=False)

        return landscape


if __name__ == "__main__":
    pe = ProblemEvaluator([])
    pe.generate_binary_patterns(3)