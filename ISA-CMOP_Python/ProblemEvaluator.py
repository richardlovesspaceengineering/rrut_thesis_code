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
        Generate starting zones (represented as binary arrays) at every 2^n/n-th vertex of the search space.
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
    
    def make_rw_generator(self, problem):
        n_var = problem.n_var

        # Experimental setup of Alsouly
        neighbourhood_size = 2 * n_var + 1
        num_steps = 1000
        step_size = 0.01  # 1% of the range of the instance domain

        # Bounds of the decision variables.
        x_lower = problem.xl
        x_upper = problem.xu
        bounds = np.vstack((x_lower, x_upper))

        # RW Generator Object
        rw = RandomWalk(bounds, num_steps, step_size, neighbourhood_size)
        return rw
    
    def sample_for_rw_features(self, problem):
        """
        Generate RW samples.
        
        For a problem of dimension n, we generate n random walks with the following properties:
        
        
        Random walk features are then computed for each walk separately prior to aggregation.\
            
        Returns a list of tuples of length n in (walk, neighbours) pairs.
        """
        
        print("Generating samples for RW features...")

        # Make RW generator object.
        rw = self.make_rw_generator(problem)
        
        # Progressive RWs will start at every 2/n-th vertex of the search space
        starting_zones = self.generate_binary_patterns(problem.n_var)
        
        walks_neighbours_list = []
        
        for ctr, starting_zone in enumerate(starting_zones):
                # Generate random walk starting at this iteration's starting zone.
                walk = rw.do_progressive_walk(seed=ctr, starting_zone = starting_zone)
                
                # Generate neighbours for each step on the walk. Currently we just randomly sample points in the [-stepsize, stepsize] hypercube
                neighbours = rw.generate_neighbours_for_walk(walk)
                
                walks_neighbours_list.append((walk, neighbours))

        print("Sampled for RW features (generated walks + neighbours)")
        return walks_neighbours_list
    
    def evaluate_populations_for_rw_features(self, problem, walks_neighbours_list):
        
        print("Evaluating populations for this sample... (ranks on for walk steps, off for neighbours)")
        
        # Lists saving populations and neighbours for features evaluation later.
        pops_walks_neighbours = []
        
        for ctr, walk_neighbours_pair in enumerate(walks_neighbours_list):
            walk = walk_neighbours_pair[0]
            neighbours = walk_neighbours_pair[1]
            
            pop_neighbours_list = []
            
            # Generate populations for walk and neighbours separately.
            # TODO: check i-th row of pop_walk corresponds to rows i*neighbourhood_size to (i+1)*neighbourhood_size of pop_neighbours.
            
            # Example: if n = 3, then pop_walk.obj[0,:] has pop_neighbours.obj[0:2,:] as its neighbours.
            pop_walk = Population(problem, n_individuals=walk.shape[0])
            
            
            # Evaluate each neighbourhood.
            for neighbourhood in neighbours:
                # None of the features related to neighbours ever require knowledge of the neighbours ranks relative to each other.
                pop_neighbourhood = Population(problem, n_individuals=neighbourhood.shape[0])
                pop_neighbourhood.evaluate(neighbourhood, eval_fronts=False) #TODO: eval fronts and include current step on walk.
                pop_neighbours_list.append(pop_neighbourhood)
            
            # Evaluate populations fully.
            pop_walk.evaluate(walk, eval_fronts=True)
            
            print("Evaluated population {} of {}.".format(ctr + 1, len(walks_neighbours_list) ))
            
            # Append to lists
            pops_walks_neighbours.append((pop_walk, pop_neighbours_list))
            
        print("Evaluated for RW features (generated walks + neighbours)")
        return pops_walks_neighbours
    
    def evaluate_rw_features_for_one_sample(self, pops_walks_neighbours):
        """
        Evaluate the RW features for one sample (i.e a set of random walks)
        """
        pops_walks = [t[0] for t in pops_walks_neighbours]
        pops_neighbours_list = [t[1] for t in pops_walks_neighbours]
        rw_features_single_sample = MultipleRandomWalkAnalysis(pops_walks, pops_neighbours_list)
        rw_features_single_sample.eval_features_for_all_populations()
        return rw_features_single_sample
    
    def aggregate_rw_features_across_samples(self, rw_features_all_samples):
        """
        Aggregate the features computed across each of the n independent samples.
        """
        return None

    def do(self, num_samples):
        for instance_name, problem_instance in self.instances:

            print(
                " ---------------- Evaluating instance: "
                + instance_name
                + " ----------------"
            )
            
            # We need to generate 30 samples per instance.
            for i in range(num_samples):
                
                print("Initialising sample {} of {}...".format(i+1, num_samples))
            
                # Evaluate Global features.

                ## Evaluate RW features.
                
                # Begin by sampling.
                walks_neighbours_list = self.sample_for_rw_features(problem_instance)
                
                # Then evaluate populations across the independent random walks.
                pops_walks_neighbours = self.evaluate_populations_for_rw_features(problem_instance, walks_neighbours_list)
                
                # Finally, evaluate features.
                rw_features = self.evaluate_rw_features_for_one_sample(pops_walks_neighbours)
            # 

            # TODO: save results to numpy binary format using savez. Will need to write functions that do so, and ones that can create a population by reading these in.

            # Evaluate features and put into landscape.
            # landscape = self.evaluate_features(pops)

            # # Append metrics to features dataframe.
            # aggregated_table = landscape.make_aggregated_feature_table(instance_name)

            # # Append the aggregated_table to features_table
            # self.features_table = pd.concat(
            #     [self.features_table, aggregated_table], ignore_index=True
            # )

            # Log success.
            print("Success!")

        # Save to a csv
        self.features_table.to_csv("features.csv", index=False)  # Save to a CSV file

    def evaluate_features(self, pops):
        """
        THIS IS ABOUT TO BE ARCHIVED
        """
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