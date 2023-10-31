# Other package imports
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import time


# User packages.
from features.GlobalAnalysis import MultipleGlobalAnalysis
from features.RandomWalkAnalysis import MultipleRandomWalkAnalysis
from features.LandscapeAnalysis import LandscapeAnalysis
from optimisation.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling

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
        step = math.ceil(num_patterns / n)
    
        for i in range(0, num_patterns, step):
            binary_pattern = np.binary_repr(i, width=n)
            patterns.append([int(bit) for bit in binary_pattern])
        return patterns
    
    def make_rw_generator(self, problem):
        n_var = problem.n_var

        # Experimental setup of Alsouly
        neighbourhood_size = 2 * n_var + 1
        num_steps = 10
        step_size = 0.01  # 1% of the range of the instance domain

        # Bounds of the decision variables.
        x_lower = problem.xl
        x_upper = problem.xu
        bounds = np.vstack((x_lower, x_upper))

        # RW Generator Object
        rw = RandomWalk(bounds, num_steps, step_size, neighbourhood_size)
        return rw
    
    import time

    def sample_for_rw_features(self, problem):
        """
        Generate RW samples.
        
        For a problem of dimension n, we generate n random walks with the following properties.
        
        Random walk features are then computed for each walk separately prior to aggregation.
            
        Returns a list of tuples of length n in (walk, neighbours) pairs.
        """

        # Make RW generator object.
        rw = self.make_rw_generator(problem)
        
        # Progressive RWs will start at every 2/n-th vertex of the search space
        starting_zones = self.generate_binary_patterns(problem.n_var)
        
        walks_neighbours_list = []
        
        print("")
        print("Generating samples (walks + neighbours) for RW features with the following properties:")
        print("- Number of walks: {}".format(len(starting_zones)))
        print("")
        
        for ctr, starting_zone in enumerate(starting_zones):
            start_time = time.time()  # Record the start time
            # Generate random walk starting at this iteration's starting zone.
            walk = rw.do_progressive_walk(seed=ctr, starting_zone=starting_zone)
            
            # Generate neighbors for each step on the walk. Currently, we just randomly sample points in the [-stepsize, stepsize] hypercube
            neighbours = rw.generate_neighbours_for_walk(walk)
            end_time = time.time()  # Record the end time
            elapsed_time = end_time - start_time

            walks_neighbours_list.append((walk, neighbours))
            print("Generated RW sample {} of {} in {:.2f} seconds.".format(ctr + 1, len(starting_zones), elapsed_time))

        return walks_neighbours_list

    
    def evaluate_populations_for_rw_features(self, problem, walks_neighbours_list):
        
        print("\nEvaluating populations for this sample... (ranks on for walk steps, off for neighbours)")
        
        # Lists saving populations and neighbours for features evaluation later.
        pops_walks_neighbours = []
        
        for ctr, walk_neighbours_pair in enumerate(walks_neighbours_list):
            walk = walk_neighbours_pair[0]
            neighbours = walk_neighbours_pair[1]
            
            pop_neighbours_list = []
            
            start_time = time.time()  # Record the start time
            
            # Generate populations for walk and neighbours separately.
            pop_walk = Population(problem, n_individuals=walk.shape[0])
            
            # Evaluate each neighbourhood.
            for neighbourhood in neighbours:
                # None of the features related to neighbours ever require knowledge of the neighbours ranks relative to each other.
                pop_neighbourhood = Population(problem, n_individuals=neighbourhood.shape[0])
                pop_neighbourhood.evaluate(neighbourhood, eval_fronts=False) #TODO: eval fronts and include current step on walk.
                pop_neighbours_list.append(pop_neighbourhood)
            
            # Evaluate populations fully.
            pop_walk.evaluate(walk, eval_fronts=True)
            
            # Record the end time.
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            print("Evaluated RW population {} of {} in {:.2f} seconds.".format(ctr + 1, len(walks_neighbours_list), elapsed_time))

            # Append to lists
            pops_walks_neighbours.append((pop_walk, pop_neighbours_list))
            
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
    
    def do_random_walk_analysis(self, problem_instance, num_samples):
        
        # Per-sample arrays.
        rw_features_list = []
        
        # We need to generate 30 samples per instance.
        for i in range(num_samples):
            
            print("Initialising RW sample {} of {}...".format(i+1, num_samples))
        
            ## Evaluate RW features.
            
            # Begin by sampling.
            walks_neighbours_list = self.sample_for_rw_features(problem_instance)
            
            # Then evaluate populations across the independent random walks.
            pops_walks_neighbours = self.evaluate_populations_for_rw_features(problem_instance, walks_neighbours_list)
            
            # Finally, evaluate features.
            rw_features_list.append(self.evaluate_rw_features_for_one_sample(pops_walks_neighbours))
        
        # Now concatenate the rw_features into one large MultipleRandomWalkAnalysis object.
        rw_features = MultipleRandomWalkAnalysis.concatenate_multiple_analyses(rw_features_list)
        return rw_features

    def sample_for_global_features(self, problem, num_samples):
        
        n_var = problem.n_var
        
        distributed_samples = []
        
        # Experimental setup of Liefooghe2021.
        num_points = 200 * n_var
        iterations = num_points
        
        print("")
        print("Generating distributed samples for Global features with the following properties:")
        print("- Num. points: {}".format(num_points))
        print("- Num. iterations: {}".format(iterations))
        print("")
        
        for i in range(num_samples):
            start_time = time.time()  # Record the start time
            lhs = LatinHypercubeSampling(criterion="maximin", iterations=iterations)
            lhs.do(n_samples=num_points, x_lower=problem.xl, x_upper=problem.xu)
            end_time = time.time()  # Record the end time
            elapsed_time = end_time - start_time

            distributed_samples.append(lhs.x)
            print("Generated Global sample {} of {} in {:.2f} seconds.".format(i + 1, num_samples, elapsed_time))
            
        return distributed_samples

    

    def evaluate_populations_for_global_features(self, problem, distributed_samples):
        print("Evaluating populations for global samples...")

        pops_global = []

        for ctr, sample in enumerate(distributed_samples):
            start_time = time.time()  # Record the start time
            pop_global = Population(problem, n_individuals=sample.shape[0])
            pop_global.evaluate(sample, eval_fronts=True)
            end_time = time.time()  # Record the end time
            elapsed_time = end_time - start_time

            pops_global.append(pop_global)
            print("Evaluated Global population {} of {} in {:.2f} seconds.".format(ctr + 1, len(distributed_samples), elapsed_time))

        return pops_global

    
    def evaluate_global_features(self, pops_global):
        
        global_features = MultipleGlobalAnalysis(pops_global)
        global_features.eval_features_for_all_populations()
        
        return global_features
    
    def do_global_analysis(self, problem_instace, num_samples):
        
        # We need to generate 30 samples per instance.
        
        # Generate distributed samples.
        distributed_samples = self.sample_for_global_features(problem_instace, num_samples)
        
        # Evaluate populations
        pops_global = self.evaluate_populations_for_global_features(problem_instace, distributed_samples)
        
        # Evaluate features.
        global_features = self.evaluate_global_features(pops_global)
        
        return global_features
        

    def do(self, num_samples):
        for instance_name, problem_instance in self.instances:

            print(
                " ---------------- Evaluating instance: "
                + instance_name
                + " ----------------"
            )
            
            # RW Analysis.
            print(
                " \n ~~~~~~~~~~~~ RW Analysis "
                + " ~~~~~~~~~~~~ \n"
            )
            rw_features = self.do_random_walk_analysis(problem_instance, num_samples)
            
            # Global Analysis.
            
            # RW Analysis.
            print(
                " \n ~~~~~~~~~~~~ Global Analysis "
                + " ~~~~~~~~~~~~ \n"
            )
            global_features = self.do_global_analysis(problem_instance, num_samples)
            
            
            # Overall landscape analysis - putting it all together.
            landscape = LandscapeAnalysis(global_features, rw_features)
            landscape.extract_feature_arrays()
            landscape.aggregate_features(YJ_transform=False)

            # TODO: save results to numpy binary format using savez. Will need to write functions that do so, and ones that can create a population by reading these in.

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


if __name__ == "__main__":
    
    # Making sure the binary pattern generator is generating the right number of starting zones.
    pe = ProblemEvaluator([])
    print(len(pe.generate_binary_patterns(10)))
    