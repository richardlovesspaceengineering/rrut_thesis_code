# Other package imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import time
import os

# User packages.
from features.feature_helpers import generate_bounds_from_problem
from features.GlobalAnalysis import MultipleGlobalAnalysis
from features.RandomWalkAnalysis import MultipleRandomWalkAnalysis
from features.AdaptiveWalkAnalysis import MultipleAdaptiveWalkAnalysis
from features.LandscapeAnalysis import LandscapeAnalysis
from optimisation.operators.sampling.latin_hypercube_sampling import (
    LatinHypercubeSampling,
)
from optimisation.operators.sampling.random_sampling import RandomSampling

import numpy as np
from optimisation.model.population import Population
from optimisation.operators.sampling.RandomWalk import RandomWalk
from optimisation.operators.sampling.AdaptiveWalk import AdaptiveWalk
from features.Analysis import Analysis, MultipleAnalysis


class ProblemEvaluator:
    def __init__(self, instance, instance_name, mode="eval"):
        """
        Possible modes are eval and debug.
        """
        self.instance = instance
        self.instance_name = instance_name
        self.features_table = pd.DataFrame()
        self.csv_filename = "features.csv"
        self.mode = mode
        print("Initialising evaluator in {} mode.".format(self.mode))

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

    def make_rw_generator(self, problem, num_steps, step_size, neighbourhood_size):
        # Bounds of the decision variables.
        x_lower = problem.xl
        x_upper = problem.xu
        bounds = np.vstack((x_lower, x_upper))

        # RW Generator Object
        rw = RandomWalk(bounds, num_steps, step_size, neighbourhood_size)
        return rw

    def sample_for_rw_features(self, problem, num_samples):
        """
        Generate RW samples.

        For a problem of dimension n, we generate n random walks with the following properties.

        Random walk features are then computed for each walk separately prior to aggregation.

        Returns a list of tuples of length n in (walk, neighbours) pairs.
        """

        n_var = problem.n_var

        if self.mode == "eval":
            # Experimental setup of Alsouly
            neighbourhood_size = 2 * n_var + 1
            num_steps = 1000
            step_size = 0.01  # 1% of the range of the instance domain
        elif self.mode == "debug":
            # Runs quickly
            neighbourhood_size = 2 * n_var + 1
            num_steps = 35
            step_size = 0.01  # 1% of the range of the instance domain

        # Make RW generator object.
        rw = self.make_rw_generator(problem, num_steps, step_size, neighbourhood_size)

        # Progressive RWs will start at every 2/n-th vertex of the search space
        starting_zones = self.generate_binary_patterns(problem.n_var)

        print("")
        print(
            "Generating {} samples (walks + neighbours) for RW features with the following properties:".format(
                num_samples
            )
        )
        print("- Number of walks: {}".format(len(starting_zones)))
        print("- Number of steps per walk: {}".format(num_steps))
        print("- Step size (% of instance domain): {}".format(step_size * 100))
        print("- Neighbourhood size: {}".format(neighbourhood_size))
        print("")

        # Initialise list of walks and neighbours across all samples.
        walks_neighbours_list_all_samples = []

        for i in range(num_samples):
            start_time = time.time()  # Record the start time for this sample
            walks_neighbours_list = []  # Walks and neighbours for this sample.

            for ctr, starting_zone in enumerate(starting_zones):
                # Generate random walk starting at this iteration's starting zone.
                walk = rw.do_progressive_walk(seed=None, starting_zone=starting_zone)

                # Generate neighbors for each step on the walk. Currently, we just randomly sample points in the [-stepsize, stepsize] hypercube
                neighbours = rw.generate_neighbours_for_walk(walk)
                walks_neighbours_list.append((walk, neighbours))
                print(
                    "Generated RW {} of {} (for this sample).".format(
                        ctr + 1,
                        len(starting_zones),
                    )
                )

            # Now append this sample's walks and neighbours to overall list.
            walks_neighbours_list_all_samples.append(walks_neighbours_list)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(
                "Generated set of RWs {} of {} in {:.2f} seconds.\n".format(
                    i + 1, num_samples, elapsed_time
                )
            )

        return walks_neighbours_list_all_samples

    def evaluate_populations_for_rw_features(
        self, problem, walks_neighbours_list_all_samples
    ):
        """
        walks_neighbours_list_all_samples is a list containing num_samples lists with the following structure:

        [[(walk00, neighbours00), (walk01, neighbours01), ...], [(walk10, neighbours10), (walk11, neighbours11), ...], ...]

        """
        print(
            "\nEvaluating populations for these walk samples... (ranks off for walk steps, on for neighbours)"
        )

        pop_walks_neighbours_all_samples = []

        for ctr, sample in enumerate(walks_neighbours_list_all_samples):
            start_time = time.time()  # Record the start time for this sample
            pops_walks_neighbours = (
                []
            )  # List of walks-neighbours pairs for this sample.
            for i, walk_neighbours_pair in enumerate(sample):
                walk = walk_neighbours_pair[0]
                neighbours = walk_neighbours_pair[1]

                pop_neighbours_list = []

                # Generate populations for walk and neighbours separately.
                pop_walk = Population(problem, n_individuals=walk.shape[0])

                # Evaluate each neighbourhood.
                for neighbourhood in neighbours:
                    # None of the features related to neighbours ever require knowledge of the neighbours ranks relative to each other.
                    pop_neighbourhood = Population(
                        problem, n_individuals=neighbourhood.shape[0]
                    )
                    pop_neighbourhood.evaluate(
                        neighbourhood, eval_fronts=True
                    )  # TODO: eval fronts and include current step on walk.
                    pop_neighbours_list.append(pop_neighbourhood)

                # Evaluate steps fully.
                pop_walk.evaluate(walk, eval_fronts=False)

                # Append to list of walks for this sample.
                pops_walks_neighbours.append((pop_walk, pop_neighbours_list))

            # Append the populations for this sample to the list.
            pop_walks_neighbours_all_samples.append(pops_walks_neighbours)

            # Record the end time.
            end_time = time.time()
            elapsed_time = end_time - start_time

            print(
                "Evaluated population {} of {} in {:.2f} seconds.".format(
                    ctr + 1, len(sample), elapsed_time
                )
            )

        return pop_walks_neighbours_all_samples

    def evaluate_rw_features_for_one_sample(self, pops_walks_neighbours):
        """
        Evaluate the RW features for one sample (i.e a set of random walks)
        """
        pops_walks = [t[0] for t in pops_walks_neighbours]
        pops_neighbours_list = [t[1] for t in pops_walks_neighbours]
        rw_features_single_sample = MultipleRandomWalkAnalysis(
            pops_walks, pops_neighbours_list
        )
        rw_features_single_sample.eval_features_for_all_populations()
        return rw_features_single_sample

    def do_random_walk_analysis(self, problem, num_samples):
        walks_neighbours_list_all_samples = self.sample_for_rw_features(
            problem, num_samples
        )

        pop_walks_neighbours_all_samples = self.evaluate_populations_for_rw_features(
            problem, walks_neighbours_list_all_samples
        )

        rw_features_list = []
        for ctr, pop_walks_neighbours in enumerate(pop_walks_neighbours_all_samples):
            # Finally, evaluate features.
            rw_features_list.append(
                self.evaluate_rw_features_for_one_sample(pop_walks_neighbours)
            )
            print(
                "Evaluated features for RW sample {} out of {}".format(
                    ctr + 1, len(pop_walks_neighbours_all_samples)
                )
            )

        # Now concatenate the rw_features into one large MultipleRandomWalkAnalysis object.
        rw_features = MultipleAnalysis.concatenate_multiple_analyses(
            rw_features_list, MultipleRandomWalkAnalysis
        )
        return rw_features

    def sample_for_global_features(self, problem, num_samples, method="lhs.scipy"):
        n_var = problem.n_var

        distributed_samples = []

        if self.mode == "eval":
            # Experimental setup of Liefooghe2021.
            num_points = int(n_var * 200)
            iterations = num_points
        elif self.mode == "debug":
            # Runs quickly.
            num_points = int(n_var * 50)
            iterations = num_points

        # Split the method string to extract the method name
        method_parts = method.split(".")
        if len(method_parts) == 2:
            method_name = method_parts[0]
            lhs_method_name = method_parts[1]
        else:
            method_name = method  # Use the full method string as the method name

        print(
            "Generating distributed samples for Global features with the following properties:"
        )
        print("- Num. points: {}".format(num_points))
        print("- Num. iterations: {}".format(iterations))
        print("- Method: {}".format(method))
        print("")

        for i in range(num_samples):
            start_time = time.time()  # Record the start time
            if method_name == "lhs":
                sampler = LatinHypercubeSampling(
                    criterion="maximin", iterations=iterations, method=lhs_method_name
                )
            elif method_name == "uniform":
                sampler = RandomSampling()

            sampler.do(n_samples=num_points, x_lower=problem.xl, x_upper=problem.xu)
            distributed_samples.append(sampler.x)
            end_time = time.time()  # Record the end time
            elapsed_time = end_time - start_time

            print(
                "Generated Global sample {} of {} in {:.2f} seconds.".format(
                    i + 1, num_samples, elapsed_time
                )
            )

        return distributed_samples

    def evaluate_populations_for_global_features(self, problem, distributed_samples):
        print("\nEvaluating populations for global samples...")

        pops_global = []

        for ctr, sample in enumerate(distributed_samples):
            start_time = time.time()  # Record the start time
            pop_global = Population(problem, n_individuals=sample.shape[0])
            pop_global.evaluate(sample, eval_fronts=True)
            end_time = time.time()  # Record the end time
            elapsed_time = end_time - start_time

            pops_global.append(pop_global)
            print(
                "Evaluated Global population {} of {} in {:.2f} seconds.".format(
                    ctr + 1, len(distributed_samples), elapsed_time
                )
            )

        return pops_global

    def evaluate_global_features(self, pops_global):
        global_features = MultipleGlobalAnalysis(pops_global)
        global_features.eval_features_for_all_populations()

        return global_features

    def do_global_analysis(self, problem, num_samples):
        # Generate distributed samples.
        distributed_samples = self.sample_for_global_features(
            problem, num_samples, method="lhs.scipy"
        )

        # Evaluate populations
        pops_global = self.evaluate_populations_for_global_features(
            problem, distributed_samples
        )

        # Evaluate features.
        global_features = self.evaluate_global_features(pops_global)

        return global_features

    def sample_for_aw_features(self, problem, num_samples):
        """
        Generate adaptive walk samples.
        """
        n_var = problem.n_var

        if self.mode == "eval":
            # Experimental setup of Alsouly
            neighbourhood_size = 2 * (2 * n_var + 1)
            max_steps = 1000
            step_size = 0.01  # 1% of the range of the instance domain

            # TODO: set this value
            distributed_sample_size = n_var * 10
        elif self.mode == "debug":
            # Runs quickly
            neighbourhood_size = 2 * n_var + 1
            max_steps = 100
            step_size = 0.01  # 1% of the range of the instance domain
            distributed_sample_size = 5

        # Now initialise adaptive walk generator.
        awGenerator = AdaptiveWalk(
            generate_bounds_from_problem(problem),
            max_steps,
            step_size,
            neighbourhood_size,
            problem,
        )

        print(
            "Generating {} adaptive walk samples with the following properties:".format(
                num_samples
            )
        )
        print(
            "- Number of walks (equals number of LHS samples): {}".format(
                distributed_sample_size
            )
        )
        print("- Maximum number of steps: {}".format(max_steps))
        print("- Step size (% of instance domain): {}".format(step_size * 100))
        print("- Neighbourhood size: {}\n".format(neighbourhood_size))

        # Initialise list of walks and neighbours across all samples.
        adaptive_walks_neighbours_list_all_samples = []

        for ctr in range(num_samples):
            start_time = time.time()
            sampler = LatinHypercubeSampling(criterion="maximin", method="scipy")
            sampler.do(
                n_samples=distributed_sample_size,
                x_lower=problem.xl,
                x_upper=problem.xu,
            )

            distributed_sample = sampler.x
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(
                "Generated starting points for AWs using LHS in {:.2f} seconds.\n".format(
                    elapsed_time
                )
            )

            # Sample along walk.
            adaptive_walks_neighbours_list = []
            start_time = time.time()
            for i in range(distributed_sample.shape[0]):
                new_walk = awGenerator.do_adaptive_phc_walk_for_starting_point(
                    distributed_sample[i, :], constrained_ranks=True
                )  # TODO: add both constrained and unconstrained.
                new_neighbours = awGenerator.generate_neighbours_for_walk(new_walk)
                adaptive_walks_neighbours_list.append((new_walk, new_neighbours))
                print(
                    "Generated AW {} of {} (for this sample). Length: {}".format(
                        i + 1,
                        distributed_sample_size,
                        adaptive_walks_neighbours_list[i][0].shape[0],
                    )
                )
            adaptive_walks_neighbours_list_all_samples.append(
                adaptive_walks_neighbours_list
            )
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(
                "Generated set of AWs {} of {} in {:.2f} seconds.\n".format(
                    ctr + 1, num_samples, elapsed_time
                )
            )

        return adaptive_walks_neighbours_list_all_samples

    def evaluate_populations_for_aw_features(
        self, problem, adaptive_walks_neighbours_list_all_samples
    ):
        # Lists saving populations and neighbours for features evaluation later.
        pops_adaptive_walks_neighbours_all_samples = (
            self.evaluate_populations_for_rw_features(
                problem, adaptive_walks_neighbours_list_all_samples
            )
        )

        return pops_adaptive_walks_neighbours_all_samples

    def evaluate_aw_features_for_one_sample(self, pops_adaptive_walks_neighbours):
        """
        Evaluate the RW features for one sample (i.e a set of random walks)
        """
        pops_walks = [t[0] for t in pops_adaptive_walks_neighbours]
        pops_neighbours_list = [t[1] for t in pops_adaptive_walks_neighbours]
        rw_features_single_sample = MultipleAdaptiveWalkAnalysis(
            pops_walks, pops_neighbours_list
        )
        rw_features_single_sample.eval_features_for_all_populations()
        return rw_features_single_sample

    def do_adaptive_walk_analysis(self, problem, num_samples):
        adaptive_walks_neighbours_list_all_samples = self.sample_for_aw_features(
            problem, num_samples
        )

        pop_adaptive_walks_neighbours_all_samples = (
            self.evaluate_populations_for_aw_features(
                problem, adaptive_walks_neighbours_list_all_samples
            )
        )

        aw_features_list = []
        for ctr, pop_walks_neighbours in enumerate(
            pop_adaptive_walks_neighbours_all_samples
        ):
            # Finally, evaluate features.
            aw_features_list.append(
                self.evaluate_aw_features_for_one_sample(pop_walks_neighbours)
            )
            print(
                "Evaluated features for AW sample {} out of {}".format(
                    ctr + 1, len(pop_adaptive_walks_neighbours_all_samples)
                )
            )

        # Now concatenate the rw_features into one large MultipleRandomWalkAnalysis object.
        aw_features = MultipleAnalysis.concatenate_multiple_analyses(
            aw_features_list, MultipleAdaptiveWalkAnalysis
        )
        return aw_features

    def do(self, num_samples, save_arrays):
        print(
            "\n------------------------ Evaluating instance: "
            + self.instance_name
            + " ------------------------"
        )

        # RW Analysis.
        print(
            " \n ~~~~~~~~~~~~ RW Analysis for "
            + self.instance_name
            + " ~~~~~~~~~~~~ \n"
        )
        rw_features = self.do_random_walk_analysis(
            self.instance,
            num_samples,
        )
        rw_features.export_unaggregated_features(self.instance_name, "rw", save_arrays)

        # Global Analysis.
        print(
            " \n ~~~~~~~~~~~~ Global Analysis for "
            + self.instance_name
            + " ~~~~~~~~~~~~ \n"
        )
        global_features = self.do_global_analysis(self.instance, num_samples)
        global_features.export_unaggregated_features(
            self.instance_name, "glob", save_arrays
        )

        # Adaptive Walk Analysis.
        print(
            " \n ~~~~~~~~~~~~ AW Analysis for "
            + self.instance_name
            + " ~~~~~~~~~~~~ \n"
        )
        aw_features = self.do_adaptive_walk_analysis(
            self.instance,
            num_samples,
        )
        rw_features.export_unaggregated_features(self.instance_name, "aw", save_arrays)

        # Overall landscape analysis - putting it all together.
        landscape = LandscapeAnalysis(global_features, rw_features, aw_features)

        # TODO: save results to numpy binary format using savez. Will need to write functions that do so, and ones that can create a population by reading these in.

        # Perform aggregation.
        landscape.aggregate_features()

        # Append metrics to features dataframe.

        aggregated_table = landscape.make_aggregated_feature_table(self.instance_name)

        # Append the aggregated_table to features_table
        self.features_table = pd.concat(
            [self.features_table, aggregated_table], ignore_index=True
        )

        # Log success.
        print("Success!")

        # Save to a csv at end of every problem instance.
        self.append_dataframe_to_csv(self.csv_filename, self.features_table)

        print("Successfully appended aggregated results to csv file.\n\n")

    def append_dataframe_to_csv(
        self,
        existing_csv,
        df_to_append,
        overwrite_existing=True,
        overwrite_column="Name",
    ):
        # Check if the existing CSV file already exists
        if os.path.isfile(existing_csv):
            # Read the existing CSV file into a DataFrame
            existing_df = pd.read_csv(existing_csv)

            if overwrite_existing:
                # Filter out rows with matching values in the specified column
                existing_df = existing_df[
                    ~existing_df[overwrite_column].isin(df_to_append[overwrite_column])
                ]

            # Concatenate the df_to_append with the existing DataFrame
            combined_df = pd.concat([existing_df, df_to_append], ignore_index=True)
        else:
            # If the CSV file doesn't exist, use the df_to_append as is
            combined_df = df_to_append

        # Write the combined DataFrame back to the CSV file
        combined_df.to_csv(existing_csv, index=False)


if __name__ == "__main__":
    # Making sure the binary pattern generator is generating the right number of starting zones.
    pe = ProblemEvaluator([])
    print(len(pe.generate_binary_patterns(10)))
