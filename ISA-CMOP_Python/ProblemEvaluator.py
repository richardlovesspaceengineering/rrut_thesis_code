# Other package imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

# User packages.
from features.feature_helpers import *
from features.GlobalAnalysis import *
from features.RandomWalkAnalysis import *
from features.AdaptiveWalkAnalysis import *
from features.LandscapeAnalysis import LandscapeAnalysis
from optimisation.operators.sampling.latin_hypercube_sampling import (
    LatinHypercubeSampling,
)
from optimisation.operators.sampling.random_sampling import RandomSampling
from PreSampler import PreSampler

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
        self.walk_normalisation_values = {}
        self.global_normalisation_values = {}
        self.rw_sample_dir = os.path.join(
            "pregen_samples",
            "rw",
        )
        print("Initialising evaluator in {} mode.".format(self.mode))

    def get_bounds(self, problem):
        # Bounds of the decision variables.
        x_lower = problem.xl
        x_upper = problem.xu
        return x_lower, x_upper

    def rescale_pregen_sample(self, x, problem):
        # TODO: check the rescaling is working correctly.
        x_lower, x_upper = self.get_bounds(problem)
        return x * (x_upper - x_lower) + x_lower

    def generate_rw_neighbours_populations(
        self, problem, walk, neighbours, eval_fronts=True
    ):
        # Generate populations for walk and neighbours separately.
        pop_walk = Population(problem, n_individuals=walk.shape[0])

        pop_neighbours_list = []

        # Evaluate each neighbourhood - remember that these are stored in 3D arrays i,j,k where i is the neighbour index within the neighbourhood, j is the step index, k is the walk index.
        for neighbourhood in neighbours:
            # None of the features related to neighbours ever require knowledge of the neighbours ranks relative to each other.
            pop_neighbourhood = Population(
                problem, n_individuals=neighbourhood.shape[0]
            )
            pop_neighbourhood.evaluate(
                self.rescale_pregen_sample(neighbourhood, problem),
                eval_fronts=eval_fronts,
            )
            pop_neighbours_list.append(pop_neighbourhood)

        # Never need to know the ranks of the walk steps relative to each other.
        pop_walk.evaluate(self.rescale_pregen_sample(walk, problem), eval_fronts=False)

        return pop_walk, pop_neighbours_list

    def compute_normalisation_values(self, pre_sampler, problem, sample_type):
        normalization_values = {}
        variables = ["var", "obj", "cv"]

        # Arrays to store max and min values from each sample
        max_values_array = {
            "var": np.empty((0, problem.n_var)),
            "obj": np.empty((0, problem.n_obj)),
            "cv": np.empty((0, 1)),
        }
        min_values_array = max_values_array

        print(
            "Initialising normalisation computations. This requires full evaluation of the entire sample set and may take some time while still being memory-efficient."
        )
        norm_start = time.time()

        # Loop over each of the samples.
        for i in range(pre_sampler.num_samples):
            # Loop over each of the walks within this sample - note that each sample contains n independent RWs.
            for j in range(pre_sampler.dim):
                if sample_type == "rw":
                    # Load the pre-generated sample.
                    walk, neighbours = pre_sampler.read_walk_neighbours(i + 1, j + 1)

                    # Create population and evaluate.
                    (
                        pop_walk,
                        pop_neighbours_list,
                    ) = self.generate_rw_neighbours_populations(
                        problem, walk, neighbours, eval_fronts=False
                    )

                # Loop over each variable.
                for which_variable in variables:
                    # Combine arrays for pops
                    if sample_type == "rw":
                        combined_array = combine_arrays_for_pops(
                            [pop_walk] + pop_neighbours_list, which_variable
                        )

                    # Deal with nans here to ensure no nans are returned.
                    combined_array = combined_array[
                        ~np.isnan(combined_array).any(axis=1)
                    ]

                    # Find the min and max of each column.
                    fmin = np.min(combined_array, axis=0)
                    fmax = np.max(combined_array, axis=0)

                    # Also consider the PF in the objectives case.
                    if which_variable == "obj":
                        PF = pop_walk.extract_pf()
                        fmin = np.minimum(fmin, np.min(PF, axis=0))
                        fmax = np.maximum(fmax, np.max(PF, axis=0))
                    elif which_variable == "cv":
                        fmin = 0  # only dilate CV values.

                    # Append min and max values for this variable
                    min_values_array[which_variable] = np.vstack(
                        (min_values_array[which_variable], fmin)
                    )
                    max_values_array[which_variable] = np.vstack(
                        (max_values_array[which_variable], fmax)
                    )

        # Calculate the final min, max, and 95th percentile values after the loop
        for which_variable in variables:
            min_values = np.min(min_values_array[which_variable], axis=0)
            max_values = np.max(max_values_array[which_variable], axis=0)
            perc95_values = np.percentile(max_values_array[which_variable], 95, axis=0)

            # Store the computed values in the dictionary
            normalization_values[f"{which_variable}_min"] = min_values
            normalization_values[f"{which_variable}_max"] = max_values
            normalization_values[f"{which_variable}_95th"] = perc95_values

        norm_end = time.time()
        elapsed_time = norm_end - norm_start
        print(
            "Evaluated the normalisation values for this sample set in {:.2f} seconds.".format(
                elapsed_time
            )
        )
        return normalization_values

    def do_random_walk_analysis(self, problem, pre_sampler, num_samples):
        self.walk_normalisation_values = self.compute_normalisation_values(
            pre_sampler, problem, "rw"
        )

        # Initialise RandomWalkAnalysis evaluator.
        rw_analysis = RandomWalkAnalysis(self.walk_normalisation_values)

        # Loop over each of the samples.
        for i in range(num_samples):
            # Initialise list of analyses for this sample.
            rw_single_sample_analyses = []

            # Loop over each of the walks within this sample - note that each sample contains n independent RWs.
            sample_start_time = time.time()
            for j in range(pre_sampler.dim):
                # Load the pre-generated sample.
                walk, neighbours = pre_sampler.read_walk_neighbours(i + 1, j + 1)

                # Create population and evaluate.
                start_time = time.time()
                pop_walk, pop_neighbours_list = self.generate_rw_neighbours_populations(
                    problem, walk, neighbours
                )
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(
                    "Evaluated population {} of {} in {:.2f} seconds.".format(
                        j + 1, pre_sampler.dim, elapsed_time
                    )
                )

                # Pass to Analysis class for evaluation.
                start_time = time.time()
                rw_analysis.eval_features(pop_walk, pop_neighbours_list)
                end_time = time.time()  # Record the end time
                elapsed_time = end_time - start_time

                # Print to log for each walk within the sample.
                print(
                    "Evaluated RW features for walk {} out of {} in {:.2f} seconds.".format(
                        j + 1, pre_sampler.dim, elapsed_time
                    )
                )
                rw_single_sample_analyses.append(rw_analysis)

            # Print to log for each sample.
            sample_end_time = time.time()
            elapsed_time = sample_end_time - sample_start_time
            print(
                "Evaluated features for sample {} out of {} in {:.2f} seconds.\n".format(
                    i + 1, num_samples, elapsed_time
                )
            )

    def sample_for_global_features(self, problem, num_samples, method="lhs.scipy"):
        n_var = problem.n_var

        distributed_samples = []

        if self.mode == "eval":
            # Experimental setup of Liefooghe2021.
            num_points = int(n_var * 200)
            iterations = num_points
        elif self.mode == "debug":
            # Runs quickly.
            num_points = int(n_var * 100)
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

    def evaluate_global_features(self, pops_global, normalisation_values):
        global_features = MultipleGlobalAnalysis(pops_global, normalisation_values)
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

        # Compute normalisation values for feature calculations later.
        # self.global_normalisation_values = compute_all_normalisation_values(pops_global)

        # Evaluate features.
        global_features = self.evaluate_global_features(
            pops_global, self.global_normalisation_values
        )

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
            distributed_sample_size = n_var * 20
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

    def evaluate_aw_features_for_one_sample(
        self, pops_adaptive_walks_neighbours, normalisation_values
    ):
        """
        Evaluate the RW features for one sample (i.e a set of random walks)
        """
        pops_walks = [t[0] for t in pops_adaptive_walks_neighbours]
        pops_neighbours_list = [t[1] for t in pops_adaptive_walks_neighbours]
        rw_features_single_sample = MultipleAdaptiveWalkAnalysis(
            pops_walks, pops_neighbours_list, normalisation_values
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

        # TODO: decide if it's best to use RW or Global normalisation values here.
        normalisation_values = self.walk_normalisation_values

        aw_features_list = []
        for ctr, pop_walks_neighbours in enumerate(
            pop_adaptive_walks_neighbours_all_samples
        ):
            # Finally, evaluate features.
            aw_features_list.append(
                self.evaluate_aw_features_for_one_sample(
                    pop_walks_neighbours, normalisation_values
                )
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

        # Load presampler.
        pre_sampler = PreSampler(self.instance.n_var, num_samples)

        # RW Analysis.
        print(
            " \n ~~~~~~~~~~~~ RW Analysis for "
            + self.instance_name
            + " ~~~~~~~~~~~~ \n"
        )
        rw_features = self.do_random_walk_analysis(
            self.instance,
            pre_sampler,
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
        aw_features.export_unaggregated_features(self.instance_name, "aw", save_arrays)

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
