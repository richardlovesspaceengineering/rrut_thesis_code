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
    def __init__(self, instance, instance_name, mode, results_dir):
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
        self.results_dir = results_dir
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

        # Evaluate each neighbourhood - remember that these are stored in 3D arrays.
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
        pop_walk.evaluate(
            self.rescale_pregen_sample(walk, problem), eval_fronts=eval_fronts
        )

        return pop_walk, pop_neighbours_list

    def generate_global_population(self, problem, global_sample, eval_fronts=True):
        pop_global = Population(problem, n_individuals=global_sample.shape[0])

        pop_global.evaluate(
            self.rescale_pregen_sample(global_sample, problem), eval_fronts=eval_fronts
        )

        return pop_global

    def compute_global_normalisation_values(self, pre_sampler, problem):
        print(
            "Initialising normalisation computations for global samples. This requires full evaluation of the entire sample set and may take some time while still being memory-efficient."
        )
        norm_start = time.time()
        variables = ["var", "obj", "cv"]

        # Arrays to store max and min values from each sample
        max_values_array = {
            "var": np.empty((0, problem.n_var)),
            "obj": np.empty((0, problem.n_obj)),
            "cv": np.empty((0, 1)),
        }
        min_values_array = max_values_array

        # Loop over each of the samples.
        for i in range(pre_sampler.num_samples):
            # Loop over each of the walks within this sample - note that each sample contains n independent RWs.
            sample_start_time = time.time()
            print(
                "\nEvaluating populations for global sample {} out of {}...".format(
                    i + 1, pre_sampler.num_samples
                )
            )

            # Load the pre-generated sample.
            global_sample = pre_sampler.read_global_sample(i + 1)

            # Create population and evaluate.
            pop_global = self.generate_global_population(
                problem, global_sample, eval_fronts=True
            )

            # Loop over each variable.
            for which_variable in variables:
                combined_array = combine_arrays_for_pops([pop_global], which_variable)

                fmin, fmax = self.compute_maxmin_for_sample(
                    combined_array, pop_global.extract_pf(), which_variable
                )

                # Append min and max values for this variable
                min_values_array[which_variable] = np.vstack(
                    (min_values_array[which_variable], fmin)
                )
                max_values_array[which_variable] = np.vstack(
                    (max_values_array[which_variable], fmax)
                )

            sample_end_time = time.time()
            elapsed_sample_time = sample_end_time - sample_start_time
            print(
                "Evaluated populations in sample {} out of {} in {:.2f} seconds.".format(
                    i + 1, pre_sampler.num_samples, elapsed_sample_time
                )
            )

        # Once all samples have been evaluated we can compute normalisation values.
        normalisation_values = self.compute_norm_values_from_maxmin_arrays(
            min_values_array,
            max_values_array,
        )
        norm_end = time.time()
        elapsed_time = norm_end - norm_start
        print(
            "Evaluated the normalisation values for this sample set in {:.2f} seconds.".format(
                elapsed_time
            )
        )
        return normalisation_values

    def compute_rw_normalisation_values(self, pre_sampler, problem):
        print(
            "Initialising normalisation computations for RW samples. This requires full evaluation of the entire sample set and may take some time while still being memory-efficient."
        )
        norm_start = time.time()
        variables = ["var", "obj", "cv"]

        # Arrays to store max and min values from each sample
        max_values_array = {
            "var": np.empty((0, problem.n_var)),
            "obj": np.empty((0, problem.n_obj)),
            "cv": np.empty((0, 1)),
        }
        min_values_array = max_values_array

        # Loop over each of the samples.
        for i in range(pre_sampler.num_samples):
            # Loop over each of the walks within this sample - note that each sample contains n independent RWs.
            sample_start_time = time.time()
            print(
                "\nEvaluating populations for RW sample {} out of {}...".format(
                    i + 1, pre_sampler.num_samples
                )
            )
            for j in range(pre_sampler.dim):
                # Load the pre-generated sample.
                walk, neighbours = pre_sampler.read_walk_neighbours(i + 1, j + 1)

                # Create population and evaluate.
                start_time = time.time()
                (
                    pop_walk,
                    pop_neighbours_list,
                ) = self.generate_rw_neighbours_populations(
                    problem, walk, neighbours, eval_fronts=True
                )
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(
                    "Evaluated population {} of {} in {:.2f} seconds.".format(
                        j + 1, pre_sampler.dim, elapsed_time
                    )
                )

                # Loop over each variable.
                for which_variable in variables:
                    combined_array = combine_arrays_for_pops(
                        [pop_walk] + pop_neighbours_list, which_variable
                    )

                    fmin, fmax = self.compute_maxmin_for_sample(
                        combined_array, pop_walk.extract_pf(), which_variable
                    )

                    # Append min and max values for this variable
                    min_values_array[which_variable] = np.vstack(
                        (min_values_array[which_variable], fmin)
                    )
                    max_values_array[which_variable] = np.vstack(
                        (max_values_array[which_variable], fmax)
                    )

            sample_end_time = time.time()
            elapsed_sample_time = sample_end_time - sample_start_time
            print(
                "Evaluated populations in sample {} out of {} in {:.2f} seconds.".format(
                    i + 1, pre_sampler.num_samples, elapsed_sample_time
                )
            )

        # Once all samples have been evaluated we can compute normalisation values.
        normalisation_values = self.compute_norm_values_from_maxmin_arrays(
            min_values_array,
            max_values_array,
        )
        norm_end = time.time()
        elapsed_time = norm_end - norm_start
        print(
            "Evaluated the normalisation values for this sample set in {:.2f} seconds.".format(
                elapsed_time
            )
        )
        return normalisation_values

    def compute_maxmin_for_sample(self, combined_array, PF, which_variable):
        # Deal with nans here to ensure no nans are returned.
        combined_array = combined_array[~np.isnan(combined_array).any(axis=1)]

        # Find the min and max of each column.
        fmin = np.min(combined_array, axis=0)
        fmax = np.max(combined_array, axis=0)

        # Also consider the PF in the objectives case.
        if which_variable == "obj":
            fmin = np.minimum(fmin, np.min(PF, axis=0))
            fmax = np.maximum(fmax, np.max(PF, axis=0))
        elif which_variable == "cv":
            fmin = 0  # only dilate CV values.

        return fmin, fmax

    def compute_norm_values_from_maxmin_arrays(
        self,
        min_values_array,
        max_values_array,
    ):
        variables = ["var", "obj", "cv"]
        normalisation_values = {}

        # Calculate the final min, max, and 95th percentile values.
        for which_variable in variables:
            min_values = np.min(min_values_array[which_variable], axis=0)
            max_values = np.max(max_values_array[which_variable], axis=0)
            perc95_values = np.percentile(max_values_array[which_variable], 95, axis=0)

            # Store the computed values in the dictionary
            normalisation_values[f"{which_variable}_min"] = min_values
            normalisation_values[f"{which_variable}_max"] = max_values
            normalisation_values[f"{which_variable}_95th"] = perc95_values

        return normalisation_values

    def do_random_walk_analysis(self, problem, pre_sampler, num_samples):
        self.walk_normalisation_values = self.compute_rw_normalisation_values(
            pre_sampler,
            problem,
        )

        rw_multiple_samples_analyses_list = []

        # Loop over each of the samples.
        for i in range(num_samples):
            # Initialise list of analyses for this sample.
            rw_single_sample_analyses_list = []

            # Loop over each of the walks within this sample - note that each sample contains n independent RWs.
            sample_start_time = time.time()
            print(
                "\nEvaluating features for RW sample {} out of {}...".format(
                    i + 1, num_samples
                )
            )
            for j in range(pre_sampler.dim):
                # Initialise RandomWalkAnalysis evaluator. Do at every iteration or existing list entries get overwritten.
                rw_analysis = RandomWalkAnalysis(
                    self.walk_normalisation_values, self.results_dir
                )

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
                rw_single_sample_analyses_list.append(rw_analysis)

            # Concatenate analyses, generate feature arrays.
            rw_single_sample = Analysis.concatenate_single_analyses(
                rw_single_sample_analyses_list
            )
            rw_multiple_samples_analyses_list.append(rw_single_sample)

            # Print to log for each sample.
            sample_end_time = time.time()
            elapsed_time = sample_end_time - sample_start_time
            print(
                "Completed RW sample {} out of {} in {:.2f} seconds.\n".format(
                    i + 1, num_samples, elapsed_time
                )
            )

        # Concatenate analyses across samples.
        rw_analysis_all_samples = MultipleAnalysis(
            rw_multiple_samples_analyses_list, self.walk_normalisation_values
        )
        rw_analysis_all_samples.generate_feature_arrays()

        return rw_analysis_all_samples

    def do_global_analysis(self, problem, pre_sampler, num_samples):
        self.global_normalisation_values = self.compute_global_normalisation_values(
            pre_sampler, problem
        )

        # Initialise list of analyses for this sample.
        global_single_sample_analyses = []

        # Loop over each of the samples.
        for i in range(num_samples):
            # Initialise GlobalAnalysis evaluator. Do at each iteration to avoid overwriting existing list entries.
            global_analysis = GlobalAnalysis(
                self.global_normalisation_values, self.results_dir
            )

            sample_start_time = time.time()
            print(
                "\nEvaluating features for global sample {} out of {}...".format(
                    i + 1, num_samples
                )
            )
            # Load the pre-generated sample.
            global_sample = pre_sampler.read_global_sample(i + 1)

            # Create population and evaluate.
            start_time = time.time()
            pop_global = self.generate_global_population(problem, global_sample)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print("Evaluated population in {:.2f} seconds.".format(elapsed_time))

            # Pass to Analysis class for evaluation.
            start_time = time.time()
            global_analysis.eval_features(pop_global)
            end_time = time.time()  # Record the end time
            elapsed_time = end_time - start_time

            # Print to log for each walk within the sample.
            print(
                "Evaluated features in {:.2f} seconds.".format(
                    +1, pre_sampler.dim, elapsed_time
                )
            )
            global_single_sample_analyses.append(global_analysis)

            # Print to log for each sample.
            sample_end_time = time.time()
            elapsed_time = sample_end_time - sample_start_time
            print(
                "Completed global sample {} out of {} in {:.2f} seconds.\n".format(
                    i + 1, num_samples, elapsed_time
                )
            )

        # Concatenate analyses, generate feature arrays and return.
        global_multiple_analysis = MultipleAnalysis(
            global_single_sample_analyses, self.global_normalisation_values
        )
        global_multiple_analysis.generate_feature_arrays()

        return global_multiple_analysis

    def do_adaptive_walk_analysis(self, problem, pre_sampler, num_samples):
        # Initialise AW generator object before any loops.
        n_var = pre_sampler.dim

        if self.mode == "eval":
            # Experimental setup of Alsouly
            neighbourhood_size = 2 * n_var + 1
            max_steps = 1000
            step_size = 0.01  # 1% of the range of the instance domain

        elif self.mode == "debug":
            # Runs quickly
            neighbourhood_size = 2 * n_var + 1
            max_steps = 100
            step_size = 0.01  # 1% of the range of the instance domain

        # Now initialise adaptive walk generator.
        awGenerator = AdaptiveWalk(
            n_var,
            max_steps,
            step_size,
            neighbourhood_size,
            problem,
        )

        # List of adaptive walk analyses.
        aw_multiple_samples_analyses_list = []

        # Loop over each of the samples.
        for i in range(num_samples):
            # Initialise list of analyses for this sample.
            aw_single_sample_analyses_list = []

            # Loop over each of the walks within this sample - note that each sample contains n independent RWs.
            sample_start_time = time.time()
            print(
                "\nEvaluating features for AW sample {} out of {}...".format(
                    i + 1, num_samples
                )
            )

            # Load in the pre-generated LHS sample as a starting point.
            distributed_sample = self.rescale_pregen_sample(
                pre_sampler.read_global_sample(i + 1), problem
            )

            # Generate an adaptive walk for each point in the sample.
            # TODO: decide if this is the best way to do this.
            for j in range(distributed_sample.shape[0]):
                # Initialise AdaptiveWalkAnalysis evaluator. Do at every iteration or existing list entries get overwritten.
                aw_analysis = AdaptiveWalkAnalysis(
                    self.global_normalisation_values, self.results_dir
                )

                # Generate the adaptive walk sample.
                walk = awGenerator.do_adaptive_phc_walk_for_starting_point(
                    distributed_sample[j, :], constrained_ranks=True
                )  # TODO: add both constrained and unconstrained.
                neighbours = awGenerator.generate_neighbours_for_walk(walk)
                print(
                    "Generated AW {} of {} (for this sample). Length: {}".format(
                        j + 1,
                        distributed_sample.shape[0],
                        walk.shape[0],
                    )
                )

                # Create population and evaluate.
                start_time = time.time()
                pop_walk, pop_neighbours_list = self.generate_rw_neighbours_populations(
                    problem, walk, neighbours
                )
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(
                    "Evaluated population {} of {} in {:.2f} seconds.".format(
                        j + 1, distributed_sample.shape[0], elapsed_time
                    )
                )

                # Pass to Analysis class for evaluation.
                start_time = time.time()
                aw_analysis.eval_features(pop_walk, pop_neighbours_list)
                end_time = time.time()  # Record the end time
                elapsed_time = end_time - start_time

                # Print to log for each walk within the sample.
                print(
                    "Evaluated AW features for walk {} out of {} in {:.2f} seconds.\n".format(
                        j + 1, pre_sampler.dim, elapsed_time
                    )
                )
                aw_single_sample_analyses_list.append(aw_analysis)

            # Concatenate analyses, generate feature arrays.
            aw_single_sample = Analysis.concatenate_single_analyses(
                aw_single_sample_analyses_list
            )
            aw_multiple_samples_analyses_list.append(aw_single_sample)

            # Print to log for each sample.
            sample_end_time = time.time()
            elapsed_time = sample_end_time - sample_start_time
            print(
                "Completed AW sample {} out of {} in {:.2f} seconds.\n".format(
                    i + 1, num_samples, elapsed_time
                )
            )

        # Concatenate analyses across samples.
        aw_analysis_all_samples = MultipleAnalysis(
            aw_multiple_samples_analyses_list, self.walk_normalisation_values
        )
        aw_analysis_all_samples.generate_feature_arrays()

        return aw_analysis_all_samples

    def do(self, num_samples, save_arrays):
        print(
            "\n------------------------ Evaluating instance: "
            + self.instance_name
            + " ------------------------"
        )

        # Load presampler.
        pre_sampler = PreSampler(self.instance.n_var, num_samples, self.mode)

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
        global_features = self.do_global_analysis(
            self.instance, pre_sampler, num_samples
        )
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
            pre_sampler,
            num_samples,
        )
        aw_features.export_unaggregated_features(self.instance_name, "aw", save_arrays)

        # Overall landscape analysis - putting it all together.
        landscape = LandscapeAnalysis(global_features, rw_features, aw_features)

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
