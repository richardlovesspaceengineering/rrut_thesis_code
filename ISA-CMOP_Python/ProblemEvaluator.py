# Other package imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import socket
import pickle
import textwrap

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

import multiprocessing
import signal
from functools import wraps
from itertools import repeat

import smtplib

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def handle_ctrl_c(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global ctrl_c_entered
        if not ctrl_c_entered:
            signal.signal(signal.SIGINT, default_sigint_handler)  # the default
            try:
                return func(*args, **kwargs)
            except KeyboardInterrupt:
                ctrl_c_entered = True
                return KeyboardInterrupt()
            finally:
                signal.signal(signal.SIGINT, pool_ctrl_c_handler)
        else:
            return KeyboardInterrupt()

    return wrapper


def pool_ctrl_c_handler(*args, **kwargs):
    global ctrl_c_entered
    ctrl_c_entered = True


def init_pool():
    # set global variable for each process in the pool:
    global ctrl_c_entered
    global default_sigint_handler
    ctrl_c_entered = False
    default_sigint_handler = signal.signal(signal.SIGINT, pool_ctrl_c_handler)


class ProblemEvaluator:
    def __init__(
        self, instance, instance_name, mode, results_dir, num_cores, regen_problems
    ):
        """
        Possible modes are eval and debug.
        """
        self.instance = instance
        self.instance_name = instance_name
        self.features_table = pd.DataFrame()
        self.mode = mode
        self.walk_normalisation_values = {}
        self.global_normalisation_values = {}

        # flag for re-evaluating populations or using pre-saved values.
        self.results_dir = results_dir
        self.csv_filename = results_dir + "/features.csv"
        self.num_cores_user_input = num_cores
        print("Initialising evaluator in {} mode.".format(self.mode))

        # Gmail account credentials for email updates.
        self.gmail_user = "rrutthesisupdates@gmail.com"  # Your full Gmail address
        self.gmail_app_password = "binsjoipgwzyszxe "  # Your generated App Password

        # Should we re-evaluate pops again?
        self.rw_reeval = regen_problems
        self.global_reeval = regen_problems

    def send_update_email(self, subject, body=""):
        # Only send in eval mode.
        if self.mode == "eval":

            # Email details
            sent_from = self.gmail_user
            to = [self.gmail_user]  # Sending to the same account or specify a recipient

            # Setup MIME
            message = MIMEMultipart()
            message["From"] = sent_from
            message["To"] = ", ".join(to)
            message["Subject"] = (
                self.results_dir.replace("instance_results/", "") + " | " + subject
            )

            # Attach the email body
            message.attach(MIMEText(body, "plain"))

            # Convert the message to a string
            email_text = message.as_string()

            try:
                # Connect to Gmail's SMTP server
                server = smtplib.SMTP_SSL("smtp.gmail.com", 465)  # SSL port 465
                server.login(self.gmail_user, self.gmail_app_password)
                server.sendmail(sent_from, to, email_text)
                server.close()

            except Exception as e:
                print(f"Failed to send email: {e}")

    def send_analysis_completion_email(self, problem_name, analysis_type, elapsed_time):

        self.send_update_email(
            f"{problem_name} finished {analysis_type} in {round(elapsed_time,2)} seconds",
            f"The analysis of {problem_name} ({analysis_type}) has completed in {round(elapsed_time,2)} seconds.",
        )

    def initialize_number_of_cores(self, num_cores, num_samples):
        # Number of cores to use for RW.
        self.num_processes_aw = min(num_cores, num_samples)

        # Dictionary mapping dimensions to the number of processes. used only for global eval currently.
        # 15,000 points uses about 4 GB memory per process.
        hostname = socket.gethostname()
        if hostname == "RichardPC":
            self.num_processes_rw_dict = {
                "15d": 5,
                "20d": 5,
                "30d": 5,
            }
            self.num_processes_glob_dict = self.num_processes_rw_dict
        else:
            # Megatrons. Assumed available RAM of 128 GB.
            self.num_processes_rw_dict = {
                "15d": 10,
                "20d": 10,
                "30d": 6,
            }

            self.num_processes_glob_dict = {
                "15d": 30,
                "20d": 15,
                "30d": 10,
            }

        # Now we will allocate num_cores_global. This value will need to be smaller to deal with memory issues related to large matrices.
        dim_key = f"{self.instance.n_var}d"  # Assuming self.dim is an integer or string that matches the keys in the dictionary

        # Check if the current dimension has a specified number of processes
        if dim_key in self.num_processes_glob_dict:
            # Update num_processes based on the dictionary entry
            self.num_processes_global = self.num_processes_glob_dict[dim_key]
            self.num_processes_rw = self.num_processes_rw_dict[dim_key]
        else:
            self.num_processes_rw = min(num_cores, num_samples)

        self.num_processes_aw = self.num_processes_global

    def send_initialisation_email(self, header):
        # Summarize the core allocation
        cores_summary = textwrap.dedent(
            f"""
        Summary of cores allocation (have taken the minimum of num_samples and num_cores except for larger-dimension global cases):
        RW processes will use {self.num_processes_rw} cores.
        AW processes will use {self.num_processes_aw} cores.
        Global processes will use {self.num_processes_global} cores.
        """
        )

        # Add the reevaluation state
        reeval_summary = f"RW reevaluation state: {'Enabled' if self.rw_reeval else 'Disabled'}.\nGlobal reevaluation state: {'Enabled' if self.global_reeval else 'Disabled'}."

        # Combine the summaries
        full_summary = cores_summary + "\n" + reeval_summary

        print(full_summary)

        self.send_update_email(header, body=full_summary)

    def create_pre_sampler(self, num_samples):
        return PreSampler(self.instance.n_var, num_samples, self.mode)

    def initialise_pf(self, problem):
        # Pymoo requires creation of a population to initialise PF.
        print_with_timestamp("\nCreating a single population to initialise PF.")
        pop = Population(problem, n_individuals=1)

    def get_bounds(self, problem):
        # Bounds of the decision variables.
        x_lower = problem.xl
        x_upper = problem.xu
        return x_lower, x_upper

    def rescale_pregen_sample(self, x, problem):
        # TODO: check the rescaling is working correctly.
        x_lower, x_upper = self.get_bounds(problem)
        return x * (x_upper - x_lower) + x_lower

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

    def generate_walk_neig_populations(
        self,
        problem,
        walk,
        neighbours,
        eval_fronts=True,
        adaptive_walk=False,
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
            if not adaptive_walk:
                pop_neighbourhood.evaluate(
                    self.rescale_pregen_sample(neighbourhood, problem),
                    eval_fronts=eval_fronts,
                )
            else:
                # Adaptive walks are already rescaled.
                pop_neighbourhood.evaluate(
                    neighbourhood,
                    eval_fronts=eval_fronts,
                )
            pop_neighbours_list.append(pop_neighbourhood)

        # Never need to know the ranks of the walk steps relative to each other.

        if not adaptive_walk:
            pop_walk.evaluate(
                self.rescale_pregen_sample(walk, problem), eval_fronts=eval_fronts
            )
        else:
            pop_walk.evaluate(walk, eval_fronts=eval_fronts)

        return pop_walk, pop_neighbours_list

    def generate_global_population(self, problem, global_sample, eval_fronts=True):
        pop_global = Population(problem, n_individuals=global_sample.shape[0])

        pop_global.evaluate(
            self.rescale_pregen_sample(global_sample, problem), eval_fronts=eval_fronts
        )

        return pop_global

    def get_global_pop(self, pre_sampler, problem, sample_number):

        # Flag for whether we should manually generate new populations.
        continue_generation = True

        if not self.global_reeval:
            try:
                # Attempting to use pre-evaluated populations.
                pop_global = pre_sampler.load_global_population(sample_number)
                # If loading is successful, no need to generate a new population.
                continue_generation = False
            except FileNotFoundError:
                print(
                    f"Global population for sample {sample_number} not found. Generating new population."
                )

        if continue_generation:
            global_sample = pre_sampler.read_global_sample(sample_number)
            pop_global = self.generate_global_population(
                problem, global_sample, eval_fronts=True
            )
            pre_sampler.save_global_population(pop_global, sample_number)

        return pop_global

    @handle_ctrl_c
    def eval_single_sample_global_features_norm(self, args):
        i, pre_sampler, problem = args
        variables = ["var", "obj", "cv"]
        max_values_array = {
            "var": np.empty((0, problem.n_var)),
            "obj": np.empty((0, problem.n_obj)),
            "cv": np.empty((0, 1)),
        }
        min_values_array = max_values_array

        pop_global = self.get_global_pop(pre_sampler, problem, i + 1)

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
        return min_values_array, max_values_array

    def compute_global_normalisation_values(self, pre_sampler, problem):
        print_with_timestamp(
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

        # Can use max amount of cores here since NDSorting does not happen here.
        with multiprocessing.Pool(
            self.num_processes_global, initializer=init_pool
        ) as pool:
            args_list = [
                (i, pre_sampler, problem) for i in range(pre_sampler.num_samples)
            ]

            results = pool.map(self.eval_single_sample_global_features_norm, args_list)
            if any(map(lambda x: isinstance(x, KeyboardInterrupt), results)):
                print("Ctrl-C was entered.")

            for min_values, max_values in results:
                for var in variables:
                    min_values_array[var] = np.vstack(
                        (min_values_array[var], min_values[var])
                    )
                    max_values_array[var] = np.vstack(
                        (max_values_array[var], max_values[var])
                    )

            normalisation_values = self.compute_norm_values_from_maxmin_arrays(
                min_values_array,
                max_values_array,
            )
            norm_end = time.time()
            elapsed_time = norm_end - norm_start
            print_with_timestamp(
                "Evaluated the normalisation values for this sample set in {:.2f} seconds.\n".format(
                    elapsed_time
                )
            )

        self.send_analysis_completion_email(
            self.instance_name, "Global norm.", elapsed_time
        )
        return normalisation_values

    def get_rw_pop(self, pre_sampler, problem, sample_number, walk_number):

        # Flag for whether we should manually generate new populations.
        continue_generation = True

        if not self.rw_reeval:
            try:
                # Attempt to use pre-generated samples.
                pop_walk, pop_neighbours_list = pre_sampler.load_walk_neig_population(
                    sample_number, walk_number
                )
                # If loading is successful, skip the generation and saving process.
                continue_generation = False
            except FileNotFoundError:
                print(
                    f"Generating RW population for sample {sample_number}, walk {walk_number} as it was not found."
                )

        if continue_generation:
            walk, neighbours = pre_sampler.read_walk_neighbours(
                sample_number, walk_number
            )
            pop_walk, pop_neighbours_list = self.generate_walk_neig_populations(
                problem, walk, neighbours, eval_fronts=True
            )
            pre_sampler.save_walk_neig_population(
                pop_walk, pop_neighbours_list, sample_number, walk_number
            )

        return pop_walk, pop_neighbours_list

    @handle_ctrl_c
    def process_rw_sample_norm(self, args):
        variables = ["var", "obj", "cv"]

        i, pre_sampler, problem = args
        max_values_array = {
            "var": np.empty((0, problem.n_var)),
            "obj": np.empty((0, problem.n_obj)),
            "cv": np.empty((0, 1)),
        }
        min_values_array = max_values_array

        for j in range(pre_sampler.dim):

            pop_walk, pop_neighbours_list = self.get_rw_pop(
                pre_sampler, problem, i + 1, j + 1
            )

            for which_variable in variables:
                combined_array = combine_arrays_for_pops(
                    [pop_walk] + pop_neighbours_list, which_variable
                )
                fmin, fmax = self.compute_maxmin_for_sample(
                    combined_array, pop_walk.extract_pf(), which_variable
                )

                min_values_array[which_variable] = np.vstack(
                    (min_values_array[which_variable], fmin)
                )
                max_values_array[which_variable] = np.vstack(
                    (max_values_array[which_variable], fmax)
                )

        return min_values_array, max_values_array

    def compute_rw_normalisation_values(self, pre_sampler, problem):
        print_with_timestamp(
            "Initialising normalisation computations for RW samples. This requires full evaluation of the entire sample set and may take some time while still being memory-efficient."
        )
        norm_start = time.time()
        variables = ["var", "obj", "cv"]

        max_values_array = {
            "var": np.empty((0, problem.n_var)),
            "obj": np.empty((0, problem.n_obj)),
            "cv": np.empty((0, 1)),
        }
        min_values_array = max_values_array

        with multiprocessing.Pool(self.num_processes_rw, initializer=init_pool) as pool:
            args_list = [
                (i, pre_sampler, problem) for i in range(pre_sampler.num_samples)
            ]

            results = pool.map(self.process_rw_sample_norm, args_list)
            if any(map(lambda x: isinstance(x, KeyboardInterrupt), results)):
                print("Ctrl-C was entered.")

            for min_values, max_values in results:
                for var in variables:
                    min_values_array[var] = np.vstack(
                        (min_values_array[var], min_values[var])
                    )
                    max_values_array[var] = np.vstack(
                        (max_values_array[var], max_values[var])
                    )

            normalisation_values = self.compute_norm_values_from_maxmin_arrays(
                min_values_array,
                max_values_array,
            )
            norm_end = time.time()
            elapsed_time = norm_end - norm_start
            print_with_timestamp(
                "Evaluated the normalisation values for this sample set in {:.2f} seconds.\n".format(
                    elapsed_time
                )
            )

        self.send_analysis_completion_email(
            self.instance_name, "RW norm.", elapsed_time
        )

        return normalisation_values

    @handle_ctrl_c
    def eval_single_sample_rw_features(self, i, pre_sampler, problem):
        print_with_timestamp("Initialising feature evaluation for RW samples.")

        rw_single_sample_analyses_list = []

        # Loop over each of the walks within this sample.
        for j in range(pre_sampler.dim):
            rw_analysis = RandomWalkAnalysis(
                self.walk_normalisation_values, self.results_dir
            )

            # We already evaluated the populations when we computed the norms.
            pop_walk, pop_neighbours_list = pre_sampler.load_walk_neig_population(
                i + 1, j + 1
            )
            rw_analysis.eval_features(pop_walk, pop_neighbours_list)

            rw_single_sample_analyses_list.append(rw_analysis)

        return rw_single_sample_analyses_list

    def do_random_walk_analysis(self, problem, pre_sampler, num_samples):
        self.walk_normalisation_values = self.compute_rw_normalisation_values(
            pre_sampler,
            problem,
        )

        rw_multiple_samples_analyses_list = []

        start_time = time.time()

        with multiprocessing.Pool(self.num_processes_rw, initializer=init_pool) as pool:
            # Use partial method here.
            print_with_timestamp(
                "\nRunning parallel computation for RW features with {} processes. \n".format(
                    self.num_processes_rw
                )
            )
            results = pool.starmap(
                self.eval_single_sample_rw_features,
                zip(range(num_samples), repeat(pre_sampler), repeat(problem)),
            )

            if any(map(lambda x: isinstance(x, KeyboardInterrupt), results)):
                print("Ctrl-C was entered.")

            for i, rw_single_sample_analyses_list in enumerate(results):
                rw_single_sample = Analysis.concatenate_single_analyses(
                    rw_single_sample_analyses_list
                )
                rw_multiple_samples_analyses_list.append(rw_single_sample)

        # Concatenate analyses across samples.
        rw_analysis_all_samples = MultipleAnalysis(
            rw_multiple_samples_analyses_list, self.walk_normalisation_values
        )
        rw_analysis_all_samples.generate_feature_arrays()

        end_time = time.time()
        print_with_timestamp(
            "Evaluated RW features in {:.2f} seconds.\n".format(end_time - start_time)
        )
        self.send_analysis_completion_email(
            self.instance_name, "RW features", end_time - start_time
        )

        return rw_analysis_all_samples

    @handle_ctrl_c
    def eval_single_sample_global_features(self, i, pre_sampler, problem):
        global_analysis = GlobalAnalysis(
            self.global_normalisation_values, self.results_dir
        )

        # We already evaluated the populations when we computed the norms.
        pop_global = pre_sampler.load_global_population(i + 1)

        # Pass to Analysis class for evaluation.
        global_analysis.eval_features(pop_global)

        return global_analysis

    def do_global_analysis(self, problem, pre_sampler, num_samples):
        self.global_normalisation_values = self.compute_global_normalisation_values(
            pre_sampler, problem
        )

        start_time = time.time()

        global_multiple_analyses_list = []

        with multiprocessing.Pool(
            self.num_processes_global, initializer=init_pool
        ) as pool:
            print_with_timestamp(
                "\nRunning parallel computation for global features with {} processes. \n".format(
                    self.num_processes_global
                )
            )

            # Use starmap with zip and repeat to pass the same pre_sampler and problem to each call
            results = pool.starmap(
                self.eval_single_sample_global_features,
                zip(range(num_samples), repeat(pre_sampler), repeat(problem)),
            )

            if any(map(lambda x: isinstance(x, KeyboardInterrupt), results)):
                print("Ctrl-C was entered.")

            for i, global_analysis in enumerate(results):
                global_multiple_analyses_list.append(global_analysis)

        # Concatenate analyses across samples.
        global_multiple_analysis = MultipleAnalysis(
            global_multiple_analyses_list, self.global_normalisation_values
        )
        global_multiple_analysis.generate_feature_arrays()

        end_time = time.time()
        print_with_timestamp(
            "Evaluated global features in {:.2f} seconds.\n".format(
                end_time - start_time
            )
        )
        self.send_analysis_completion_email(
            self.instance_name, "Global features", end_time - start_time
        )

        return global_multiple_analysis

    @handle_ctrl_c
    def eval_single_sample_aw_features(self, i, pre_sampler, problem, awGenerator):
        aw_single_sample_analyses_list = []

        # Loop over each of the walks within this sample.
        sample_start_time = time.time()
        print(
            "\nEvaluating features for AW sample {} out of {}...".format(
                i + 1, pre_sampler.num_samples
            )
        )

        # Load in the pre-generated LHS sample as a starting point.
        distributed_sample = self.rescale_pregen_sample(
            pre_sampler.read_global_sample(i + 1), problem
        )

        # Each sample contains 10n walks.
        num_walks = int(distributed_sample.shape[0] / 100)

        for j in range(num_walks):
            # Initialise AdaptiveWalkAnalysis evaluator. Do at every iteration or existing list entries get overwritten.
            aw_analysis = AdaptiveWalkAnalysis(
                self.global_normalisation_values, self.results_dir
            )

            # Generate the adaptive walk sample.
            walk = awGenerator.do_adaptive_phc_walk_for_starting_point(
                distributed_sample[j, :], constrained_ranks=True
            )
            neighbours = awGenerator.generate_neighbours_for_walk(walk)
            # _, neighbours = pre_sampler.read_walk_neighbours(i + 1, 10)
            print(
                "Generated AW {} of {} (for this sample). Length: {}".format(
                    j + 1,
                    num_walks,
                    walk.shape[0],
                )
            )

            # Create population and evaluate.
            start_time = time.time()
            pop_walk, pop_neighbours_list = self.generate_walk_neig_populations(
                problem, walk, neighbours, adaptive_walk=True
            )
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(
                "Evaluated population {} of {} in {:.2f} seconds.".format(
                    j + 1, num_walks, elapsed_time
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
                    j + 1, num_walks, elapsed_time
                )
            )
            aw_single_sample_analyses_list.append(aw_analysis)

        return aw_single_sample_analyses_list

    def do_adaptive_walk_analysis(self, problem, pre_sampler, num_samples):
        # Initialise AW generator object before any loops.
        n_var = pre_sampler.dim

        if self.mode == "eval":
            # Experimental setup of Alsouly
            neighbourhood_size = 2 * n_var + 1
            max_steps = 500
            step_size = 0.01  # 1% of the range of the instance domain

        elif self.mode == "debug":
            # Runs quickly
            neighbourhood_size = 2 * n_var + 1
            max_steps = 10
            step_size = 0.01  # 1% of the range of the instance domain

        # Now initialise adaptive walk generator.
        awGenerator = AdaptiveWalk(
            n_var,
            max_steps,
            step_size,
            neighbourhood_size,
            problem,
        )

        aw_multiple_samples_analyses_list = []

        start_time = time.time()

        with multiprocessing.Pool(self.num_processes_aw, initializer=init_pool) as pool:
            print_with_timestamp(
                "\nRunning parallel computation for AW features with {} processes. \n".format(
                    self.num_processes_aw
                )
            )
            results = pool.starmap(
                self.eval_single_sample_aw_features,
                zip(
                    range(num_samples),
                    repeat(pre_sampler),
                    repeat(problem),
                    repeat(awGenerator),
                ),
            )

            if any(map(lambda x: isinstance(x, KeyboardInterrupt), results)):
                print("Ctrl-C was entered.")

            for i, aw_single_sample_analyses_list in enumerate(results):
                aw_single_sample = Analysis.concatenate_single_analyses(
                    aw_single_sample_analyses_list
                )
                aw_multiple_samples_analyses_list.append(aw_single_sample)

        # Concatenate analyses across samples.
        aw_analysis_all_samples = MultipleAnalysis(
            aw_multiple_samples_analyses_list, self.walk_normalisation_values
        )
        aw_analysis_all_samples.generate_feature_arrays()

        end_time = time.time()
        print_with_timestamp(
            "Evaluated AW features in {:.2f} seconds.\n".format(end_time - start_time)
        )
        self.send_analysis_completion_email(
            self.instance_name, "AW features", end_time - start_time
        )

        return aw_analysis_all_samples

    def initialize_evaluator(self, num_samples):
        # Load presampler and create directories for populations.
        pre_sampler = self.create_pre_sampler(num_samples)
        pre_sampler.create_pregen_sample_dir()
        pre_sampler.create_pops_dir(self.instance_name)

        # Define number of cores for multiprocessing.
        self.initialize_number_of_cores(self.num_cores_user_input, num_samples)

        # Initialise PF text file.
        self.initialise_pf(self.instance)

        return pre_sampler

    def do(self, num_samples, save_arrays):
        print(
            "\n------------------------ Evaluating instance: "
            + self.instance_name
            + " ------------------------"
        )

        pre_sampler = self.initialize_evaluator(num_samples)

        self.send_initialisation_email(f"STARTED RUN OF {self.instance_name}.")

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

        print_with_timestamp(
            "Successfully appended aggregated results to csv file.\n\n"
        )

        self.send_update_email(f"COMPLETED RUN OF {self.instance_name}.")

    def run_populations(self, num_samples):
        """
        Evaluate populations for the problem first to cut down on CPU overhead later.
        """
        print(
            "\n------------------------ Evaluating instance (POPULATIONS ONLY): "
            + self.instance_name
            + " ------------------------"
        )

        pre_sampler = self.initialize_evaluator(num_samples)

        self.send_initialisation_email(f"STARTED POPS RUN OF {self.instance_name}.")

        # RW Analysis.
        print(
            " \n ~~~~~~~~~~~~ RW Populations for "
            + self.instance_name
            + " ~~~~~~~~~~~~ \n"
        )

        with multiprocessing.Pool(self.num_processes_rw, initializer=init_pool) as pool:
            args_list = [
                (i, pre_sampler, self.instance) for i in range(pre_sampler.num_samples)
            ]

            results = pool.map(self.process_rw_sample_norm, args_list)
            if any(map(lambda x: isinstance(x, KeyboardInterrupt), results)):
                print("Ctrl-C was entered.")

        # Global Analysis.
        print(
            " \n ~~~~~~~~~~~~ Global Populations for "
            + self.instance_name
            + " ~~~~~~~~~~~~ \n"
        )

        # Can use max amount of cores here since NDSorting does not happen here.
        with multiprocessing.Pool(self.num_processes_rw, initializer=init_pool) as pool:
            args_list = [
                (i, pre_sampler, self.instance) for i in range(pre_sampler.num_samples)
            ]

            results = pool.map(self.eval_single_sample_global_features_norm, args_list)
            if any(map(lambda x: isinstance(x, KeyboardInterrupt), results)):
                print("Ctrl-C was entered.")

        self.send_update_email(f"COMPLETED POPS RUN OF {self.instance_name}.")

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
