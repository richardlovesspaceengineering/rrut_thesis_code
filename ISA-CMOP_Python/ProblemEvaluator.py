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
from features.ancillary_functions import *
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
from itertools import repeat
import matlab.engine

import smtplib

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from multiprocessing_util import *


class ProblemEvaluator:
    def __init__(
        self, instance, instance_name, mode, results_dir, num_samples, num_cores
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
        self.num_samples_user_input = num_samples
        print("Initialising evaluator in {} mode.".format(self.mode))

        # Gmail account credentials for email updates.
        self.gmail_user = "rrutthesisupdates@gmail.com"  # Your full Gmail address
        self.gmail_app_password = "binsjoipgwzyszxe "  # Your generated App Password

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

    def check_if_modact(self):
        instance_name_lower = self.instance_name.lower()
        return (
            instance_name_lower.startswith(("cs", "ct"))
            and "ctp" not in instance_name_lower
        )

    def check_if_aerofoil(self):
        instance_name_lower = self.instance_name.lower()
        return instance_name_lower.startswith("xa")

    def check_if_modact_or_aerofoil(self):
        return self.check_if_aerofoil() or self.check_if_modact()

    def check_if_lircmop(self):
        return "lircmop" in self.instance_name.lower()

    def check_if_platemo(self):
        return self.instance_name.lower().startswith(("cf", "sdc", "rwmop"))

    def initialize_number_of_samples(self):
        n_var = self.instance.n_var

        if self.check_if_modact_or_aerofoil() or self.check_if_platemo():
            self.num_samples = min(self.num_samples_user_input, 10)

            # Number of independent random walks within a single RW sample.
            if self.check_if_aerofoil() or n_var >= 10:
                self.num_walks_rw = int(n_var / 3)
                self.num_walks_aw = int(2 * n_var)
            else:
                self.num_walks_rw = int(n_var)
                self.num_walks_aw = int(10 * n_var)
        else:
            self.num_samples = self.num_samples_user_input
            self.num_walks_rw = self.instance.n_var

            if n_var <= 10:
                self.num_walks_aw = int(10 * n_var)
            else:
                self.num_walks_aw = int(5 * n_var)

        # Account for debug mode.
        if self.mode == "debug":
            self.num_walks_rw = 2
            self.num_walks_aw = 2

    def initialize_number_of_cores(self):
        # Number of cores to use for RW.
        self.num_processes_aw = min(self.num_cores_user_input, self.num_samples)

        # Dictionary mapping dimensions to the number of processes. used only for global eval currently.
        # 15,000 points uses about 4 GB memory per process.
        hostname = socket.gethostname()
        if hostname == "RichardPC":
            self.num_processes_rw_norm_dict = {
                "15d": 5,
                "20d": 5,
                "30d": 5,
            }
            self.num_processes_global_norm_dict = self.num_processes_rw_norm_dict
            self.num_processes_rw_eval_dict = self.num_processes_rw_norm_dict
            self.num_processes_global_eval_dict = self.num_processes_rw_norm_dict

            if self.check_if_aerofoil():
                self.num_processes_parallel_seed = 20  # max cores
            else:
                self.num_processes_parallel_seed = 10  # max cores
        else:
            # Megatrons. Assumed available RAM of 128 GB.
            self.num_processes_rw_norm_dict = {
                "15d": 30,
                "20d": 30,
                "30d": 30,
            }

            self.num_processes_rw_eval_dict = {
                "15d": 30,
                "20d": 30,
                "30d": 30,
            }

            self.num_processes_global_norm_dict = {
                "15d": 30,
                "20d": 30,
                "30d": 30,
            }

            self.num_processes_global_eval_dict = {
                "15d": 30,
                "20d": 30,
                "30d": 30,
            }

            if self.check_if_aerofoil():
                if "4" in self.instance_name:
                    # XA4 runs too slow.
                    self.num_processes_parallel_seed = 32
                else:
                    self.num_processes_parallel_seed = 64  # max cores
            else:
                # TODO: experiment with optimal value for ModAct
                self.num_processes_parallel_seed = 24  # max cores

        # Now we will allocate num_cores_global. This value will need to be smaller to deal with memory issues related to large matrices.
        dim_key = f"{self.instance.n_var}d"  # Assuming self.dim is an integer or string that matches the keys in the dictionary

        # Check if the current dimension has a specified number of processes. Just check one dictionary since they all have the same keys.
        if dim_key in self.num_processes_global_norm_dict:
            # Update num_processes based on the dictionary entry
            self.num_processes_global_norm = min(
                self.num_processes_global_norm_dict[dim_key], self.num_samples
            )
            self.num_processes_global_eval = min(
                self.num_processes_global_eval_dict[dim_key], self.num_samples
            )
            self.num_processes_rw_norm = min(
                self.num_processes_rw_norm_dict[dim_key], self.num_samples
            )
            self.num_processes_rw_eval = min(
                self.num_processes_rw_eval_dict[dim_key], self.num_samples
            )
        else:
            self.num_processes_global_norm = min(
                self.num_cores_user_input, self.num_samples
            )
            self.num_processes_global_eval = self.num_processes_global_norm
            self.num_processes_rw_norm = min(
                self.num_cores_user_input, self.num_samples
            )
            self.num_processes_rw_eval = self.num_processes_rw_norm

        self.num_processes_aw = self.num_processes_global_eval

    def send_initialisation_email(self, header, pre_sampler):
        # Summarize the core allocation
        cores_summary = textwrap.dedent(
            f"""
        Summary of sampling:
        - Dim: {self.instance.n_var}
        - Number of samples: {self.num_samples}
        - Number of independent RWs within each sample: {self.num_walks_rw}.
        - Number of independent AWs within each sample: {self.num_walks_aw}.
        - Neighbourhood size: {pre_sampler.neighbourhood_size_rw}.       
        
        Summary of cores allocation (have taken the minimum of self.num_samples and num_cores except for larger-dimension global cases):
        Parallel eval will use {self.num_processes_parallel_seed}.
        RW processes will use {self.num_processes_rw_norm} cores for normalisation, {self.num_processes_rw_eval} cores for evaluation.
        Global processes will use {self.num_processes_global_norm} cores for normalisation, {self.num_processes_global_eval} cores for evaluation.
        AW processes will use {self.num_processes_aw} cores.
        """
        )

        # Combine the summaries
        full_summary = cores_summary

        print(full_summary)

        self.send_update_email(header, body=full_summary)

    def create_pre_sampler(self):
        is_aerofoil = self.check_if_aerofoil()
        return PreSampler(self.instance.n_var, self.num_samples, self.mode, is_aerofoil)

    def initialise_pf(self, problem):
        # Pymoo requires creation of a population to initialise PF.
        print_with_timestamp("\nCreating a single population to initialise PF.")
        pop = Population(problem, n_individuals=1)

    def get_bounds(self, problem):
        return problem.xl, problem.xu

    def rescale_pregen_sample(self, x, problem):
        # TODO: check the rescaling is working correctly.
        x_lower, x_upper = self.get_bounds(problem)
        return x * (x_upper - x_lower) + x_lower

    def compute_maxmin_for_sample(self, combined_array, PF, which_variable):

        # Initialise fmin and fmax.
        fmin = np.zeros(combined_array.shape[1])
        fmax = np.ones(combined_array.shape[1])

        # Deal with nans here to ensure no nans are returned.
        combined_array = combined_array[~np.isnan(combined_array).any(axis=1)]

        # Check if combined_array is empty after removing NaNs.
        if combined_array.size == 0:
            # Return dummy min and max values if combined_array is empty.
            return fmin, fmax

        # Find the min and max of each column.
        fmin = np.min(combined_array, axis=0)
        fmax = np.max(combined_array, axis=0)

        # Also consider the PF in the objectives case.
        if which_variable == "obj":
            if PF:
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

    def generate_neig_populations(
        self,
        problem,
        neighbours,
        eval_fronts=False,
        num_processes=1,
        adaptive_walk=False,
    ):

        # Squash the list of neighbourhoods into one large numpy array to speed up evaluation.
        all_neighbours = np.concatenate(neighbours, axis=0)

        # Generate one large population for all neighbours combined
        pop_total = Population(problem, n_individuals=all_neighbours.shape[0])

        print_with_timestamp("Evaluating neighbours...")

        if not adaptive_walk:
            pop_total.evaluate(
                self.rescale_pregen_sample(all_neighbours, problem),
                eval_fronts=False,  # no need to compare all neighbours to all steps
                num_processes=num_processes,
                show_msg=True,
            )
        else:
            # Adaptive walks are already rescaled
            pop_total.evaluate(
                all_neighbours,
                eval_fronts=eval_fronts,
                num_processes=num_processes,
                show_msg=True,
            )

        # Now, split the population back into the respective neighbourhoods
        pop_neighbours_list = []
        start_idx = 0
        for neighbourhood in neighbours:
            end_idx = start_idx + neighbourhood.shape[0]
            pop_neighbourhood = pop_total[start_idx:end_idx]

            # Only evaluate fronts within neighbourhoods
            if eval_fronts:
                pop_neighbourhood.evaluate_fronts()
            pop_neighbours_list.append(pop_neighbourhood)
            start_idx = end_idx

        return pop_neighbours_list

    def generate_walk_neig_populations(
        self,
        problem,
        walk,
        neighbours,
        eval_fronts=True,
        eval_pops_parallel=False,
        adaptive_walk=False,
    ):

        if eval_pops_parallel and not self.check_if_platemo():
            num_processes = self.num_processes_parallel_seed
        else:
            num_processes = 1

        # Generate populations for walk and neighbours separately.
        pop_walk = Population(problem, n_individuals=walk.shape[0])

        pop_neighbours_list = self.generate_neig_populations(
            problem, neighbours, eval_fronts, num_processes=num_processes
        )

        print_with_timestamp("Evaluating walk...")

        if not adaptive_walk:
            pop_walk.evaluate(
                self.rescale_pregen_sample(walk, problem),
                eval_fronts=eval_fronts,
                num_processes=num_processes,
                show_msg=True,
            )
        else:
            pop_walk.evaluate(
                walk,
                eval_fronts=eval_fronts,
                num_processes=num_processes,
                show_msg=True,
            )

        return pop_walk, pop_neighbours_list

    def generate_global_population(
        self, problem, global_sample, eval_pops_parallel=False, eval_fronts=True
    ):

        if eval_pops_parallel and not self.check_if_platemo():
            num_processes = self.num_processes_parallel_seed
        else:
            num_processes = 1

        pop_global = Population(problem, n_individuals=global_sample.shape[0])

        pop_global.evaluate(
            self.rescale_pregen_sample(global_sample, problem),
            eval_fronts=eval_fronts,
            num_processes=num_processes,
        )

        return pop_global

    def get_global_pop(
        self,
        pre_sampler,
        problem,
        sample_number,
        eval_pops_parallel=False,
        eval_fronts=True,
    ):

        # Flag for whether we should manually generate new populations.
        continue_generation = True

        try:
            # Attempting to use pre-evaluated populations.
            pop_global = pre_sampler.load_global_population(problem, sample_number)

            # Evaluate fronts.
            if eval_fronts and not pop_global.is_ranks_evaluated():
                pop_global.evaluate_fronts(show_time=True)

                # Save again to save us having to re-evaluate the fronts.
                pre_sampler.save_global_population(pop_global, sample_number)

            # If loading is successful, no need to generate a new population.
            continue_generation = False
        except FileNotFoundError:
            print(
                f"Global population for sample {sample_number} not found. Generating new population."
            )

        if continue_generation:
            global_sample = pre_sampler.read_global_sample(sample_number)
            pop_global = self.generate_global_population(
                problem,
                global_sample,
                eval_fronts=eval_fronts,
                eval_pops_parallel=eval_pops_parallel,
            )
            pre_sampler.save_global_population(pop_global, sample_number)

        return pop_global

    @handle_ctrl_c
    def eval_single_sample_global_features_norm(self, args):
        i, pre_sampler, problem, eval_pops_parallel = args
        variables = ["var", "obj", "cv"]
        max_values_array = {
            "var": np.empty((0, problem.n_var)),
            "obj": np.empty((0, problem.n_obj)),
            "cv": np.empty((0, 1)),
        }
        min_values_array = max_values_array

        # We will wait until features evaluation to evaluate fronts if we are using parallel processing to evaluate seeds. This means we can run NDSort on num_samples seeds simultaneously.

        pop_global = self.get_global_pop(
            pre_sampler,
            problem,
            i + 1,
            eval_fronts=False,
            eval_pops_parallel=eval_pops_parallel,
        )
        pf = pop_global.extract_pf()

        # Loop over each variable.
        for which_variable in variables:
            combined_array = Analysis.combine_arrays_for_pops(
                [pop_global], which_variable
            )

            fmin, fmax = self.compute_maxmin_for_sample(
                combined_array, pf, which_variable
            )

            # Append min and max values for this variable
            min_values_array[which_variable] = np.vstack(
                (min_values_array[which_variable], fmin)
            )
            max_values_array[which_variable] = np.vstack(
                (max_values_array[which_variable], fmax)
            )
        return min_values_array, max_values_array

    def compute_global_normalisation_values(
        self, pre_sampler, problem, eval_pops_parallel=False
    ):
        norm_start = time.time()
        variables = ["var", "obj", "cv"]

        # Arrays to store max and min values from each sample
        max_values_array = {
            "var": np.empty((0, problem.n_var)),
            "obj": np.empty((0, problem.n_obj)),
            "cv": np.empty((0, 1)),
        }
        min_values_array = max_values_array

        print_with_timestamp(
            f"Initialising normalisation computations for global samples with {self.num_processes_global_norm} processes. This requires full evaluation of the entire sample set and may take some time while still being memory-efficient."
        )

        if not eval_pops_parallel:

            print("Running seeds in series.")

            # Can use max amount of cores here since NDSorting does not happen here.
            with multiprocessing.Pool(
                self.num_processes_global_norm, initializer=init_pool
            ) as pool:
                args_list = [
                    (i, pre_sampler, problem, False) for i in range(self.num_samples)
                ]

                results = pool.map(
                    self.eval_single_sample_global_features_norm, args_list
                )
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

        else:

            print("Running seeds in parallel.")

            init_pool()  # set some global variables for ctrl + c handling
            for i in range(self.num_samples):
                start_time = time.time()

                min_values, max_values = self.eval_single_sample_global_features_norm(
                    (i, pre_sampler, problem, True)
                )
                for var in variables:
                    min_values_array[var] = np.vstack(
                        (min_values_array[var], min_values[var])
                    )
                    max_values_array[var] = np.vstack(
                        (max_values_array[var], max_values[var])
                    )

                end_time = time.time()

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

    def evaluate_ranks_for_walk_pop(
        self, pre_sampler, sample_number, walk_number, pop_walk, pop_neighbours_list
    ):
        need_to_resave = False

        if not pop_walk.is_ranks_evaluated():
            pop_walk.evaluate_fronts(show_time=True)
            need_to_resave = True

        for (
            pop_neighbourhood
        ) in pop_neighbours_list:  # Only evaluate fronts within neighbourhoods
            if not pop_neighbourhood.is_ranks_evaluated():
                pop_neighbourhood.evaluate_fronts(show_time=False)
                need_to_resave = True

        # Save again to save us having to re-evaluate the fronts.
        if need_to_resave:
            pre_sampler.save_walk_neig_population(
                pop_walk, pop_neighbours_list, sample_number, walk_number
            )

        return pop_walk, pop_neighbours_list

    def get_rw_pop(
        self,
        pre_sampler,
        problem,
        sample_number,
        walk_number,
        eval_pops_parallel=False,
        eval_fronts=True,
    ):

        # Flag for whether we should manually generate new populations.
        continue_generation = True

        try:
            # Attempt to use pre-generated samples.
            pop_walk, pop_neighbours_list = pre_sampler.load_walk_neig_population(
                problem, sample_number, walk_number, is_adaptive=False
            )

            # Evaluate fronts.
            if eval_fronts:
                pop_walk, pop_neighbours_list = self.evaluate_ranks_for_walk_pop(
                    pre_sampler,
                    sample_number,
                    walk_number,
                    pop_walk,
                    pop_neighbours_list,
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
                problem,
                walk,
                neighbours,
                eval_fronts=eval_fronts,
                eval_pops_parallel=eval_pops_parallel,
            )
            pre_sampler.save_walk_neig_population(
                pop_walk, pop_neighbours_list, sample_number, walk_number
            )

        return pop_walk, pop_neighbours_list

    def get_aw_pop(
        self,
        pre_sampler,
        problem,
        sample_number,
        walk_number,
        awGenerator,
        start_point,
        eval_pops_parallel=False,
        eval_fronts=False,
    ):

        if eval_pops_parallel and not self.check_if_platemo():
            num_processes = self.num_processes_parallel_seed
        else:
            num_processes = 1

        # Flag for whether we should manually generate new populations.
        continue_generation = True

        try:
            # Attempt to use pre-generated samples.
            pop_walk, pop_neighbours_list = pre_sampler.load_walk_neig_population(
                problem, sample_number, walk_number, is_adaptive=True
            )

            # Evaluate fronts.
            if eval_fronts:
                pop_walk, pop_neighbours_list = self.evaluate_ranks_for_walk_pop(
                    pre_sampler,
                    sample_number,
                    walk_number,
                    pop_walk,
                    pop_neighbours_list,
                )

            # If loading is successful, skip the generation and saving process.
            continue_generation = False
        except FileNotFoundError:
            print(
                f"Generating AW population for sample {sample_number}, walk {walk_number} as it was not found."
            )

        if continue_generation:

            # Always best to just use 1 process here since we have to go step by step anyway.
            walk, pop_walk = awGenerator.do_adaptive_phc_walk_for_starting_point(
                start_point,
                constrained_ranks=True,
                return_pop=True,
                num_processes=num_processes,
            )
            neighbours = awGenerator.generate_neighbours_for_walk(walk)
            print(
                "Generated AW {} of {} (for sample {}). Length: {}".format(
                    walk_number + 1,
                    self.num_walks_aw,
                    sample_number,
                    walk.shape[0],
                )
            )

            # Now evaluate neighbours using parallel evaluation.
            pop_neighbours_list = self.generate_neig_populations(
                problem,
                neighbours,
                eval_fronts=False,
                adaptive_walk=True,
                num_processes=num_processes,
            )
            pre_sampler.save_walk_neig_population(
                pop_walk,
                pop_neighbours_list,
                sample_number,
                walk_number,
                is_adaptive=True,
            )

        return pop_walk, pop_neighbours_list

    @handle_ctrl_c
    def process_rw_sample_norm(self, args):
        variables = ["var", "obj", "cv"]

        i, pre_sampler, problem, eval_pops_parallel = args
        max_values_array = {
            "var": np.empty((0, problem.n_var)),
            "obj": np.empty((0, problem.n_obj)),
            "cv": np.empty((0, 1)),
        }
        min_values_array = max_values_array

        for j in range(self.num_walks_rw):

            pop_walk, pop_neighbours_list = self.get_rw_pop(
                pre_sampler,
                problem,
                i + 1,
                j + 1,
                eval_pops_parallel=eval_pops_parallel,
                eval_fronts=False,
            )

            pf = pop_walk.extract_pf()

            for which_variable in variables:
                combined_array = Analysis.combine_arrays_for_pops(
                    [pop_walk] + pop_neighbours_list, which_variable
                )
                fmin, fmax = self.compute_maxmin_for_sample(
                    combined_array, pf, which_variable
                )

                min_values_array[which_variable] = np.vstack(
                    (min_values_array[which_variable], fmin)
                )
                max_values_array[which_variable] = np.vstack(
                    (max_values_array[which_variable], fmax)
                )

        return min_values_array, max_values_array

    def compute_rw_normalisation_values(
        self, pre_sampler, problem, eval_pops_parallel=False
    ):

        norm_start = time.time()
        variables = ["var", "obj", "cv"]

        max_values_array = {
            "var": np.empty((0, problem.n_var)),
            "obj": np.empty((0, problem.n_obj)),
            "cv": np.empty((0, 1)),
        }
        min_values_array = max_values_array

        print_with_timestamp(
            f"Initialising normalisation computations for RW samples using {self.num_processes_rw_norm} cores. This requires full evaluation of the entire sample set and may take some time while still being memory-efficient."
        )

        if not eval_pops_parallel:

            print("Running seeds in series.")

            # Evaluate features in parallel
            with multiprocessing.Pool(
                self.num_processes_rw_norm, initializer=init_pool
            ) as pool:
                args_list = [
                    (i, pre_sampler, problem, False) for i in range(self.num_samples)
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
        else:

            if not self.check_if_platemo():
                print("Running seeds in parallel.")
            else:
                print("Running MATLAB from Python does not support parallelisation.")

            # Evaluate populations in parallel
            init_pool()  # set some global variables for ctrl + c handling
            for i in range(self.num_samples):
                start_time = time.time()
                min_values, max_values = self.process_rw_sample_norm(
                    (i, pre_sampler, problem, True)
                )
                for var in variables:
                    min_values_array[var] = np.vstack(
                        (min_values_array[var], min_values[var])
                    )
                    max_values_array[var] = np.vstack(
                        (max_values_array[var], max_values[var])
                    )
                end_time = time.time()

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

        for j in range(self.num_walks_rw):

            # Directly wrap the call to get_rw_pop inside the instantiation of RandomWalkAnalysis. We also need to evaluate the fronts now.
            rw_analysis = RandomWalkAnalysis(
                *self.get_rw_pop(pre_sampler, problem, i + 1, j + 1, eval_fronts=True),
                self.walk_normalisation_values,
                self.results_dir,
            )

            rw_analysis.eval_features()

            rw_analysis.clear_for_storage()

            rw_single_sample_analyses_list.append(rw_analysis)

        return rw_single_sample_analyses_list

    def do_random_walk_analysis(self, problem, pre_sampler, eval_pops_parallel=False):
        self.walk_normalisation_values = self.compute_rw_normalisation_values(
            pre_sampler, problem, eval_pops_parallel=eval_pops_parallel
        )

        rw_multiple_samples_analyses_list = []

        # Cannot pickle MATLAB objects for multiprocessing. Remove them temporarily.
        if self.check_if_platemo():
            matlab_prob, matlab_engine = problem.pop_matlab_objects()

        start_time = time.time()

        with multiprocessing.Pool(
            self.num_processes_rw_eval, initializer=init_pool
        ) as pool:
            # Use partial method here.
            print_with_timestamp(
                "\nRunning parallel computation for RW features with {} processes. \n".format(
                    self.num_processes_rw_eval
                )
            )
            results = pool.starmap(
                self.eval_single_sample_rw_features,
                zip(range(self.num_samples), repeat(pre_sampler), repeat(problem)),
            )

            if any(map(lambda x: isinstance(x, KeyboardInterrupt), results)):
                print("Ctrl-C was entered.")

            for i, rw_single_sample_analyses_list in enumerate(results):
                rw_single_sample = Analysis.concatenate_single_analyses(
                    rw_single_sample_analyses_list
                )
                rw_multiple_samples_analyses_list.append(rw_single_sample)

        # Re-insert matlab objects now that multiprocessing is done.
        if self.check_if_platemo():
            problem.insert_matlab_objects(matlab_prob, matlab_engine)

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

        # We already evaluated the populations when we computed the norms. Still need ranks though.
        pop_global = self.get_global_pop(pre_sampler, problem, i + 1, eval_fronts=True)

        global_analysis = GlobalAnalysis(
            pop_global,
            self.global_normalisation_values,
            self.results_dir,
        )

        # Pass to Analysis class for evaluation.
        global_analysis.eval_features()

        global_analysis.clear_for_storage()

        return global_analysis

    def do_global_analysis(self, problem, pre_sampler, eval_pops_parallel=False):
        self.global_normalisation_values = self.compute_global_normalisation_values(
            pre_sampler, problem, eval_pops_parallel=eval_pops_parallel
        )

        start_time = time.time()

        global_multiple_analyses_list = []

        # Cannot pickle MATLAB objects for multiprocessing. Remove them temporarily.
        if self.check_if_platemo():
            matlab_prob, matlab_engine = problem.pop_matlab_objects()

        with multiprocessing.Pool(
            self.num_processes_global_eval, initializer=init_pool
        ) as pool:
            print_with_timestamp(
                "\nRunning parallel computation for global features with {} processes. \n".format(
                    self.num_processes_global_eval
                )
            )

            # Use starmap with zip and repeat to pass the same pre_sampler and problem to each call
            results = pool.starmap(
                self.eval_single_sample_global_features,
                zip(range(self.num_samples), repeat(pre_sampler), repeat(problem)),
            )

            if any(map(lambda x: isinstance(x, KeyboardInterrupt), results)):
                print("Ctrl-C was entered.")

            for i, global_analysis in enumerate(results):
                global_multiple_analyses_list.append(global_analysis)

        # Re-insert matlab objects now that multiprocessing is done.
        if self.check_if_platemo():
            problem.insert_matlab_objects(matlab_prob, matlab_engine)

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
    def eval_single_sample_aw_features(
        self, i, pre_sampler, problem, awGenerator, eval_pops_parallel
    ):
        aw_single_sample_analyses_list = []

        # Loop over each of the walks within this sample.

        # Load in the pre-generated LHS sample as a starting point.
        pop_global = self.get_global_pop(pre_sampler, problem, i + 1, eval_fronts=False)
        pop_global_clean, _ = pop_global.remove_nan_inf_rows("global")
        distributed_sample = pop_global_clean.extract_var()

        for j in range(self.num_walks_aw):

            pop_walk, pop_neighbours_list = self.get_aw_pop(
                pre_sampler,
                problem,
                i + 1,
                j + 1,
                awGenerator,
                distributed_sample[j, :],
                eval_pops_parallel=eval_pops_parallel,
                eval_fronts=False,
            )

            # Cannot pickle MATLAB objects for multiprocessing. Remove them temporarily.
            if self.check_if_platemo():
                matlab_prob, matlab_engine = problem.pop_matlab_objects()

            aw_analysis = AdaptiveWalkAnalysis(
                pop_walk,
                pop_neighbours_list,
                self.global_normalisation_values,
                self.results_dir,
            )

            # Pass to Analysis class for evaluation.
            aw_analysis.eval_features()

            aw_analysis.clear_for_storage()
            aw_single_sample_analyses_list.append(aw_analysis)

            # Re-insert matlab objects now that multiprocessing is done.
            if self.check_if_platemo():
                problem.insert_matlab_objects(matlab_prob, matlab_engine)

        return aw_single_sample_analyses_list

    def do_adaptive_walk_analysis(self, problem, pre_sampler, eval_pops_parallel=False):
        # Initialise AW generator object before any loops.
        n_var = pre_sampler.dim

        if self.mode == "eval":
            # Experimental setup of Alsouly
            neighbourhood_size = pre_sampler.neighbourhood_size_rw

            max_steps = 500
            step_size = 0.01  # 1% of the range of the instance domain

        elif self.mode == "debug":
            # Runs quickly
            neighbourhood_size = 5
            max_steps = 5
            step_size = 0.1  # 1% of the range of the instance domain

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

        if not eval_pops_parallel:

            print("Running seeds in series.")

            with multiprocessing.Pool(
                self.num_processes_aw, initializer=init_pool
            ) as pool:
                print_with_timestamp(
                    "\nRunning parallel computation for AW features with {} processes. \n".format(
                        self.num_processes_aw
                    )
                )
                results = pool.starmap(
                    self.eval_single_sample_aw_features,
                    zip(
                        range(self.num_samples),
                        repeat(pre_sampler),
                        repeat(problem),
                        repeat(awGenerator),
                        repeat(False),
                    ),
                )

                if any(map(lambda x: isinstance(x, KeyboardInterrupt), results)):
                    print("Ctrl-C was entered.")

                for i, aw_single_sample_analyses_list in enumerate(results):
                    aw_single_sample = Analysis.concatenate_single_analyses(
                        aw_single_sample_analyses_list
                    )
                    aw_multiple_samples_analyses_list.append(aw_single_sample)
        else:
            print("Running seeds in parallel.")
            init_pool()
            for i in range(self.num_samples):
                start_time_seed = time.time()
                aw_single_sample_analyses_list = self.eval_single_sample_aw_features(
                    i, pre_sampler, problem, awGenerator, eval_pops_parallel=True
                )
                aw_single_sample = Analysis.concatenate_single_analyses(
                    aw_single_sample_analyses_list
                )
                aw_multiple_samples_analyses_list.append(aw_single_sample)
                end_time_seed = time.time()
                self.send_update_email(
                    f"{self.instance_name} finished AW seed {i+1}/{self.num_samples} in {end_time_seed - start_time_seed:.2f} seconds."
                )
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

    def initialize_evaluator(self, temp_pops_dir):

        # Define number of samples.
        self.initialize_number_of_samples()

        # Load presampler and create directories for populations.
        pre_sampler = self.create_pre_sampler()
        pre_sampler.create_pregen_sample_dir()
        pre_sampler.create_pops_dir(self.instance_name, temp_pops_dir)

        # Define number of cores for multiprocessing.
        self.initialize_number_of_cores()

        # Initialise PF text file.
        if not self.check_if_platemo():
            self.initialise_pf(self.instance)

        return pre_sampler

    def do(self, save_arrays, temp_pops_dir=None):
        print(
            "\n------------------------ Evaluating instance: "
            + self.instance_name
            + " ------------------------"
        )

        pre_sampler = self.initialize_evaluator(temp_pops_dir)

        self.send_initialisation_email(
            f"STARTED RUN OF {self.instance_name}.", pre_sampler
        )

        # MATLAB engine does not support multiprocessing.
        if self.check_if_aerofoil() or self.check_if_platemo():
            eval_pops_parallel = True
            eval_aw_pops_parallel = True
        elif self.check_if_modact():
            eval_pops_parallel = True
            eval_aw_pops_parallel = False
        else:
            eval_pops_parallel = False
            eval_aw_pops_parallel = False

        print(
            f"Evaluation parameters: RW/Global evaluating seeds in parallel is {eval_pops_parallel}, AW evaluating seeds in parallel is {eval_aw_pops_parallel}"
        )

        # For PlatEMO - Combine arrays and evaluate everything first.
        if self.check_if_platemo() or self.check_if_aerofoil():

            print(
                "Since this is a slow to evaluate instance, we will evaluated all RW and Global seeds first to speed up calculations."
            )

            if self.check_if_aerofoil():
                num_processes = self.num_processes_parallel_seed
            else:
                num_processes = 1

            # RWs
            for i in range(self.num_samples):
                self.evaluate_and_save_all_walks_in_sample(
                    i + 1,
                    self.instance,
                    eval_fronts=False,
                    num_processes=num_processes,
                    pre_sampler=pre_sampler,
                )

            # Global samples
            self.evaluate_and_save_all_global_samples(
                self.instance,
                eval_fronts=False,
                num_processes=num_processes,
                pre_sampler=pre_sampler,
            )

        # RW Analysis.
        print(
            " \n ~~~~~~~~~~~~ RW Analysis for "
            + self.instance_name
            + " ~~~~~~~~~~~~ \n"
        )

        rw_features = self.do_random_walk_analysis(
            self.instance, pre_sampler, eval_pops_parallel=eval_pops_parallel
        )
        rw_features.export_unaggregated_features(self.instance_name, "rw", save_arrays)

        # Global Analysis.
        print(
            " \n ~~~~~~~~~~~~ Global Analysis for "
            + self.instance_name
            + " ~~~~~~~~~~~~ \n"
        )

        global_features = self.do_global_analysis(
            self.instance, pre_sampler, eval_pops_parallel=eval_pops_parallel
        )
        global_features.export_unaggregated_features(
            self.instance_name, "glob", save_arrays
        )

        # Adaptive Walk Analysis. Always do in series since we need evaluations of one point to get to the next.
        print(
            " \n ~~~~~~~~~~~~ AW Analysis for "
            + self.instance_name
            + " ~~~~~~~~~~~~ \n"
        )
        aw_features = self.do_adaptive_walk_analysis(
            self.instance, pre_sampler, eval_pops_parallel=eval_aw_pops_parallel
        )
        aw_features.export_unaggregated_features(self.instance_name, "aw", save_arrays)

        # Close MATLAB session.
        if self.check_if_platemo():
            self.instance.end_matlab_session()

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

    def evaluate_and_save_all_walks_in_sample(
        self, sample_number, problem, eval_fronts, num_processes, pre_sampler
    ):
        """
        Load all walks and neighbors, concatenate them, evaluate in one go,
        and then split and save the results.

        Parameters:
        - sample_number: The identifier for the sample.
        - problem: The problem instance associated with the population.
        - eval_fronts: Boolean indicating whether to evaluate fronts.
        - num_processes: Number of processes to use for evaluation.
        """

        if pre_sampler.check_rw_preeval_pops_for_sample(
            sample_number, self.num_walks_rw
        ):
            print(f"All walks and neighbours for sample number {sample_number} exist.")
            return

        init_pool()

        st = time.time()

        # Initialize lists to hold all walks and neighbors
        all_walks = []
        all_neighbours = []

        # Load the saved walks and neighbours

        for ind_walk_number in range(self.num_walks_rw):
            walk, neighbours = pre_sampler.read_walk_neighbours(
                sample_number, ind_walk_number
            )
            all_walks.extend(walk)
            # Assuming neighbours are stored as a list of arrays, one for each step
            for neighbour in neighbours:
                all_neighbours.extend(neighbour)

        # Concatenate all walks and neighbours for evaluation
        concatenated_array = np.concatenate([all_walks, all_neighbours])

        print(f"Evaluating array of size {len(concatenated_array)}.")

        # Evaluate the concatenated array
        # Placeholder for evaluation, replace with your evaluation logic
        pop_total = Population(problem, n_individuals=len(concatenated_array))
        pop_total.evaluate(
            self.rescale_pregen_sample(concatenated_array, problem),
            eval_fronts=eval_fronts,
            num_processes=num_processes,
            show_msg=True,
        )

        # Split the evaluated array back into individual walks and neighbours
        num_walk_points = len(all_walks)
        pop_walks = pop_total[:num_walk_points]
        pop_neighbours = pop_total[num_walk_points:]

        # Save the evaluated walks and neighbours
        for ind_walk_number in range(self.num_walks_rw):
            walk_size = pre_sampler.num_steps_rw
            neighbour_size = pre_sampler.neighbourhood_size_rw

            pop_walk = pop_walks[:walk_size]
            pop_walks = pop_walks[walk_size:]

            pop_neighbours_list = []
            for _ in range(walk_size):
                pop_neighbour = pop_neighbours[:neighbour_size]
                pop_neighbours_list.append(pop_neighbour)
                pop_neighbours = pop_neighbours[neighbour_size:]

            pre_sampler.save_walk_neig_population(
                pop_walk,
                pop_neighbours_list,
                sample_number,
                ind_walk_number + 1,
                is_adaptive=False,
            )
        et = time.time()

        self.send_update_email(
            f"{self.instance_name} finished RW seed {sample_number + 1}/{self.num_samples} in {et - st:.2f} seconds."
        )

    def evaluate_and_save_all_global_samples(
        self, problem, eval_fronts, num_processes, pre_sampler
    ):
        """
        Load all global samples, concatenate them, evaluate in one go, and then save the results.

        Parameters:
        - problem: The problem instance associated with the population.
        - eval_fronts: Boolean indicating whether to evaluate fronts.
        - num_processes: Number of processes to use for evaluation.
        """

        if pre_sampler.check_global_preeval_pops():
            print(f"All global populations exist.")
            return

        init_pool()

        st = time.time()

        # Initialize a list to hold all global samples
        all_global_samples = []

        # Load the saved global samples
        for sample_number in range(1, self.num_samples + 1):
            global_sample = pre_sampler.read_global_sample(sample_number)
            all_global_samples.append(global_sample)

        # Concatenate all global samples for evaluation
        concatenated_array = np.concatenate(all_global_samples)

        print(f"Evaluating array of size {len(concatenated_array)}.")

        # Evaluate the concatenated array
        # Placeholder for evaluation, replace with your evaluation logic
        pop_total = Population(problem, n_individuals=len(concatenated_array))
        pop_total.evaluate(
            self.rescale_pregen_sample(concatenated_array, problem),
            eval_fronts=eval_fronts,
            num_processes=num_processes,
            show_msg=True,
        )

        # Split the evaluated array back into individual samples and save them
        start_idx = 0
        for sample_number, global_sample in enumerate(all_global_samples, start=1):
            end_idx = start_idx + len(global_sample)
            population = pop_total[start_idx:end_idx]

            # Save the evaluated population
            pre_sampler.save_global_population(population, sample_number)

            start_idx = end_idx

        et = time.time()
        self.send_update_email(
            f"{self.instance_name} finished all global seeds in {et - st:.2f} seconds."
        )


if __name__ == "__main__":
    # Making sure the binary pattern generator is generating the right number of starting zones.
    pe = ProblemEvaluator([])
    print(len(pe.generate_binary_patterns(10)))
