from optimisation.operators.sampling.RandomWalk import RandomWalk
from optimisation.operators.sampling.latin_hypercube_sampling import (
    LatinHypercubeSampling,
)
from optimisation.operators.sampling.random_sampling import RandomSampling
from features.feature_helpers import print_with_timestamp
import time
import math
import numpy as np
import os
import sys
import shutil
import pickle


class PreSampler:
    def __init__(self, dim, num_samples, mode):
        self.dim = dim
        self.num_samples = num_samples
        self.mode = mode
        self.save_experimental_setup()

    def save_experimental_setup(self):
        if self.mode == "eval":
            # Alsouly's experimental setup for RWs.
            self.neighbourhood_size_rw = 2 * self.dim + 1
            self.num_steps_rw = 1000
            self.step_size_rw = 0.01  # 1% of the range of the instance domain

            if self.dim <= 10:
                self.num_points_glob = int(self.dim * 1000)
            else:
                self.num_points_glob = int(1e4 / 40 * self.dim + 0.75e4)
            self.iterations_glob = self.num_points_glob  # not relevant for lhs scipy.
        elif self.mode == "debug":
            self.neighbourhood_size_rw = 2 * self.dim + 1
            self.num_steps_rw = 20
            self.step_size_rw = 0.01  # 1% of the range of the instance domain

            # Experimental setup of Liefooghe2021 for global.
            self.num_points_glob = int(self.dim * 20)
            self.iterations_glob = self.num_points_glob  # not relevant for lhs scipy.

    def create_pregen_sample_dir(self):
        base_dir = "../pregen_samples"
        mode_dir = os.path.join(base_dir, self.mode)  # "eval" or "debug" based on mode
        rw_dir = os.path.join(mode_dir, "rw")
        global_dir = os.path.join(mode_dir, "global")

        for directory in [base_dir, mode_dir, rw_dir, global_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)

        # Create subdirectory for the specific dimension inside the "rw" directory
        self.rw_samples_dir = os.path.join(rw_dir, f"{self.dim}d")
        if not os.path.exists(self.rw_samples_dir):
            os.makedirs(self.rw_samples_dir)

        # Create subdirectory for the specific dimension inside the "global" directory
        self.global_samples_dir = os.path.join(global_dir, f"{self.dim}d")
        if not os.path.exists(self.global_samples_dir):
            os.makedirs(self.global_samples_dir)

    def create_pops_dir(self, problem_name, reeval_pops=False):
        pops_dir = "../evaluated_pops"
        dirs_to_create = ["global", "rw"]

        for dir_name in dirs_to_create:
            dir_path = os.path.join(pops_dir, problem_name, self.mode, dir_name)

            # Remove the directory if it exists and reeval_pops is True
            if os.path.exists(dir_path) and reeval_pops:
                shutil.rmtree(dir_path)
                print(f"Cleaning directory at: {dir_path} since reval_pops=True.")

            # Create the directory if not exists or after removal
            os.makedirs(dir_path, exist_ok=True)

            # Save the directory paths to instance variables for later use
            if dir_name == "global":
                self.global_pop_dir = dir_path
            elif dir_name == "rw":
                self.rw_pop_dir = dir_path

        print(
            f"Directories created or refreshed:\nGlobal: {self.global_pop_dir}\nRW: {self.rw_pop_dir}"
        )

    def generate_binary_patterns(self):
        """
        Generate starting zones (represented as binary arrays) at every 2^n/n-th edge of the search space.
        """

        num_patterns = 2**self.dim
        patterns = []
        step = math.ceil(num_patterns / self.dim)

        for i in range(0, num_patterns, step):
            binary_pattern = np.binary_repr(i, width=self.dim)
            patterns.append([int(bit) for bit in binary_pattern])
        return patterns

    def generate_rw_samples(self):
        """
        Generate a RW sample on the unit hypercube.

        Note that a RW sample refers to a set of independent RWs.
        """

        # Make RW generator object.
        rwGenerator = RandomWalk(
            self.dim, self.num_steps_rw, self.step_size_rw, self.neighbourhood_size_rw
        )
        starting_zones = self.generate_binary_patterns()

        print("")
        print(
            "Generating {} samples (walks + neighbours) for RW features with the following properties:".format(
                self.num_samples
            )
        )
        print("- Number of walks: {}".format(len(starting_zones)))
        print("- Number of steps per walk: {}".format(self.num_steps_rw))
        print("- Step size (% of instance domain): {}".format(self.step_size_rw * 100))
        print("- Neighbourhood size: {}".format(self.neighbourhood_size_rw))
        print("")

        for i in range(self.num_samples):
            start_time = time.time()  # Record the start time for this sample

            # Create a folder for each sample
            sample_folder = os.path.join(self.rw_samples_dir, f"sample{i + 1}")
            os.makedirs(sample_folder, exist_ok=True)

            for ctr, starting_zone in enumerate(starting_zones):
                # Generate random walk starting at this iteration's starting zone.
                walk = rwGenerator.do_progressive_walk(
                    seed=None, starting_zone=starting_zone
                )

                # Generate neighbors for each step on the walk. Currently, we just randomly sample points in the [-stepsize, stepsize] hypercube
                neighbours = rwGenerator.generate_neighbours_for_walk(walk)

                # Save walk and neighbours arrays in the sample folder
                save_path = os.path.join(
                    sample_folder, f"walk_neighbours_{ctr + 1}.npz"
                )
                np.savez(save_path, walk=walk, neighbours=neighbours)

                print(
                    "Generated RW {} of {} (for this sample).".format(
                        ctr + 1,
                        len(starting_zones),
                    )
                )

            # Record elapsed time and print.
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(
                "Generated set of RWs {} of {} in {:.2f} seconds.\n".format(
                    i + 1, self.num_samples, elapsed_time
                )
            )

    def generate_global_samples(self, method="lhs.scipy"):
        """
        Generate a LHS sample on the unit hypercube.
        """

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
        print("- Num. points: {}".format(self.num_points_glob))
        print("- Method: {}".format(method))
        print("")

        for i in range(self.num_samples):
            start_time = time.time()  # Record the start time
            if method_name == "lhs":
                sampler = LatinHypercubeSampling(
                    criterion="maximin",
                    iterations=self.iterations_glob,
                    method=lhs_method_name,
                )
            elif method_name == "uniform":
                sampler = RandomSampling()

            sampler.do(
                n_samples=self.num_points_glob,
                x_lower=np.zeros(self.dim),
                x_upper=np.ones(self.dim),
            )

            # Save to numpy binary.
            save_path = os.path.join(self.global_samples_dir, f"lhs_sample_{i + 1}.npz")
            np.savez(save_path, global_sample=sampler.x)
            end_time = time.time()  # Record the end time
            elapsed_time = end_time - start_time

            print(
                "Generated Global sample {} of {} in {:.2f} seconds.".format(
                    i + 1, self.num_samples, elapsed_time
                )
            )

    def read_walk_neighbours(self, sample_number, ind_walk_number):
        """
        Read the walk and neighbours data from the pregen_samples directory.

        Parameters:
        - sample_number (int): The sample number.

        Returns:
        - Tuple containing the walk array and neighbours array.
        """
        sample_folder = f"sample{sample_number}"
        sample_path = os.path.join(self.rw_samples_dir, sample_folder)

        if not os.path.exists(sample_path):
            raise FileNotFoundError(f"Sample folder not found: {sample_path}")

        walk_neighbours_file = f"walk_neighbours_{ind_walk_number}.npz"
        file_path = os.path.join(sample_path, walk_neighbours_file)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Walk and neighbours file not found: {file_path}")

        # Load data from the npz file
        data = np.load(file_path)
        walk = data["walk"]
        neighbours = data["neighbours"]

        return walk, neighbours

    def read_global_sample(self, sample_number):
        """
        Read the global sample data from the pregen_samples directory.

        Parameters:
        - sample_number (int): The sample number.

        Returns:
        - Array containing the global sample.
        """
        file_name = f"lhs_sample_{sample_number}.npz"
        file_path = os.path.join(self.global_samples_dir, file_name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Global sample file not found: {file_path}")

        # Load data from the npz file
        data = np.load(file_path)
        global_sample = data["global_sample"]

        return global_sample

    def save_global_population(self, pop_global, sample_number):

        # Ensure the "global" directory exists
        if not os.path.exists(self.global_pop_dir):
            os.makedirs(self.global_pop_dir, exist_ok=True)

        # Define the path for the file to save the global population
        file_path = os.path.join(self.global_pop_dir, f"pop_global_{sample_number}.pkl")

        # Save the `pop_global` object to the file
        with open(file_path, "wb") as file:
            pickle.dump(pop_global, file)

        print(f"Global population for sample {sample_number} saved to {file_path}.")

    def load_global_population(self, sample_number):
        # Path for the file from which to load the global population
        file_path = os.path.join(self.global_pop_dir, f"pop_global_{sample_number}.pkl")

        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"No saved global population found for sample {sample_number} at {file_path}"
            )

        # Load and return the `pop_global` object from the file
        with open(file_path, "rb") as file:
            pop_global = pickle.load(file)

        print_with_timestamp(
            f"Global population for sample {sample_number} loaded from {file_path}."
        )

        return pop_global

    def save_walk_neig_population(
        self, pop_walk, pop_neighbours_list, sample_number, walk_ind_number
    ):

        # Path for the specific sample directory within the "rw" directory
        sample_dir_path = os.path.join(self.rw_pop_dir, f"sample{sample_number}")

        # Ensure the sample directory exists
        if not os.path.exists(sample_dir_path):
            os.makedirs(sample_dir_path, exist_ok=True)

        # Define the file paths for the walk and neighbors populations within the sample directory
        walk_file_path = os.path.join(
            sample_dir_path, f"pop_walk_{walk_ind_number}.pkl"
        )
        neighbours_file_path = os.path.join(
            sample_dir_path, f"pop_neighbours_list_{walk_ind_number}.pkl"
        )

        # Save the `pop_walk` object to its file
        with open(walk_file_path, "wb") as file:
            pickle.dump(pop_walk, file)

        # Save the `pop_neighbours_list` object to its file
        with open(neighbours_file_path, "wb") as file:
            pickle.dump(pop_neighbours_list, file)

        print(
            f"Walk and neighbours populations for sample {sample_number}, walk {walk_ind_number} saved in {sample_dir_path}."
        )

    def load_walk_neig_population(self, sample_number, walk_ind_number):
        # Path for the specific sample directory within the "rw" directory
        sample_dir_path = os.path.join(self.rw_pop_dir, f"sample{sample_number}")

        # Define the file paths for the walk and neighbors populations within the sample directory
        walk_file_path = os.path.join(
            sample_dir_path, f"pop_walk_{walk_ind_number}.pkl"
        )
        neighbours_file_path = os.path.join(
            sample_dir_path, f"pop_neighbours_list_{walk_ind_number}.pkl"
        )

        # Check if the files exist and raise an error if not
        if not os.path.exists(walk_file_path) or not os.path.exists(
            neighbours_file_path
        ):
            raise FileNotFoundError(
                f"Files for sample {sample_number}, walk {walk_ind_number} not found in {sample_dir_path}"
            )

        # Load the `pop_walk` object from its file
        with open(walk_file_path, "rb") as file:
            pop_walk = pickle.load(file)

        # Load the `pop_neighbours_list` object from its file
        with open(neighbours_file_path, "rb") as file:
            pop_neighbours_list = pickle.load(file)

        print_with_timestamp(
            f"Walk and neighbours populations for sample {sample_number}, walk {walk_ind_number} loaded from {sample_dir_path}."
        )

        return pop_walk, pop_neighbours_list


def main():
    if len(sys.argv) != 4:
        print("Usage: python PreSampler.py dim num_samples mode")
        return

    dim = int(sys.argv[1])
    num_samples = int(sys.argv[2])
    mode = str(sys.argv[3])

    # Create an instance of the PreSampler class
    pre_sampler = PreSampler(dim, num_samples, mode)

    # Generate a random walk sample with the specified number of samples
    pre_sampler.generate_rw_samples()

    # Generate a LHS sample.
    pre_sampler.generate_global_samples()


if __name__ == "__main__":
    main()
