from optimisation.operators.sampling.RandomWalk import RandomWalk
from optimisation.operators.sampling.latin_hypercube_sampling import (
    LatinHypercubeSampling,
)
from optimisation.operators.sampling.random_sampling import RandomSampling
from features.ancillary_functions import print_with_timestamp
import time
import math
import numpy as np
import os
import sys
import shutil
import pickle
from optimisation.model.population import Population


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
            self.neighbourhood_size_rw = 5
            self.num_steps_rw = 5
            self.step_size_rw = 0.1  # 10% of the range of the instance domain

            # Experimental setup of Liefooghe2021 for global.
            self.num_points_glob = int(self.dim * 5)
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

    def create_pops_dir(self, problem_name, temp_pops_dir):

        if temp_pops_dir:
            pops_dir = temp_pops_dir
        else:
            pops_dir = "../temp_pops"

        dirs_to_create = ["global", "rw", "aw"]
        print(
            f"Starting the creation process for population directories for problem: {problem_name} in mode: {self.mode}"
        )

        for dir_name in dirs_to_create:
            dir_path = os.path.join(pops_dir, problem_name, self.mode, dir_name)

            # Check if the directory already exists
            if not os.path.exists(dir_path):
                print(f"Directory does not exist and will be created: {dir_path}")
                os.makedirs(dir_path)
                print(f"Directory created: {dir_path}")
            else:
                print(
                    f"Directory already exists and does not need to be created: {dir_path}"
                )

            # Save the directory paths to instance variables for later use
            if dir_name == "global":
                self.global_pop_dir = dir_path
            elif dir_name == "rw":
                self.rw_pop_dir = dir_path
            elif dir_name == "aw":
                self.aw_pop_dir = dir_path

        print(
            f"Population directories setup complete.\nGlobal population directory: {self.global_pop_dir}\nRW population directory: {self.rw_pop_dir}"
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

    def generate_single_rw_walk(self, sample_number, ind_walk_number):
        """
        Generate a single walk within a sample (collection of walks)
        """

        # Make RW generator object.
        rwGenerator = RandomWalk(
            self.dim, self.num_steps_rw, self.step_size_rw, self.neighbourhood_size_rw
        )

        starting_zones = self.generate_binary_patterns()
        starting_zone = starting_zones[ind_walk_number - 1]  # indices start at 1

        # Generate random walk starting at this iteration's starting zone.
        walk = rwGenerator.do_progressive_walk(seed=None, starting_zone=starting_zone)

        # Generate neighbors for each step on the walk. Currently, we just randomly sample points in the [-stepsize, stepsize] hypercube
        neighbours = rwGenerator.generate_neighbours_for_walk(walk)

        # Save walk and neighbours arrays in the sample folder
        # Create a folder for each sample
        sample_folder = os.path.join(self.rw_samples_dir, f"sample{sample_number}")

        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder, exist_ok=True)

        save_path = os.path.join(
            sample_folder, f"walk_neighbours_{ind_walk_number}.npz"
        )
        np.savez(save_path, walk=walk, neighbours=neighbours)

        print(
            "Generated RW {} of {} (for this sample).".format(
                ind_walk_number,
                len(starting_zones),
            )
        )

        return walk, neighbours

    def generate_single_rw_sample(self, sample_number):

        starting_zones = self.generate_binary_patterns()

        start_time = time.time()  # Record the start time for this sample

        for ctr, starting_zone in enumerate(starting_zones):
            self.generate_single_rw_walk(sample_number, ctr + 1)

        # Record elapsed time and print.
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(
            "Generated set of RWs {} of {} in {:.2f} seconds.\n".format(
                sample_number, self.num_samples, elapsed_time
            )
        )

    def generate_rw_samples(self):
        """
        Generate a RW sample on the unit hypercube.

        Note that a RW sample refers to a set of independent RWs.
        """

        print("")
        print(
            "Generating {} samples (walks + neighbours) for RW features with the following properties:".format(
                self.num_samples
            )
        )
        print("- Number of walks: {}".format(self.dim))
        print("- Number of steps per walk: {}".format(self.num_steps_rw))
        print("- Step size (% of instance domain): {}".format(self.step_size_rw * 100))
        print("- Neighbourhood size: {}".format(self.neighbourhood_size_rw))
        print("")

        for i in range(self.num_samples):
            self.generate_single_rw_sample(i + 1)

    def generate_single_global_sample(self, sample_number):
        start_time = time.time()  # Record the start time
        sampler = LatinHypercubeSampling(
            criterion="maximin",
            iterations=self.iterations_glob,
            method="scipy",
        )

        sampler.do(
            n_samples=self.num_points_glob,
            x_lower=np.zeros(self.dim),
            x_upper=np.ones(self.dim),
        )

        # Save to numpy binary.
        save_path = os.path.join(
            self.global_samples_dir, f"lhs_sample_{sample_number}.npz"
        )
        np.savez(save_path, global_sample=sampler.x)
        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time

        print(
            "Generated Global sample {} of {} in {:.2f} seconds.".format(
                sample_number, self.num_samples, elapsed_time
            )
        )

    def generate_all_global_samples(self):
        """
        Generate a LHS sample on the unit hypercube.
        """

        print(
            "Generating distributed samples for Global features with the following properties:"
        )
        print("- Num. points: {}".format(self.num_points_glob))
        print("- Method: {}".format("lhs.scipy"))
        print("")

        for i in range(self.num_samples):
            self.generate_single_global_sample(i + 1)

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

        walk_neighbours_file = f"walk_neighbours_{ind_walk_number}.npz"
        file_path = os.path.join(sample_path, walk_neighbours_file)

        if not os.path.exists(file_path):
            print(
                f"RW sample no. {sample_number} / walk no. {ind_walk_number} does not exist. Generating..."
            )
            return self.generate_single_rw_walk(sample_number, ind_walk_number)

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
            print(f"Global sample no. {sample_number} does not exist. Generating...")
            self.generate_single_global_sample(sample_number)

        # Load data from the npz file
        data = np.load(file_path)
        global_sample = data["global_sample"]
        print(
            f"Global sample size for {self.dim}d, no. {sample_number}: {global_sample.shape[0]}"
        )

        return global_sample

    def save_global_population(self, pop_global, sample_number):

        # Ensure the "global" directory exists
        if not os.path.exists(self.global_pop_dir):
            os.makedirs(self.global_pop_dir, exist_ok=True)

        # Define the path for the file to save the global population
        file_path = os.path.join(self.global_pop_dir, f"pop_global_{sample_number}.npz")

        # Save the `pop_global` object to the file
        # with open(file_path, "wb") as file:
        #     pickle.dump(pop_global, file)
        pop_global.save_population_attributes(file_path)

        print_with_timestamp(
            f"Saved global population and individual neighbours for sample {sample_number} in {file_path}."
        )

    def load_global_population(self, problem, sample_number):
        # Path for the file from which to load the global population
        file_path = os.path.join(self.global_pop_dir, f"pop_global_{sample_number}.npz")

        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"No saved global population found for sample {sample_number} at {file_path}"
            )

        # Load and return the `pop_global` object from the file
        # with open(file_path, "rb") as file:
        #     pop_global = pickle.load(file)
        pop_global = Population.from_saved_attributes(file_path, problem)

        print_with_timestamp(
            f"Global population for sample {sample_number} loaded from {file_path}. Length: {len(pop_global)}"
        )

        return pop_global

    def check_pop_size(self, file_path, expected_length):
        data = np.load(file_path)
        var = data["var"]
        if len(var) != expected_length:
            print(
                f"Population does not contain the expected number of rows ({expected_length}). Found {len(var)}."
            )
            return False
        else:
            return True

    def check_global_preeval_pops(self):

        print(
            "\nChecking if global populations have already been evaluated for this problem instance."
        )

        # Check 1: Folder exists
        if not os.path.exists(self.global_pop_dir):
            print("Global pre-eval pops folder does not exist.")
            return False

        # Check 2: Correct number of files
        files = [
            f
            for f in os.listdir(self.global_pop_dir)
            if os.path.isfile(os.path.join(self.global_pop_dir, f))
        ]
        if len(files) < self.num_samples:
            print(
                f"Folder does not contain enough sample files to generate all samples ({self.num_samples}). Found {len(files)} files."
            )
            return False

        # Check 3: Each file has correct number of points
        for i, file_name in enumerate(files):
            file_path = os.path.join(self.global_pop_dir, file_name)
            if not self.check_pop_size(file_path, self.num_points_glob):
                print(
                    f"Population {i + 1} does not contain the expected number of rows ({self.num_points_glob}). Found {len()}."
                )
                return False

        print("All global checks passed successfully.")
        return True

    def save_walk_neig_population(
        self,
        pop_walk,
        pop_neighbours_list,
        sample_number,
        walk_ind_number,
        is_adaptive=False,
    ):

        # Path for the specific sample directory within the "rw" directory
        if is_adaptive:
            sample_dir_path = os.path.join(self.aw_pop_dir, f"sample{sample_number}")
        else:
            sample_dir_path = os.path.join(self.rw_pop_dir, f"sample{sample_number}")

        # Ensure the sample directory exists
        if not os.path.exists(sample_dir_path):
            os.makedirs(sample_dir_path, exist_ok=True)

        # Define the file path for the walk population within the sample directory
        walk_file_path = os.path.join(
            sample_dir_path, f"pop_walk_{walk_ind_number}.npz"
        )

        # Save the `pop_walk` object to its file
        # with open(walk_file_path, "wb") as file:
        #     pickle.dump(pop_walk, file)
        pop_walk.save_population_attributes(walk_file_path)

        # Save each item in the `pop_neighbours_list` to its own file
        for i, neighbour in enumerate(pop_neighbours_list):
            neighbour_file_path = os.path.join(
                sample_dir_path, f"pop_neighbours_{walk_ind_number}_{i}.npz"
            )
            # with open(neighbour_file_path, "wb") as file:
            #     pickle.dump(neighbour, file)
            neighbour.save_population_attributes(neighbour_file_path)

        print_with_timestamp(
            f"Saved walk population and individual neighbours for sample {sample_number}, walk {walk_ind_number} in {sample_dir_path}."
        )

    def load_walk_neig_population(
        self, problem, sample_number, walk_ind_number, is_adaptive=False
    ):
        # Path for the specific sample directory within the "rw" directory
        if is_adaptive:
            sample_dir_path = os.path.join(self.aw_pop_dir, f"sample{sample_number}")
        else:
            sample_dir_path = os.path.join(self.rw_pop_dir, f"sample{sample_number}")

        # Define the file path for the walk population within the sample directory
        walk_file_path = os.path.join(
            sample_dir_path, f"pop_walk_{walk_ind_number}.npz"
        )

        # Check if the walk file exists and raise an error if not
        if not os.path.exists(walk_file_path):
            raise FileNotFoundError(
                f"Walk file for sample {sample_number}, walk {walk_ind_number} not found in {sample_dir_path}"
            )

        # Load the `pop_walk` object from its file
        # with open(walk_file_path, "rb") as file:
        #     pop_walk = pickle.load(file)
        pop_walk = Population.from_saved_attributes(walk_file_path, problem)

        # Initialize an empty list to hold the neighbours
        pop_neighbours_list = []

        # List all neighbour files for the given walk in the directory
        neighbours_files = [
            f
            for f in os.listdir(sample_dir_path)
            if f.startswith(f"pop_neighbours_{walk_ind_number}_")
        ]

        # Compute the number of neighbour files dynamically
        number_of_neighbours_files = len(neighbours_files)

        # Assuming file names are in the format "pop_neighbours_{walk_ind_number}_{index}.npz", sort and load in numeric order so that steps and neighbours agree.

        for i in range(number_of_neighbours_files):
            neighbour_file_str = f"pop_neighbours_{walk_ind_number}_{i}.npz"
            neighbour_file_path = os.path.join(sample_dir_path, neighbour_file_str)
            if os.path.exists(neighbour_file_path):
                # with open(neighbour_file_path, "rb") as file:
                #     neighbour = pickle.load(file)
                neighbour = Population.from_saved_attributes(
                    neighbour_file_path, problem
                )
                pop_neighbours_list.append(neighbour)

        print_with_timestamp(
            f"Loaded walk and individual neighbours for sample {sample_number}, walk {walk_ind_number} from {sample_dir_path}."
        )

        return pop_walk, pop_neighbours_list

    def check_rw_preeval_pops_for_sample(self, sample_number, max_walk_number):

        print(
            "\nChecking if RW populations have already been evaluated for this problem instance."
        )

        # Check 1: RW directory exists
        if not os.path.exists(self.rw_pop_dir):
            print("RW pre-eval pops folder does not exist.")
            return False

        sample_dir_path = os.path.join(self.rw_pop_dir, f"sample{sample_number}")
        # Check 2: Sample directory exists
        if not os.path.exists(sample_dir_path):
            print(f"Sample directory {sample_dir_path} does not exist.")
            return False

        # Loop through each of the walks (there are n of them per sample)
        for walk_ind_number in range(max_walk_number):

            # Just look at walks because they are always saved in tandem with neighbours.
            walk_file_path = os.path.join(
                sample_dir_path, f"pop_walk_{walk_ind_number}.npz"
            )
            # neighbours_file_path = os.path.join(
            #     sample_dir_path, f"pop_neighbours_list_{walk_ind_number}.npz"
            # )

            # Check 3: Walk and neighbours files exist
            if not os.path.isfile(walk_file_path):
                print(
                    f"Missing files for sample {sample_number}, walk {walk_ind_number}."
                )
                return False

            # Check 4: walk sizes are correct.
            if not self.check_pop_size(walk_file_path, self.num_steps_rw):
                return False

        print("All checks passed successfully for RW populations.")
        return True


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
