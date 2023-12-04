from optimisation.operators.sampling.RandomWalk import RandomWalk
from optimisation.operators.sampling.latin_hypercube_sampling import (
    LatinHypercubeSampling,
)
from optimisation.operators.sampling.random_sampling import RandomSampling
import time
import math
import numpy as np
import os
import sys


class PreSampler:
    def __init__(self, dim, num_samples):
        self.dim = dim
        self.num_samples = num_samples
        self.create_pregen_sample_dir()

    def create_pregen_sample_dir(self):
        base_dir = "pregen_samples"
        rw_dir = os.path.join(base_dir, "rw")
        global_dir = os.path.join(base_dir, "global")

        for directory in [base_dir, rw_dir, global_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)

        # Create subdirectory for the specific dimension inside the "rw" directory
        self.rw_dim_subdirectory = os.path.join(rw_dir, f"{self.dim}d")
        if not os.path.exists(self.rw_dim_subdirectory):
            os.makedirs(self.rw_dim_subdirectory)

        # Create subdirectory for the specific dimension inside the "global" directory
        self.global_dim_subdirectory = os.path.join(global_dir, f"{self.dim}d")
        if not os.path.exists(self.global_dim_subdirectory):
            os.makedirs(self.global_dim_subdirectory)

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

        # Alsouly's experimental setup.
        neighbourhood_size = 2 * self.dim + 1
        num_steps = 1000
        step_size = 0.01  # 1% of the range of the instance domain

        # Make RW generator object.
        rwGenerator = RandomWalk(self.dim, num_steps, step_size, neighbourhood_size)
        starting_zones = self.generate_binary_patterns()

        print("")
        print(
            "Generating {} samples (walks + neighbours) for RW features with the following properties:".format(
                self.num_samples
            )
        )
        print("- Number of walks: {}".format(len(starting_zones)))
        print("- Number of steps per walk: {}".format(num_steps))
        print("- Step size (% of instance domain): {}".format(step_size * 100))
        print("- Neighbourhood size: {}".format(neighbourhood_size))
        print("")

        for i in range(self.num_samples):
            start_time = time.time()  # Record the start time for this sample

            # Create a folder for each sample
            sample_folder = os.path.join(self.rw_dim_subdirectory, f"sample{i + 1}")
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
        # Experimental setup of Liefooghe2021.
        num_points = int(self.dim * 200)
        iterations = num_points  # not relevant for lhs scipy.

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
        print("- Method: {}".format(method))
        print("")

        for i in range(self.num_samples):
            start_time = time.time()  # Record the start time
            if method_name == "lhs":
                sampler = LatinHypercubeSampling(
                    criterion="maximin", iterations=iterations, method=lhs_method_name
                )
            elif method_name == "uniform":
                sampler = RandomSampling()

            sampler.do(
                n_samples=num_points,
                x_lower=np.zeros(self.dim),
                x_upper=np.ones(self.dim),
            )

            # Save to numpy binary.
            save_path = os.path.join(
                self.global_dim_subdirectory, f"lhs_sample_{i + 1}.npz"
            )
            np.savez(save_path, global_sample=sampler.x)
            end_time = time.time()  # Record the end time
            elapsed_time = end_time - start_time

            print(
                "Generated Global sample {} of {} in {:.2f} seconds.".format(
                    i + 1, self.num_samples, elapsed_time
                )
            )


def main():
    if len(sys.argv) != 3:
        print("Usage: python PreSampler.py dim num_samples")
        return

    dim = int(sys.argv[1])
    num_samples = int(sys.argv[2])

    # Create an instance of the PreSampler class
    pre_sampler = PreSampler(dim, num_samples)

    # Generate a random walk sample with the specified number of samples
    pre_sampler.generate_rw_samples()

    # Generate a LHS sample.
    pre_sampler.generate_global_samples()


if __name__ == "__main__":
    main()
