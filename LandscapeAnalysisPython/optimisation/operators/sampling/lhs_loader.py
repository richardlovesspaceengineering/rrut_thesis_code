import numpy as np
from optimisation.model.sampling import Sampling

from optimisation.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling


class LHSLoader(Sampling):

    def __init__(self, base_path='./cases/LHS_Initialisation/', lhs_iter=10000):
        super().__init__()
        self.base_path = base_path
        self.full_path = None
        self.lhs = LatinHypercubeSampling(iterations=lhs_iter)

    def _do(self, dim, n_samples, seed=None):
        print('LHS Loader in use')
        # Obtain path of LHS file
        local_path = f"{dim}D_{n_samples}N_{seed}S.lhs"
        self.full_path = self.base_path + local_path

        # Attempt to load the LHS into memory
        try:
            self.x = np.genfromtxt(self.full_path)
        except OSError:
            raise Exception(f"The LHS file at {self.full_path} was not found!")

    def _generate(self, dims, n_samples, seeds):
        # Generate
        for i, dim in enumerate(dims):
            x_lower = np.zeros(dim)
            x_upper = np.ones(dim)

            for seed in seeds:
                # Determine full path and filename
                local_path = f"{dim}D_{n_samples[i]}N_{seed}S.lhs"
                full_path = self.base_path + local_path

                # Generate LHS sampling points
                n = seed * 100
                self.lhs.do(n_samples=n_samples[i], x_lower=x_lower, x_upper=x_upper, seed=n)
                x_lhs = self.lhs.x

                # Save file to specified path
                np.savetxt(full_path, x_lhs)  # delimiter is default " " space
                print('n_seed:', seed, 'seed:', n, 'x_lhs.shape:', x_lhs.shape)


if __name__ == "__main__":
    dims = np.array([14])
    seeds = list(range(0, 30))
    n_samples = 5 * dims

    loader = LHSLoader(base_path='../../../cases/LHS_Initialisation/')
    loader._generate(n_samples=n_samples, dims=dims, seeds=seeds)
