import numpy as np
import pickle
import random
from pyDOE2 import lhs

from optimisation.model.sampling import Sampling

# TODO: need to make sure that the seed airfoils are scaled between 0 and 1 when going in here
# TODO: work out how to pass a file into this (additional argument is probably the best option)


class SeedSampling(Sampling):

    def __init__(self):
        super().__init__()
        self.filename = './lib/normalised_seeds.pkl'

    def _do(self, dim, n_samples, seed=None):

        with open(self.filename, 'rb') as f:
            data = pickle.load(f)
            f.close()

        nr_seeds = len(data)
        self.x = []
        if nr_seeds >= n_samples:
            # select random seeds from data
            random_list = random.sample(range(0, nr_seeds), n_samples)
            #  TODO need to add columns for flight conditions (AoA)
            temp = data[random_list, :]
            extra_cols = dim - temp.shape[1]
            if extra_cols > 0:  # Need to be careful here - call with 0 samples to LHS fails
                extra_vals = lhs(extra_cols, samples=n_samples, criterion='maximin', iterations=20, random_state=seed)
                self.x = np.hstack((temp, extra_vals))
            else:
                self.x = temp
        else:
            # need additional seeds - latin hypercube sampling for now
            nr_additional_seeds = n_samples - len(data)
            dim1 = data.shape[1]
            x = lhs(dim1, samples=nr_additional_seeds, criterion='maximin', iterations=20, random_state=seed)
            # replace previous line with next line if random sampling is preferred for additional samples
            # self.x = np.random.random((nr_additional_seeds, dim))
            x = np.append(x, data, axis=0)
            extra_cols = dim - x.shape[1]
            if extra_cols > 0:  # Need to be careful here - call with 0 samples to LHS fails
                extra_vals = lhs(extra_cols, samples=n_samples, criterion='maximin', iterations=20, random_state=seed)
                self.x = np.hstack((x, extra_vals))
            else:
                self.x = x

