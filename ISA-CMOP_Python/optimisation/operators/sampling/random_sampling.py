import numpy as np

from optimisation.model.sampling import Sampling


class RandomSampling(Sampling):

    def __init__(self):
        super().__init__()

    def _do(self, dim, n_samples, seed=None):

        self.x = np.random.random((n_samples, dim))

