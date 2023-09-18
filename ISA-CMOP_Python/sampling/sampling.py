import numpy as np


class Sampling:
    def __init__(self):
        self.x = None

    def do(self, n_samples, x_lower, x_upper, seed=None):
        # Number of dimensions
        dim = x_lower.shape[0]

        # Sample design space
        self._do(dim, n_samples, seed)

        # Scaling sampling to fit design variable limits
        self.x = self.x * (x_upper - x_lower) + x_lower

    def _do(self, dim, n_samples, seed=None):
        pass
