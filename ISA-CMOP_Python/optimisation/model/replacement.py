import numpy as np


class ReplacementStrategy:

    def do(self, problem, pop, off, return_indices=False, **kwargs):

        idxs = self._do(problem, pop, off, **kwargs)

        if return_indices:
            return idxs
        else:
            pop[idxs] = off[idxs]
            return pop

    def _do(self, problem, pop, off, **kwargs):

        pass

