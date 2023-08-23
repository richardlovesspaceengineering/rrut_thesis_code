import numpy as np

from optimisation.model.decomposition import Decomposition


class Chebychev(Decomposition):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _do(self, obj_array, weights, **kwargs):

        v = np.abs(obj_array - self.utopian_point)*weights
        chebychev_decomp = v.max(axis=1)

        return chebychev_decomp

