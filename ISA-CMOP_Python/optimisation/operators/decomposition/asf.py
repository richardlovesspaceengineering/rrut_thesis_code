import numpy as np

from optimisation.model.decomposition import Decomposition


class ASF(Decomposition):

    def __init__(self, weight_0=1e-10, **kwargs):

        super().__init__(**kwargs)
        self.weight_0 = weight_0

    def _do(self, obj_array, weights, **kwargs):

        # Modify zero weights
        _weights = weights.copy()
        _weights[weights == 0.0] = self.weight_0

        out = np.max((obj_array - self.utopian_point)/_weights, axis=1)

        return out

