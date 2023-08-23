import numpy as np

from optimisation.model.decomposition import Decomposition
from optimisation.operators.decomposition.util import calc_distance_to_weights


class PBI(Decomposition):

    def __init__(self, theta=5.0, **kwargs):

        super().__init__(**kwargs)
        self.theta = theta

    def _do(self, obj_array, weights, **kwargs):

        d1, d2 = calc_distance_to_weights(obj_array, weights, self.utopian_point)
        out = d1 + self.theta*d2

        return out

