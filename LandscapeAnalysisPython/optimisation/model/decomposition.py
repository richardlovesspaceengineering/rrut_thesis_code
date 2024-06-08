import numpy as np

from optimisation.util.misc import at_least_2d_array, to_1d_array_if_possible


class Decomposition:

    def __init__(self, eps=0.0, type='auto', **kwargs):
        self.eps = eps
        self.type = type

        self.ideal_point = None
        self.utopian_point = None
        self.nadir_point = None

    def do(self, obj_array, weights, type='auto', ideal_point=None, utopian_point=None, nadir_point=None, **kwargs):

        # Eliminate singular dimensions
        _obj_array = to_1d_array_if_possible(obj_array)
        _weights = to_1d_array_if_possible(weights)

        if type == 'auto':
            if _obj_array.ndim == 1 and _weights.ndim > 1:
                type = 'one_to_many'
            elif _obj_array.ndim > 1 and _weights.ndim == 1:
                type = 'many_to_one'
            elif _obj_array.ndim == 2 and _weights.ndim == 2 and (_obj_array.shape[0] == _weights.shape[0]):
                type = 'one_to_one'
            else:
                type = 'many_to_many'

        # Ensure arrays are at least 2d
        obj_array = np.atleast_2d(obj_array)
        weights = np.atleast_2d(weights)

        # Extract the number of points and weights
        n_points = obj_array.shape[0]
        n_weights = weights.shape[0]

        # Set ideal point default value
        self.ideal_point = ideal_point
        if self.ideal_point is None:
            self.ideal_point = np.zeros(obj_array.shape[1])

        # Set utopian point default value
        self.utopian_point = utopian_point
        if self.utopian_point is None:
            self.utopian_point = self.ideal_point - self.eps

        # Set nadir point default value
        self.nadir_point = nadir_point
        if self.nadir_point is None:
            self.nadir_point = self.utopian_point + np.ones(obj_array.shape[1])

        # Conduct decomposition
        if type == 'one_to_one':
            decomp = self._do(obj_array, weights=weights, **kwargs).flatten()
        elif type == 'one_to_many':
            obj_array = np.repeat(obj_array, n_weights, axis=0)
            decomp = self._do(obj_array, weights=weights, **kwargs).flatten()
        elif type == 'many_to_one':
            weights = np.repeat(weights, n_points, axis=0)
            decomp = self._do(obj_array, weights=weights, **kwargs).flatten()
        elif type == 'many_to_many':
            obj_array = np.repeat(obj_array, n_weights, axis=0)
            weights = np.tile(weights, (n_points, 1))
            decomp = self._do(obj_array, weights=weights, **kwargs).flatten()
        else:
            raise Exception('Unknown type for decomposition: %s' % type)

        return decomp

    def _do(self, obj_array, weights, **kwargs):
        pass



