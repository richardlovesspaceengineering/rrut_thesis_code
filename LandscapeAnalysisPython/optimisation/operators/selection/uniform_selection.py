import numpy as np

from optimisation.model.selection import Selection


class UniformSelection(Selection):

    def __init__(self):

        super().__init__()

    def _do(self, pop, n_select, n_parents):

        # number of individuals needed
        n_required = n_select * n_parents

        # Generate indices
        selections = range(n_required)

        return np.reshape(selections, (n_select, n_parents))

