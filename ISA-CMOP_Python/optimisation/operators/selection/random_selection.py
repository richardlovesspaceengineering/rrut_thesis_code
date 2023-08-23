import math
import numpy as np

from optimisation.model.selection import Selection
from optimisation.util.misc import random_permutations


class RandomSelection(Selection):

    def __init__(self):

        super().__init__()

    def _do(self, pop, n_select, n_parents):

        # number of random individuals needed
        n_random = n_select * n_parents

        # number of permutations needed
        n_perms = math.ceil(n_random / len(pop))

        # get random permutations and reshape them
        permutations = random_permutations(n_perms, len(pop))[:n_random]

        return np.reshape(permutations, (n_select, n_parents))
