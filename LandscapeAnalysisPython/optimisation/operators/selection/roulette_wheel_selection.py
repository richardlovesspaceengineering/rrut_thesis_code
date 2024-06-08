import math
import numpy as np

from optimisation.model.selection import Selection


class RouletteWheelSelection(Selection):

    def __init__(self):

        super().__init__()

    def _do(self, pop, n_select, n_parents, probabilities=None):

        if probabilities is None:
            raise Exception('Probabilities must be passed')
        elif len(probabilities) != len(pop):
            raise Exception('Probability array must be of same length as population array')
        else:
            # Number of random individuals needed
            n_random = n_select * n_parents

            # Compute random number array
            r = np.random.random(n_random)

            # Compute cumulative probabilities (essentially CDF)
            cdf = np.cumsum(probabilities)

            # Select individuals based on roulette wheel
            selected = np.zeros(n_random, dtype=int)
            for i in range(n_random):
                selected[i] = wheel_region(cdf, r[i])

            return np.reshape(selected, (n_select, n_parents))


def wheel_region(cfd, r):

    if r <= cfd[0]:
        return 0
    else:
        for i in range(1, len(cfd)):
            if cfd[i-1] <= r <= cfd[i]:
                return i


