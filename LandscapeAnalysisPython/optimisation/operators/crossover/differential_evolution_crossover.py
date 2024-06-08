import numpy as np

from optimisation.model.crossover import Crossover
from optimisation.model.repair import InversePenaltyBoundsRepair


class DifferentialEvolutionCrossover(Crossover):

    def __init__(self, weight=0.8, dither=None, jitter=False, *args, **kwargs):

        super().__init__(3, 1, *args, **kwargs)

        self.weight = weight
        self.dither = dither
        self.jitter = jitter

    def _do(self, problem, parent_var, **kwargs):

        """
        Differential evolution algorithm
        Reference: https://doi.org/10.1109/SDE.2014.7031528
        """

        # Extracting number of matings and number of variables
        n_parents, n_mating, n_var = parent_var.shape

        # Dither
        if self.dither == 'vector':
            weight = (self.weight + np.random.random(n_mating)*(1.0 - self.weight))[:, None]
        elif self.dither == 'scalar':
            weight = self.weight + np.random.random()*(1.0 - self.weight)
        else:
            weight = self.weight

        # Jitter
        if self.jitter:
            gamma = 0.0001
            weight = (self.weight*(1.0 + gamma*(np.random.random(n_mating) - 0.5)))[:, None]

        # Generate children
        children_var = parent_var[0] + weight*(parent_var[1] - parent_var[2])
        children_var = InversePenaltyBoundsRepair().do(problem, children_var, parent_array=parent_var[0])

        return children_var[None, ...]
