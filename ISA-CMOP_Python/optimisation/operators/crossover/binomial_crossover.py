import numpy as np

from optimisation.model.crossover import Crossover


class BiasedCrossover(Crossover):

    def __init__(self, bias, **kwargs):

        super().__init__(2, 1, **kwargs)

        self.bias = bias

    def _do(self, problem, parent_var, **kwargs):

        # Extracting number of matings and number of variables
        _, n_mating, n_var = parent_var.shape

        mask = np.random.random((n_mating, n_var)) < self.bias

        # Use parent variables as template for children
        children_var = np.copy(parent_var)

        # Conduct crossover
        children_var[0][mask] = parent_var[1][mask]
        children_var[1][mask] = parent_var[0][mask]

        return children_var

    def do(self, problem, pop, parents, **kwargs):

        offspring = super().do(problem, pop, parents, **kwargs)

        return offspring[:int(len(offspring)/2)]

