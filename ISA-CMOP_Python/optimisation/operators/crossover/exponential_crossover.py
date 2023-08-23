import numpy as np

from optimisation.model.crossover import Crossover


class ExponentialCrossover(Crossover):

    def __init__(self, prob_exp=0.75, **kwargs):

        super().__init__(2, 2, **kwargs)

        self.prob_exp = prob_exp

    def _do(self, problem, parent_var, **kwargs):

        # Extracting number of matings and number of variables
        _, n_mating, n_var = parent_var.shape

        # Crossover mask
        mask = np.ones((n_mating, n_var), dtype=bool)

        # Crossover start point
        n = np.random.randint(0, n_var, size=n_mating)

        # Crossover probabilities
        r = np.random.random((n_mating, n_var)) < self.prob_exp

        # Edit crossover mask according to probabilities
        for i in range(n_mating):

            # Index where crossover begins
            start_idx = n[i]
            for j in range(n_var):

                # Current index
                current_idx = (start_idx + j) % n_var

                # Conduct crossover if random value is below prob_exp
                if r[i, current_idx]:
                    mask[i, current_idx] = True
                else:
                    break

        # Use parent variables as template for children
        children_var = np.copy(parent_var)

        # Conduct crossover
        children_var[0][mask] = parent_var[1][mask]
        children_var[1][mask] = parent_var[0][mask]

        return children_var
