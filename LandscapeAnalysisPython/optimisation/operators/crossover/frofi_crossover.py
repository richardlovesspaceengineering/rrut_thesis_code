import numpy as np

from optimisation.model.crossover import Crossover
from optimisation.model.population import Population


class FROFICrossover(Crossover):

    def __init__(self, prob=0.5, f=0.3, **kwargs):

        super().__init__(1, 1, **kwargs)

        self.prob = prob
        self.f = f

    def do(self, problem, pop, parents, **kwargs):

        if 'opt' in kwargs:
            opt = kwargs['opt']
            del kwargs['opt']
        else:
            raise Exception('opt undefined: The optimum solution (ignoring constraints) must be passed')

        # Extract the design variables from the parents (passed in as indices)
        parent_var = np.zeros((parents.shape[1], parents.shape[0], problem.n_var))
        for i in range(parents.shape[1]):
            for j in range(parents.shape[0]):
                parent_var[i, j, :] = pop[parents[j, i]].var

        if parent_var.shape[0] == 1:
            parent_var = np.squeeze(parent_var)

        # Creating crossover boolean mask to determine type of crossover used for each parent
        r = np.random.random(len(parents))
        type_mask = r < self.prob

        # Compute the crossover
        offspring_var = self._do(problem, parent_var, type_mask=type_mask, r=r, f=self.f, opt=opt, **kwargs)

        # Create population object
        offspring = Population(problem, n_individuals=parent_var.shape[0])

        # Assign offspring var to offspring output
        offspring.assign_var(problem, offspring_var)

        return offspring, type_mask

    def _do(self, problem, parent_var, type_mask=None, r=None, f=0.3, opt=None, **kwargs):

        # Pre-allocate variable arrays of mutated individuals
        children_var = np.zeros(parent_var.shape)

        # Indices of randomly selected individuals from population
        rand_idx = np.random.randint(0, parent_var.shape[0]-1, 3)

        # Current-to-random
        children_var[type_mask, :] = parent_var[type_mask, :] \
                                     + np.tile(r[type_mask][:, None], (1, parent_var.shape[1]))*(np.tile(parent_var[rand_idx[0], :], (np.count_nonzero(type_mask), 1)) - parent_var[type_mask, :]) \
                                     + self.f*(np.tile(parent_var[rand_idx[1], :], (np.count_nonzero(type_mask), 1)) - np.tile(parent_var[rand_idx[2], :], (np.count_nonzero(type_mask), 1)))

        # Random to best
        children_var[~type_mask, :] = parent_var[~type_mask, :] \
                                      + np.tile(r[~type_mask][:, None], (1, parent_var.shape[1]))*(np.tile(opt.var, (np.count_nonzero(~type_mask), 1)) - np.tile(parent_var[rand_idx[0], :], (np.count_nonzero(~type_mask), 1))) \
                                      + self.f*(np.tile(parent_var[rand_idx[1], :], (np.count_nonzero(~type_mask), 1)) - np.tile(parent_var[rand_idx[2], :], (np.count_nonzero(~type_mask), 1)))

        return children_var
