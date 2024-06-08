import numpy as np

from optimisation.model.population import Population


class Crossover:

    def __init__(self, n_parents, n_offspring, prob=0.9):

        self.prob = prob
        self.n_parents = n_parents
        self.n_offspring = n_offspring

    def do(self, problem, pop, parents, **kwargs):

        if self.n_parents != parents.shape[1]:
            raise ValueError('Exception during crossover: Number of parents differs from defined at crossover.')

        # Extract the design variables from the parents (passed in as indices)
        parent_var = np.zeros((parents.shape[1], parents.shape[0], problem.n_var))
        for i in range(parents.shape[1]):
            for j in range(parents.shape[0]):
                parent_var[i, j, :] = pop[parents[j, i]].var

        # Creating crossover boolean mask using the crossover probability
        do_crossover = np.random.random(len(parents)) < self.prob

        # Compute the crossover
        offspring_var = self._do(problem, parent_var, **kwargs)

        # Applying crossover boolean mask
        parent_var[:, do_crossover, :] = offspring_var[:, do_crossover, :]

        # flatten the array to become a 2d-array
        parent_var = parent_var.reshape(-1, parent_var.shape[-1])

        # Create population object
        offspring = Population(problem, n_individuals=parent_var.shape[0])

        # Assign (now modified) parent var to offspring output
        offspring.assign_var(problem, parent_var)

        return offspring

    def _do(self, problem, parent_var, **kwargs):
        pass

