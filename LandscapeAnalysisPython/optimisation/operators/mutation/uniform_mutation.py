import numpy as np

from optimisation.model.mutation import Mutation


class UniformMutation(Mutation):

    def __init__(self, eta=20, prob=None, perturbation=0.5):

        super().__init__()

        # Index parameter
        self.eta = eta

        # Probability of mutation
        self.prob = prob

        # Perturbation of the mutation - based on jmetalpy
        self.perturbation = perturbation

    def _do(self, problem, pop_var, **kwargs):

        # Initialise updated population variable array
        updated_pop_var = np.full(pop_var.shape, np.inf)

        # If probability of mutation is not set, set it equal to 1/n_var
        if self.prob is None:
            self.prob = 1.0/problem.n_var

        # Construct mutation mask
        _rand_temp = np.random.random(pop_var.shape)
        do_mutation = _rand_temp < self.prob
        temp = (_rand_temp - 0.5)*self.perturbation

        # Setting updated population variable array to be a copy of passed population variable array
        updated_pop_var[:, :] = pop_var

        # Tiling upper and lower bounds arrays across pop individuals and variables designated for mutation
        x_l = np.repeat(problem.x_lower[np.newaxis, :], pop_var.shape[0], axis=0)[do_mutation]
        x_u = np.repeat(problem.x_upper[np.newaxis, :], pop_var.shape[0], axis=0)[do_mutation]

        # Extracting variables designated for mutation from population variable array
        pop_var = pop_var[do_mutation]

        # Mutated values
        mutated_var = pop_var + temp[do_mutation]*(x_u - x_l)

        # Enforcing bounds (floating point issues)
        mutated_var[mutated_var < x_l] = x_l[mutated_var < x_l]
        mutated_var[mutated_var > x_u] = x_u[mutated_var > x_u]

        # Set output variable array values
        updated_pop_var[do_mutation] = mutated_var

        return updated_pop_var

