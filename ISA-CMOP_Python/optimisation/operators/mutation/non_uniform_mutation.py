import numpy as np

from optimisation.model.mutation import Mutation


class NonUniformMutation(Mutation):

    def __init__(self, eta=20, prob=None, perturbation=0.5):

        super().__init__()

        # Index parameter
        self.eta = eta

        # Probability of mutation
        self.prob = prob

        # Perturbation of the mutation - based on jmetalpy
        self.perturbation = perturbation

    def _do(self, problem, pop_var, **kwargs):

        current_iteration = kwargs['current_iteration']
        max_iterations = kwargs['max_iterations']

        # Initialise updated population variable array
        updated_pop_var = np.full(pop_var.shape, np.inf)

        # If probability of mutation is not set, set it equal to 1/n_var
        if self.prob is None:
            self.prob = 1.0/problem.n_var


        # Tiling upper and lower bounds arrays across pop individuals and variables designated for mutation
        x_l = np.repeat(problem.x_lower[np.newaxis, :], pop_var.shape[0], axis=0)
        x_u = np.repeat(problem.x_upper[np.newaxis, :], pop_var.shape[0], axis=0)

        # Setting updated population variable array to be a copy of passed population variable array
        updated_pop_var[:, :] = pop_var

        # Construct mutation mask
        for i in range(problem.n_var):
            if np.random.random() <= self.prob:
                rand = np.random.random()

                if rand <= 0.5:
                    temp = self.__delta(x_u[:, i] - updated_pop_var[:, i], self.perturbation, current_iteration,
                                        max_iterations)
                else:
                    temp = self.__delta(x_l[:, i] - updated_pop_var[:, i], self.perturbation, current_iteration,
                                        max_iterations)
                updated_pop_var[:, i] += temp




        # Enforcing bounds (floating point issues)
        updated_pop_var[updated_pop_var < x_l] = x_l[updated_pop_var < x_l]
        updated_pop_var[updated_pop_var > x_u] = x_u[updated_pop_var > x_u]

        return updated_pop_var

    def __delta(self, y, b_mutation_parameter, current_iteration, max_iterations):
        return (y * (1.0 - np.power(np.random.random(),
                               np.power((1.0 - 1.0 * current_iteration / max_iterations), b_mutation_parameter))))
