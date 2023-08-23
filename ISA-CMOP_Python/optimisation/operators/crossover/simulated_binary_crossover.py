import numpy as np

from optimisation.model.crossover import Crossover


class SimulatedBinaryCrossover(Crossover):

    def __init__(self, eta=15, n_offspring=2, prob_per_variable=0.5, **kwargs):

        super().__init__(n_parents=2, n_offspring=n_offspring, **kwargs)

        self.eta = eta
        self.prob_per_variable = prob_per_variable

    def _do(self, problem, parent_var, **kwargs):

        # Extracting number of matings and number of variables
        _, n_mating, n_var = parent_var.shape

        # Design variable lower and upper limits
        x_l = problem.x_lower
        x_u = problem.x_upper

        # Crossover boolean mask
        do_crossover = np.full(parent_var[0].shape, True)

        # Update mask using per-variable-probability
        do_crossover[np.random.random((n_mating, problem.n_var)) > self.prob_per_variable] = False

        # If variable values are too close to each other, no mating is conducted
        do_crossover[np.abs(parent_var[0] - parent_var[1]) <= 1e-14] = False

        # Assign the smaller values to y_1 and the larger values to y_2
        y_1 = np.min(parent_var, axis=0)
        y_2 = np.max(parent_var, axis=0)

        # Calculate random values for each individual
        rand = np.random.random((n_mating, problem.n_var))

        # Calculate the difference between all variables
        delta = (y_2 - y_1)

        # Ensure we are not dividing through by zero
        delta[delta < 1.0e-10] = 1.0e-10

        # Conducting binary crossover to construct children
        beta = 1.0 + (2.0*(y_1 - x_l)/delta)
        beta_q = self.calculate_beta_q(beta, rand)
        c_1 = 0.5*((y_1 + y_2) - beta_q*delta)

        beta = 1.0 + (2.0*(x_u - y_2)/delta)
        beta_q = self.calculate_beta_q(beta, rand)
        c_2 = 0.5*((y_1 + y_2) + beta_q*delta)

        # Construct mask to randomly swap variables
        b = np.random.random((n_mating, problem.n_var)) <= 0.5
        val = np.copy(c_1[b])
        c_1[b] = c_2[b]
        c_2[b] = val

        # Use parent variables as template for children
        children_var = np.copy(parent_var)

        # Copy the positions where crossover is conducted
        children_var[0, do_crossover] = c_1[do_crossover]
        children_var[1, do_crossover] = c_2[do_crossover]

        if self.n_offspring == 1:   # Randomly select one offspring
            children_var = children_var[np.random.choice(2, parent_var.shape[1]), np.arange(parent_var.shape[1])]
            children_var = children_var.reshape((1, parent_var.shape[1], parent_var.shape[2]))

        return children_var

    def calculate_beta_q(self, beta, rand):
        alpha = 2.0 - np.power(beta, -(self.eta + 1.0))

        mask, mask_not = (rand <= (1.0 / alpha)), (rand > (1.0 / alpha))

        beta_q = np.zeros(mask.shape)
        beta_q[mask] = np.power((rand*alpha), (1.0/(self.eta + 1.0)))[mask]
        beta_q[mask_not] = np.power((1.0/(2.0 - rand*alpha)), (1.0/(self.eta + 1.0)))[mask_not]

        return beta_q

