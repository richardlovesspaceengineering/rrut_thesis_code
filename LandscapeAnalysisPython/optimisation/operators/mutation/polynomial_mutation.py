import numpy as np

from optimisation.model.mutation import Mutation


class PolynomialMutation(Mutation):

    def __init__(self, eta=20, prob=None):

        super().__init__()

        # Index parameter
        self.eta = eta

        # Probability of mutation
        self.prob = prob

    def _do(self, problem, pop_var, **kwargs):

        # Initialise updated population variable array
        updated_pop_var = np.full(pop_var.shape, np.inf)

        # If probability of mutation is not set, set it equal to 1/n_var
        if self.prob is None:
            self.prob = 1.0/problem.n_var

        # Construct mutation mask
        do_mutation = np.random.random(pop_var.shape) < self.prob

        # Setting updated population variable array to be a copy of passed population variable array
        updated_pop_var[:, :] = pop_var

        # Tiling upper and lower bounds arrays across pop individuals and variables designated for mutation
        x_l = np.repeat(problem.x_lower[np.newaxis, :], pop_var.shape[0], axis=0)[do_mutation]
        x_u = np.repeat(problem.x_upper[np.newaxis, :], pop_var.shape[0], axis=0)[do_mutation]

        # Extracting variables designated for mutation from population variable array
        pop_var = pop_var[do_mutation]

        # Calculating delta arrays
        delta_1 = (pop_var - x_l)/(x_u - x_l)
        delta_2 = (x_u - pop_var)/(x_u - x_l)

        # Creating left/right mask
        rand = np.random.random(pop_var.shape)
        mask_left = rand <= 0.5
        mask_right = np.logical_not(mask_left)

        # Creating mutation delta array
        delta_q = np.zeros(pop_var.shape)

        # Mutation exponent
        mut_pow = 1.0 / (self.eta + 1.0)

        # Calculating left terms
        xy = 1.0 - delta_1
        val = 2.0*rand + (1.0 - 2.0*rand)*(np.power(xy, (self.eta + 1.0)))
        d = np.power(val, mut_pow) - 1.0
        delta_q[mask_left] = d[mask_left]

        # Calculating right terms
        xy = 1.0 - delta_2
        val = 2.0*(1.0 - rand) + 2.0*(rand - 0.5)*(np.power(xy, (self.eta + 1.0)))
        d = 1.0 - (np.power(val, mut_pow))
        delta_q[mask_right] = d[mask_right]

        # Mutated values
        mutated_var = pop_var + delta_q*(x_u - x_l)

        # Enforcing bounds (floating point issues)
        mutated_var[mutated_var < x_l] = x_l[mutated_var < x_l]
        mutated_var[mutated_var > x_u] = x_u[mutated_var > x_u]

        # Set output variable array values
        updated_pop_var[do_mutation] = mutated_var

        return updated_pop_var

