import numpy as np


class PolynomialMutation:
    def __init__(self, eta=20, prob=None):
        # Index parameter
        self.eta = eta

        # Probability of mutation
        self.prob = prob

    def _do(self, walk, bounds, **kwargs):
        # Initialise updated population variable array
        updated_walk = np.full(walk.shape, np.inf)

        # If probability of mutation is not set, set it equal to 1/n_var
        # if self.prob is None:
        #     self.prob = 1.0 / problem.n_var

        # Construct mutation mask
        do_mutation = np.random.random(walk.shape) < self.prob

        # Setting updated population variable array to be a copy of passed population variable array
        updated_walk[:, :] = walk

        # Tiling upper and lower bounds arrays across pop individuals and variables designated for mutation
        x_lower = bounds[0, :]
        x_upper = bounds[1, :]
        x_l = np.repeat(x_lower[np.newaxis, :], walk.shape[0], axis=0)[do_mutation]
        x_u = np.repeat(x_upper[np.newaxis, :], walk.shape[0], axis=0)[do_mutation]

        # Extracting variables designated for mutation from population variable array
        walk = walk[do_mutation]

        # Calculating delta arrays
        delta_1 = (walk - x_l) / (x_u - x_l)
        delta_2 = (x_u - walk) / (x_u - x_l)

        # Creating left/right mask
        rand = np.random.random(walk.shape)
        mask_left = rand <= 0.5
        mask_right = np.logical_not(mask_left)

        # Creating mutation delta array
        delta_q = np.zeros(walk.shape)

        # Mutation exponent
        mut_pow = 1.0 / (self.eta + 1.0)

        # Calculating left terms
        xy = 1.0 - delta_1
        val = 2.0 * rand + (1.0 - 2.0 * rand) * (np.power(xy, (self.eta + 1.0)))
        d = np.power(val, mut_pow) - 1.0
        delta_q[mask_left] = d[mask_left]

        # Calculating right terms
        xy = 1.0 - delta_2
        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (np.power(xy, (self.eta + 1.0)))
        d = 1.0 - (np.power(val, mut_pow))
        delta_q[mask_right] = d[mask_right]

        # Mutated values
        mutated_var = walk + delta_q * (x_u - x_l)

        # Enforcing bounds (floating point issues)
        mutated_var[mutated_var < x_l] = x_l[mutated_var < x_l]
        mutated_var[mutated_var > x_u] = x_u[mutated_var > x_u]

        # Set output variable array values
        updated_walk[do_mutation] = mutated_var

        return updated_walk
