import abc
import numpy as np

from optimisation.model.population import Population
from optimisation.model.swarm import Swarm


class Repair:

    def do(self, problem, pop_or_var, **kwargs):
        return self._do(problem, pop_or_var, **kwargs)

    def _do(self, problem, pop_or_var, **kwargs):
        pass


class BoundsRepair(Repair):

    def _do(self, problem, pop_or_var, **kwargs):

        # Determine whether population or variable array is passed
        is_var_array = not (isinstance(pop_or_var, Population) or isinstance(pop_or_var, Swarm))

        # Continue if variable array is passed, otherwise extract it
        var_array = pop_or_var if is_var_array else pop_or_var.extract_var()

        # Conduct bounds repair
        var_array = self.repair_out_of_bounds(problem, var_array, **kwargs)

        # Return the relevant data
        if is_var_array:
            return var_array
        else:
            pop_or_var.assign_var(problem, var_array)
            return pop_or_var

    @abc.abstractmethod
    def repair_out_of_bounds(self, problem, var_array, **kwargs):
        pass


class BasicBoundsRepair(BoundsRepair):

    def repair_out_of_bounds(self, problem, var_array, **kwargs):

        # Upper and lower bounds masks
        upper_mask = var_array > problem.x_upper
        lower_mask = var_array < problem.x_lower

        # Repair variables lying outside bounds
        var_array[upper_mask] = np.tile(problem.x_upper, (len(var_array), 1))[upper_mask]
        var_array[lower_mask] = np.tile(problem.x_lower, (len(var_array), 1))[lower_mask]

        return var_array


class BounceBackBoundsRepair(BoundsRepair):

    def repair_out_of_bounds(self, problem, var_array, **kwargs):

        # Variable ranges
        x_range = problem.x_upper - problem.x_lower

        # Upper and lower bounds masks
        upper_mask = var_array > problem.x_upper
        lower_mask = var_array < problem.x_lower

        # Repair variables lying outside bounds
        var_array[upper_mask] = (problem.x_upper - np.mod(var_array - problem.x_upper, x_range))[upper_mask]
        var_array[lower_mask] = (problem.x_lower + np.mod(problem.x_lower - var_array, x_range))[lower_mask]

        return var_array


class InversePenaltyBoundsRepair(BoundsRepair):

    def repair_out_of_bounds(self, problem, var_array, parent_array=None, **kwargs):

        # Ensure parent is passed
        if parent_array is None:
            raise Exception('A parent must be passed to this method')

        # Ensure parent and variable arrays are of same size
        assert len(var_array) == len(parent_array)

        # Conduct bounds handling method
        for i in range(len(var_array)):
            var_array[i, :] = inverse_penalty(var_array[i, :], parent_array[i, :], problem.x_lower, problem.x_upper)

        return var_array


def inverse_penalty(x, p, x_lower, x_upper, alpha=None):

    # Determine euclidian distance between variable array and parent array
    norm_v = np.linalg.norm(p - x)

    # Determine violated bounds
    lower_bounds_mask = x < x_lower
    upper_bounds_mask = x > x_upper

    # If no bounds are violated, return the passed variable vector
    if not np.any(np.logical_or(lower_bounds_mask, upper_bounds_mask)):
        return x
    else:

        # Calculate lower bound on y
        diff = p - x
        diff[diff == 0.0] = 1e-32
        d_1 = norm_v*np.max(np.maximum(lower_bounds_mask*(x_lower - x)/diff,
                                       upper_bounds_mask*(x_upper - x)/diff))

        # Calculate upper bound on Y
        temp = np.array([~lower_bounds_mask*((x_lower - x)/diff), ~upper_bounds_mask*((x_upper - x)/diff)])
        d_2 = norm_v*np.min(temp[temp > 0.0])

        if alpha is None:
            alpha = (norm_v - d_1)/norm_v
            alpha += 1e-32

        r = np.random.random()
        if r > 0.0:
            y = d_1*(1.0 + alpha*np.tan(r + np.arctan((d_2 - d_1)/(alpha*d_1))))
        else:
            y = d_1

        # Compute repaired vector
        x_repaired = x + (p - x) * y/norm_v

        # Check bounds (possibility of floating point error)
        x_repaired[x_repaired < x_lower] = x_lower[x_repaired < x_lower]
        x_repaired[x_repaired > x_upper] = x_upper[x_repaired > x_upper]

        return x_repaired

