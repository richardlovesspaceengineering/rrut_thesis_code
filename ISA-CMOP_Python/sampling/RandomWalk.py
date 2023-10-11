# %%
from sampling.sampling import Sampling
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sampling.PolynomialMutation import PolynomialMutation


class RandomWalk(Sampling):
    def __init__(self, bounds, num_steps, step_size=0.2):
        # super().__init__()
        self.bounds = bounds
        self.num_steps = num_steps
        self.step_size = step_size
        self.check_step_size_prop()  # warn user if using too big a step size.

    def check_step_size_prop(self):
        if self.step_size > 0.2:
            message = "Warning: RandomWalk may result in an infinite loop for a step size greater than 0.02."
            warnings.warn(message)

    def random_pm(self):
        # correct implementation for a true RW
        # return 1 if np.random.random() < 0.5 else -1

        ## Alsouly's implementation gives closer-matching features.
        return 1

    def within_bounds(self, point, dim):
        return point <= self.bounds[1, dim] and point >= self.bounds[0, dim]

    def _do(self, seed=None):
        """
        Simulate the random walk in each dimension.
        """
        # Initialisation.
        dim = self.bounds.shape[1]  # dimensionality of the problem
        walk = np.zeros((self.num_steps, dim))  # array to store the walk
        x = np.zeros(dim)
        np.random.seed(seed)

        # First step of the random walk.
        for i in range(dim):
            x[i] = (
                self.bounds[0, i]
                + (self.bounds[1, i] - self.bounds[0, i]) * np.random.random()
            )

        # Save the first step.
        walk[0, :] = x

        # Continue the rest of the walk.
        for j in range(1, self.num_steps):
            curr = np.reshape(walk[j - 1, :], (1, -1))  # get previous step's position
            i = 0  # start from first dimension

            while i < dim:
                sign = self.random_pm()  # Determine positive or negative direction

                # Defines range of step sizes based on neighbourhood_size
                r = (
                    (self.bounds[0, i] * self.step_size)
                    + ((self.bounds[1, i] - self.bounds[0, i]) * self.step_size)
                ) * np.random.random()
                temp = curr[0, i] + r * sign

                # Handling if the walk leaves the bounds.
                if self.within_bounds(temp, i):
                    # Leave as is
                    s = temp

                    # Saving and iteration to the next dimension.
                    x[i] = s
                    i += 1
                else:
                    # Try and change direction.
                    temp = curr[0, i] - r * sign

                    # if we can't change direction and stay within the bounds, then try a different perturbation vector.
                    if self.within_bounds(temp, i):
                        # Leave as is
                        s = temp

                        # Saving and iteration to the next dimension.
                        x[i] = s
                        i += 1
                    else:
                        continue

            walk[j, :] = x

        return np.atleast_2d(walk)

    def generate_neighbours_for_walk(self, walk, neighbourhood_size):
        updated_walk = np.empty((neighbourhood_size * walk.shape[0], walk.shape[1]))

        array_idx_start = 0
        array_idx_end = walk.shape[0]
        for i in range(neighbourhood_size):
            curr = PolynomialMutation(prob=1)._do(walk, self.bounds)
            updated_walk[array_idx_start:array_idx_end, :] = curr

            # Shift index window.
            array_idx_start += walk.shape[0]
            array_idx_end += walk.shape[0]
        return updated_walk

    # def GenerateNeighbours(self):
    #     return PolynomialMutation(prob=1)._do()


if __name__ == "__main__":
    # Create a 4x4 grid of subplots
    fig, ax = plt.subplots(5, 4, figsize=(10, 10))
    for i in range(5):
        for j in range(4):
            # Define bounds for each subplot
            bounds = np.array([[-1, 0], [1, 3]])
            rw = RandomWalk(
                bounds,
                1000,
                0.2,
            )
            walk = rw._do(seed=None)

            new_walk = rw.generate_neighbours_for_walk(walk, 5)

            # Plot the random walk on the current subplot
            ax[i, j].plot(walk[:, 0], walk[:, 1])
            ax[i, j].scatter(new_walk[:, 0], new_walk[:, 1], s=2, color="red")

            # Add dotted lines at the bounds on the current subplot
            ax[i, j].axvline(
                x=bounds[0, 0], color="gray", linestyle="--", label="Lower X Bound"
            )
            ax[i, j].axvline(
                x=bounds[1, 0], color="gray", linestyle="--", label="Upper X Bound"
            )
            ax[i, j].axhline(
                y=bounds[0, 1], color="gray", linestyle="--", label="Lower Y Bound"
            )
            ax[i, j].axhline(
                y=bounds[1, 1], color="gray", linestyle="--", label="Upper Y Bound"
            )

    # Adjust subplot spacing
    plt.tight_layout()

    # Show the plot
    plt.show()

# %%
