# %%
from sampling.Sampling import Sampling
import numpy as np
import matplotlib.pyplot as plt
import warnings


class RandomWalk(Sampling):
    def __init__(self, bounds, num_steps, step_size_prop, neighbourhood_size):
        # super().__init__()
        self.bounds = bounds
        self.num_steps = num_steps
        self.step_size_prop = step_size_prop
        self.check_step_size_prop()  # warn user if using too big a step size.
        self.neighbourhood_size = neighbourhood_size

    def check_step_size_prop(self):
        if self.step_size_prop > 0.02:
            message = "Warning: RandomWalk may result in an infinite loop for a step size greater than 0.02."
            warnings.warn(message)

    def random_pm(self, seed=None):
        np.random.seed(seed)
        return 1 if np.random.random() < 0.5 else -1

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
                + (self.bounds[1, i] - self.bounds[0, 1]) * np.random.random()
            )

        # Save the first step.
        walk[0, :] = x

        # Continue the rest of the walk.
        for j in range(1, self.num_steps):
            curr = np.reshape(walk[j - 1, :], (1, -1))  # get previous step's position
            i = 0  # start from first dimension

            while i < dim:
                sign = self.random_pm(seed)  # Determine positive or negative direction

                # Define the step size for this dimension.
                step_size = self.step_size_prop * (
                    self.bounds[1, i] - self.bounds[0, i]
                )

                # Defines range of step sizes based on neighbourhood_size
                r = (
                    (self.bounds[0, i] * step_size)
                    + ((self.bounds[1, i] - self.bounds[0, i]) * step_size)
                ) * np.random.random(size=(self.neighbourhood_size, 1))
                temp = curr[0, i] + np.sum(r) * sign

                # Handling if the walk leaves the bounds.
                if self.within_bounds(temp, i):
                    # Leave as is
                    s = temp

                    # Saving and iteration to the next dimension.
                    x[i] = s
                    i += 1
                else:
                    # Try and change direction.
                    temp = curr[0, i] - np.sum(r) * sign

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

        return walk


if __name__ == "__main__":
    # Create a 4x4 grid of subplots
    fig, ax = plt.subplots(5, 4, figsize=(10, 10))
    neighbourhood_sizes = [1, 2, 3, 4, 5]
    for i in range(5):
        for j in range(4):
            # Define bounds for each subplot
            bounds = np.array([[-1, 0], [1, 3]])
            rw = RandomWalk(
                bounds, 1000, 0.02, neighbourhood_size=neighbourhood_sizes[i]
            )
            walk = rw._do(seed=None)

            # Plot the random walk on the current subplot
            ax[i, j].plot(walk[:, 0], walk[:, 1])

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
