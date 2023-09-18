# %%
from sampling import Sampling
import numpy as np
import matplotlib.pyplot as plt


class RandomWalk(Sampling):
    def __init__(self, bounds, num_steps, step_size):
        super().__init__()
        self.bounds = bounds
        self.num_steps = num_steps
        self.step_size = step_size

    def random_pm(self, seed=None):
        np.random.seed(seed)
        return 1 if np.random.random() < 0.5 else -1

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

                # Defines range of step sizes.
                r = (self.bounds[0, i] * self.step_size) + (
                    (self.bounds[1, i] - self.bounds[0, i]) * self.step_size
                ) * np.random.random()
                temp = curr[0, i] + r * sign

                # Handling if the walk leaves the bounds.
                if temp <= self.bounds[1, i] and temp >= self.bounds[0, i]:
                    # Leave as is
                    s = temp
                else:
                    # Otherwise change direction
                    s = curr[0, i] - r * sign

                # Saving and iteration to next dimension.
                x[i] = s
                i = i + 1

            walk[j, :] = x

        return walk


if __name__ == "__main__":
    # Create a 4x4 grid of subplots
    fig, ax = plt.subplots(4, 4, figsize=(10, 10))

    for i in range(4):
        for j in range(4):
            # Define bounds for each subplot
            bounds = np.array([[-1, 0], [1, 3]])
            rw = RandomWalk(bounds, 1000, 0.05)
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
