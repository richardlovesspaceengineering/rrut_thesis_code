from sampling.Sampling import Sampling
import numpy as np

# import matplotlib.pyplot as plt


class RandomSample(Sampling):
    def __init__(self, bounds, num_samples):
        super().__init__()
        self.bounds = bounds
        self.num_samples = num_samples

    def _do(self, seed=None):
        """
        Take num_samples random samples for each coordinate.
        """
        # Dimensionality of the problem.
        dim = self.bounds.shape[1]  # dimensionality of the problem

        # Extract the bounds
        x_lower = self.bounds[0, :]
        x_upper = self.bounds[1, :]

        np.random.seed(seed)
        x = np.random.uniform(x_lower, x_upper, size=(self.num_samples, dim))

        return x


if __name__ == "__main__":
    # Create a 4x4 grid of subplots
    fig, ax = plt.subplots(4, 4, figsize=(10, 10))

    for i in range(4):
        for j in range(4):
            # Define bounds for each subplot
            bounds = np.array([[-1, 0], [1, 3]])
            samp = RandomSample(bounds, 1000)
            sample = samp._do(seed=None)

            # Plot the random sample on the current subplot
            ax[i, j].scatter(sample[:, 0], sample[:, 1], s=2)

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
