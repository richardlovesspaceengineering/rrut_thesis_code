import numpy as np


class Sampling:

    def __init__(self):

        self.x = None

    def do(self, n_samples, x_lower, x_upper, seed=None):

        # Number of dimensions
        dim = x_lower.shape[0]

        # Sample design space
        self._do(dim, n_samples, seed)

        # Scaling sampling to fit design variable limits
        self.x = self.x*(x_upper - x_lower) + x_lower

    def _do(self, dim, n_samples, seed=None):
        pass


if __name__ == '__main__':

    from optimisation.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
    from optimisation.operators.sampling.random_sampling import RandomSampling

    # Number of sampling points
    num = 50

    x_lower = np.array([0.0, 0.0])
    x_upper = np.array([4.0, 3.0])

    sampling = LatinHypercubeSampling()
    sampling_2 = RandomSampling()

    sampling.do(num, x_lower, x_upper)
    sampling_2.do(num, x_lower, x_upper)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(sampling.x[:, 0], sampling.x[:, 1], 'o', color='C0')
    plt.plot(sampling_2.x[:, 0], sampling_2.x[:, 1], 'o', color='C1')
    plt.xlabel('x_0')
    plt.ylabel('x_1')
    plt.show()
    plt.close()

