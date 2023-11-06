from pyDOE2 import lhs
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import qmc



from optimisation.model.sampling import Sampling


class LatinHypercubeSampling(Sampling):
    def __init__(self, criterion="maximin", iterations=10000, method="pyDOE"):
        super().__init__()
        self.criterion = criterion
        self.iterations = iterations
        self.method = method

    def _do(self, dim, n_samples, seed=None):
        if self.method == "pyDOE":
            self.x = lhs(
                dim,
                samples=n_samples,
                criterion=self.criterion,
                iterations=self.iterations,
                random_state=seed,
            )
            
        elif self.method == "modified":
            self.x = self._modified_lhs_maximin(dim, n_samples, seed)
        elif self.method == "scipy":
            if self.criterion == "maximin":
                scramble = True
            else:
                scramble = False

            self.x = qmc.LatinHypercube(
                dim,
                scramble=True,
                strength=1,
                optimization=None,
                seed=None,
            ).random(n=n_samples)
            print("Discrepancy: {:.6f}".format(qmc.discrepancy(self.x)))
            

    def _modified_lhs_maximin(self, dim, n_samples, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # Ali's implementation of LHS maximin
        R = dim**0.5
        n = 0
        X = np.zeros((n_samples, dim))
        while n < n_samples:
            x = np.random.rand(
                1,
                dim,
            )
            if n == 0:
                X[n, :] = x
                n += 1
            else:
                dis = cdist(X[0:n], x, metric="euclidean")
                if np.min(dis) > R:
                    X[n, :] = x
                    n += 1
                else:
                    R = R / 1.001

        return X


if __name__ == "__main__":
    import numpy as np
    from scipy.spatial.distance import cdist

    dim = 2
    N = 50

    lhs_obj = LatinHypercubeSampling(criterion="centermaximin")
    lhs_obj._do(dim=dim, n_samples=N)
    x_lhs = lhs_obj.x

    lhs_obj = LatinHypercubeSampling(criterion="maximin")
    lhs_obj._do(dim=dim, n_samples=N)
    x_lhs1 = lhs_obj.x

    # Ali's LHS maximin
    x_lhs2 = lhs_obj._modified_lhs_maximin(dim, N)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.scatter(x_lhs[:, 0], x_lhs[:, 1], c="k", label="pydoe2-maximin")
    ax.scatter(x_lhs1[:, 0], x_lhs1[:, 1], c="r", label="pydoe2-centermaximin")
    ax.scatter(x_lhs2[:, 0], x_lhs2[:, 1], c="b", label="Ali-maximin")
    fig.legend()
    plt.show()
