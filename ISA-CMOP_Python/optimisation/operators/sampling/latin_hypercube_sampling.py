from pyDOE2 import lhs

from optimisation.model.sampling import Sampling


class LatinHypercubeSampling(Sampling):

    def __init__(self,
                 criterion='maximin',
                 iterations=20):
        super().__init__()

        self.criterion = criterion
        self.iterations = iterations

    def _do(self, dim, n_samples, seed=None):

        self.x = lhs(dim, samples=n_samples, criterion=self.criterion, iterations=self.iterations, random_state=seed)
