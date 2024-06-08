import numpy as np
import random

from optimisation.model.individual import Individual
from optimisation.util.non_dominated_sorting import NonDominatedSorting


class Particle(Individual):

    def __init__(self, problem):

        super().__init__(problem)

        self.velocity = np.zeros(problem.n_var)
        self.optimum = Individual(problem)

    def compute_optimum(self, problem):

        obj_array = np.vstack((self.optimum.obj, self.obj))
        cons_array = np.array([[self.optimum.cons_sum], [self.cons_sum]])

        # Compute optimum position using non-dominated sort between current position and (current) optimum position
        _, rank = NonDominatedSorting().do(obj_array, cons_val=cons_array, return_rank=True)

        # If current position is better than optimum, replace optimum
        if rank[1] < rank[0]:
            self.optimum.set_var(problem, self.var)
            self.optimum.obj = self.obj
            self.optimum.cons = self.cons
            self.optimum.cons_sum = self.cons_sum

