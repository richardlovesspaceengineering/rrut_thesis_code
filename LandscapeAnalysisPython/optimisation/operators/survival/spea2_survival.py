import copy

import numpy as np

from optimisation.model.survival import Survival

from optimisation.util import dominator
from optimisation.util.hyperplane_normalisation import get_extreme_points
from optimisation.model.normalisation import Normalisation

from scipy.spatial.distance import cdist


class SPEA2Survival(Survival):

    def __init__(self, filter_infeasible=True, normalise=False, domination='pareto'):

        super().__init__(filter_infeasible=filter_infeasible)

        self.filter_infeasible = filter_infeasible
        self.normalise = normalise
        self.domination = domination
        self.norm = Normalisation()

    def _do(self, problem, pop, n_survive, cons_val=None, gen=None, max_gen=None, **kwargs):
        # Extract objectives
        obj_val = np.atleast_2d(pop.extract_obj())

        # Calculate domination matrix
        M = dominator.calculate_domination_matrix(obj_val, cons_val, domination_type=self.domination)

        # Number of solutions each individual dominates
        S = np.sum(M == 1, axis=0)

        # The raw fitness of each solution (strength of its dominators)
        R = np.sum(((M == -1) * S), axis=1)

        # Determine the k-th nearest neighbour
        k = int(np.sqrt(len(pop)))
        if k >= len(pop):
            k -= 1

        if self.normalise:
            _pop = self.norm.do(copy.deepcopy(pop), recalculate=True)
            _obj_val = np.atleast_2d(_pop.extract_obj())
        else:
            _obj_val = obj_val

        # Calculate distance matrix and sort by nearest neighbours
        dist_mat = cdist(_obj_val, _obj_val)
        np.fill_diagonal(dist_mat, np.inf)
        sorted_dist_mat = np.sort(dist_mat, axis=1)

        # Inverse distance metric
        D = 1.0 / (sorted_dist_mat[:, k] + 2.0)

        # SPEA2 fitness
        fitness = R + D

        # Select all non-dominated individuals
        non_dominated = list(np.where(np.all(M >= 0, axis=1))[0])
        remaining = [ind for ind in range(len(pop)) if ind not in non_dominated]

        if self.normalise:
            # Assign low value to extreme points, so as to make them highest priority in selection
            extreme_points = get_extreme_points(_obj_val, np.array([0.0, 0.0]))
            closest_to_ideal = np.argmin(cdist(extreme_points, _obj_val), axis=1)
            fitness[closest_to_ideal] = -1.0

        # Conduct final survival selection
        survived = non_dominated
        if len(survived) < n_survive:
            # Sort via fitness and assign missing individuals
            sorted_fitness = np.argsort(fitness[remaining])
            survived.extend(np.array(remaining)[sorted_fitness[:n_survive - len(survived)]].tolist())

        elif len(survived) > n_survive:
            # Remove one-by-one until n_survive is reached
            while len(survived) > n_survive:
                min_dist = np.min(dist_mat[survived][:, survived], axis=1)
                to_remove = np.argmin(min_dist)
                survived = [survived[i] for i in range(len(survived)) if i != to_remove]

        return survived
