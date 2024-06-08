import numpy as np

from optimisation.model.survival import Survival
from optimisation.util.non_dominated_sorting import NonDominatedSorting
from optimisation.util.rank_fronts import rank_fronts


class TwoRankingSurvival(Survival):

    def __init__(self, filter_infeasible=False):

        super().__init__(filter_infeasible=filter_infeasible)

        self.filter_infeasible = filter_infeasible

    def _do(self, problem, pop, n_survive, cons_val=None, gen=None, max_gen=None, **kwargs):

        # Extract the objective function values from the population
        obj_array = pop.extract_obj()

        # Fraction of population that is feasible
        feasible_fraction = np.count_nonzero((cons_val <= 0.0))/len(pop)

        # Conduct non-dominated sorting (considering constraints & objectives)
        fronts = NonDominatedSorting().do(obj_array, cons_val=cons_val, n_stop_if_ranked=len(pop))

        # Conduct non-dominated sorting (considering objectives only)
        obj_only_fronts = NonDominatedSorting().do(obj_array, n_stop_if_ranked=len(pop))

        # Rank fronts (considering constraints & objectives)
        rank_1 = rank_fronts(fronts, obj_array, pop, np.zeros(0, dtype=np.int))

        # Rank fronts (considering objectives only)
        rank_2 = rank_fronts(obj_only_fronts, obj_array, pop, np.zeros(0, dtype=np.int))

        # Generate composite rankings
        alpha = 0.5 + 0.5*feasible_fraction
        rank_composite = alpha*rank_1 + (1 - alpha)*rank_2

        # Survivors generated from sorted composite rank
        survivors = np.argsort(rank_composite)

        # If necessary, truncate number of survivors
        if len(survivors) > n_survive:
            survivors = survivors[:n_survive]

        return survivors
