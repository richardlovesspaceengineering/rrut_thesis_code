import numpy as np

from optimisation.model.survival import Survival
from optimisation.util.rank_by_front_and_crowding import rank_by_front_and_crowding


class RankAndCrowdingSurvival(Survival):

    def __init__(self, filter_infeasible=True):

        super().__init__(filter_infeasible=filter_infeasible)

        self.filter_infeasible = filter_infeasible

    def _do(self, problem, pop, n_survive, cons_val=None, gen=None, max_gen=None, **kwargs):

        survivors = rank_by_front_and_crowding(pop, n_survive, cons_val)

        return survivors
