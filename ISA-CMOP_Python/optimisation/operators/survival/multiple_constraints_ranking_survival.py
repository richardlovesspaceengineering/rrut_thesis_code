import numpy as np
import warnings

from optimisation.model.survival import Survival
from optimisation.util.non_dominated_sorting import NonDominatedSorting
from optimisation.util.rank_by_front_and_crowding import rank_by_front_and_crowding


class MultipleConstraintsRankingSurvival(Survival):
    """
    Selection procedure based on Ho and Shimizu Ranking
    Implemented based on dePaulaGarcia2017:
    A rank-based constraint handling technique for engineering design optimization problems solved by genetic algorithms
    Computers and Structures 187 (2017) 77-87
    Stored locally as dePaulaGarcia2017
    """

    def __init__(self, filter_infeasible=False):

        super().__init__(filter_infeasible=filter_infeasible)

        self.filter_infeasible = filter_infeasible

    def _do(self, problem, pop, n_survive, cons_val=None, gen=None, max_gen=None, **kwargs):

        if problem.n_con > 0:
            # Extract the total constraint violation from the population
            cons_array = pop.extract_cons_sum()
            cons_pos = cons_array
            cons_pos[cons_pos <= 0.0] = 0.0

            # Extract the constraint values from the population
            cons_values = pop.extract_cons()
            cons_values[cons_values <= 0.0] = 0.0

            cons_max = np.amax(np.concatenate((cons_values, np.zeros((1, cons_values.shape[1])))), axis=0)
            cons_max[cons_max == 0.0] = 1e-12

            # Extract the objective function values from the population
            obj_array = pop.extract_obj()

            # Extract the number of non-violated constraints
            nr_cons_violated = np.count_nonzero((cons_values > 0.0), axis=1)

            # Fraction of population that is feasible
            feasible_fraction = np.count_nonzero((cons_val <= 0.0)) / len(pop)

            # Conduct ranking based on objectives only
            obj_only_fronts = NonDominatedSorting().do(obj_array, n_stop_if_ranked=len(pop))
            rank_objective_value = self.rank_front_only(obj_only_fronts, (len(pop)))

            # Conduct ranking based on number of violated constraints
            nr_violated_cons_fronts = NonDominatedSorting().do(nr_cons_violated.reshape((len(pop), 1)), n_stop_if_ranked=len(pop))
            rank_nr_violated_cons = self.rank_front_only(nr_violated_cons_fronts, (len(pop)))

            # Conduct ranking for each constraint
            rank_for_cons = np.zeros(cons_values.shape)
            for cntr in range(problem.n_con):
                cons_to_be_ranked = cons_values[:, cntr]
                fronts_to_be_ranked = NonDominatedSorting().do(cons_to_be_ranked.reshape((len(pop), 1)), n_stop_if_ranked=len(pop))
                rank_for_cons[:, cntr] = self.rank_front_only(fronts_to_be_ranked, (len(pop)))
            rank_constraints = np.sum(rank_for_cons, axis=1)

            # Create the fitness function for the final ranking
            if feasible_fraction == 0.0:
                fitness_for_ranking = rank_constraints + rank_nr_violated_cons
            else:
                fitness_for_ranking = rank_constraints + rank_nr_violated_cons + rank_objective_value

        else:
            warnings.warn('you should not use this selection method if your problem is not constrained')
            fitness_for_ranking = rank_by_front_and_crowding(pop, n_survive, cons_val=None)

        # extract the survivors
        survivors = fitness_for_ranking.argsort()[:n_survive]
        return survivors

    @staticmethod
    def rank_front_only(fronts, n_survive):

        cntr_rank = 1
        rank = np.zeros(n_survive)
        for k, front in enumerate(fronts):

            # Save rank and crowding to the individuals
            for j, i in enumerate(front):
                rank[i] = cntr_rank

            cntr_rank += len(front)

        return rank
