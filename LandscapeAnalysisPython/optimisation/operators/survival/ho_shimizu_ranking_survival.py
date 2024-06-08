import numpy as np
import warnings

from optimisation.model import survival
from optimisation.model.survival import Survival
from optimisation.util.non_dominated_sorting import NonDominatedSorting
from optimisation.util.rank_fronts import rank_fronts
from optimisation.util.rank_by_front_and_crowding import rank_by_front_and_crowding


class HoShimizuRankingSurvival(Survival):
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

            # Scale constraints by the maximum of each constraint
            cons_scaled = (cons_values/cons_max)**2

            # Extract the objective function values from the population
            obj_array = pop.extract_obj()

            # Extract the number of non-violated constraints
            nr_cons_violated = np.count_nonzero((cons_values > 0.0), axis=1)

            # Fraction of population that is feasible
            feasible_fraction = np.count_nonzero((cons_val <= 0.0)) / len(pop)

            # Filtering population indices by feasible and infeasible and sort based on number of constraints violated
            feasible, infeasible = survival.split_by_feasibility(nr_cons_violated, sort_infeasible_by_cv=False)

            # Conduct ranking based on objectives only
            obj_only_fronts = NonDominatedSorting().do(obj_array, n_stop_if_ranked=len(pop))
            rank_objective_value = rank_fronts(obj_only_fronts, obj_array, pop, np.zeros(0, dtype=np.int))
            rank_objective_value += 1

            # Conduct ranking based on number of violated constraints
            nr_violated_cons_fronts = NonDominatedSorting().do(nr_cons_violated.reshape((len(pop), 1)), n_stop_if_ranked=len(pop))
            rank_nr_violated_cons = rank_fronts(nr_violated_cons_fronts, obj_array, pop, np.zeros(0, dtype=np.int))
            rank_nr_violated_cons += 1

            # Conduct ranking based on squared sum of the constraint violations
            scaled_cons_fronts = NonDominatedSorting().do(cons_scaled, n_stop_if_ranked=len(pop))
            rank_scaled_cons = rank_fronts(scaled_cons_fronts, obj_array, pop, np.zeros(0, dtype=np.int))
            rank_scaled_cons += 1

            # Create the fitness function for the final ranking
            fitness_for_ranking = np.zeros(rank_objective_value.shape)
            if feasible_fraction == 0.0:
                fitness_for_ranking = rank_scaled_cons + rank_nr_violated_cons
            elif feasible_fraction == 1.0:
                fitness_for_ranking = rank_objective_value
            else:
                for idx in range(len(pop)):
                    if idx in feasible:
                        fitness_for_ranking[idx] = rank_objective_value[idx]+2
                    else:
                        fitness_for_ranking[idx] = rank_scaled_cons[idx] + rank_nr_violated_cons[idx] + \
                                                   rank_objective_value[idx]

        else:
            warnings.warn('you should not use this selection method if your problem is not constrained')
            fitness_for_ranking = rank_by_front_and_crowding(pop, n_survive, cons_val=None)

        # extract the survivors
        survivors = fitness_for_ranking.argsort()[:n_survive]
        return survivors
