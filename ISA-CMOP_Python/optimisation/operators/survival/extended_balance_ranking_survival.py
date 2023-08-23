import numpy as np
import warnings

from optimisation.model import survival
from optimisation.model.survival import Survival
from optimisation.util.rank_by_front_and_crowding import rank_by_front_and_crowding


class ExtendedBalancedRankingSurvival(Survival):
    """
    Extended balanced ranking survival method.
    Implemented based on the description given in:
    E-BRM: A constraint handling technique to solve optimization problems with evolutionary
    algorithms, Applied Soft Computing 72 (2018) 14?29
    Stored locally as Rodrigues2018.pdf
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

            # Extract the number of non-violated constraints
            nr_cons_not_violated = np.count_nonzero((cons_values == 0), axis=1)
            sigma = sum(nr_cons_not_violated) / (problem.n_con * len(pop))

            # Fraction of population that is feasible
            feasible_fraction = np.count_nonzero((cons_val <= 0.0)) / len(pop)

            # Calculate f_sigma and beta (equations 25 and 24 of paper)
            f_sigma = 0.0 if (feasible_fraction == 0.0) else sigma
            beta = 2.0 + (1.0 - f_sigma)

            # calculate constraint_penalty (equation 23 in paper)
            constraint_penalty = np.asarray(np.sum((cons_values ** beta), axis=1))

            # Filtering population indices by feasible and infeasible
            _, infeasible = survival.split_by_feasibility(constraint_penalty, sort_infeasible_by_cv=False)
            feasible, infeasible_sorted = survival.split_by_feasibility(constraint_penalty, sort_infeasible_by_cv=True)

            # Set up the 3 sorting ranks
            rank_1, rank_2, rank_3 = np.zeros(len(pop)), np.zeros(len(pop)), np.zeros(len(pop))

            # Rank feasible based on objective only (rank 1)
            rank_feasible_obj = rank_by_front_and_crowding(pop[feasible], len(feasible), cons_val=None)
            rank_feasible_obj = feasible[rank_feasible_obj]
            rank_1[rank_feasible_obj] = range(len(rank_feasible_obj))
            rank_1[rank_1 == 0] = np.nan

            # Rank infeasible based on objective only (rank 2)
            rank_infeasible_obj = rank_by_front_and_crowding(pop[infeasible], len(infeasible), cons_val=None)
            rank_infeasible_obj = infeasible[rank_infeasible_obj]
            rank_2[rank_infeasible_obj] = range(len(rank_infeasible_obj))
            rank_2[rank_2 == 0] = np.nan

            # Rank infeasible based on penalty constraint (rank 3)
            rank_3[infeasible_sorted] = range(len(infeasible_sorted))
            rank_3[rank_3 == 0] = np.nan

            rank_weighted = (rank_3*(1-f_sigma)+rank_2*f_sigma)/2
            sqrt_delta = np.sqrt((len(feasible)*len(infeasible))/len(pop))

            # Calculate the final rank
            if (len(infeasible)+sqrt_delta) <= len(feasible):
                phi_zero = len(feasible) - (len(infeasible) + sqrt_delta)
                psi_zero = phi_zero / (len(infeasible)-1) * (rank_3 -1)
            else:
                psi_zero = np.zeros(rank_3.shape)
            rank_final = np.zeros(len(pop))
            for idx in range(len(rank_feasible_obj)):
                rank_final[rank_feasible_obj[idx]] = rank_1[rank_feasible_obj[idx]]
            for idx in range(len(rank_infeasible_obj)):
                rank_final[rank_infeasible_obj[idx]] = rank_weighted[rank_infeasible_obj[idx]] + sqrt_delta + \
                                                       psi_zero[rank_infeasible_obj[idx]]
        else:
            warnings.warn('you should not use this selection method if your problem is not constrained')
            rank_final = rank_by_front_and_crowding(pop, n_survive, cons_val=None)

        # extract the survivors
        survivors = rank_final.argsort()[:n_survive]
        return survivors

