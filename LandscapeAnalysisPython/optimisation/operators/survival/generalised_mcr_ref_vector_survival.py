import numpy as np
import warnings

from optimisation.model.survival import Survival
from optimisation.util.non_dominated_sorting import NonDominatedSorting
from optimisation.util.rank_by_front_and_crowding import rank_by_front_and_crowding
from optimisation.operators.survival.theta_survival import ThetaSurvival


class gMCRSurvival(Survival):
    """
    Selection procedure based on Ho and Shimizu Ranking
    Implemented based on dePaulaGarcia2017:
    A rank-based constraint handling technique for engineering design optimization problems solved by genetic algorithms
    Computers and Structures 187 (2017) 77-87
    Stored locally as dePaulaGarcia2017
    """

    def __init__(self, ref_dirs=None, filter_infeasible=False, use_generalised_mcr=False):

        super().__init__(filter_infeasible=filter_infeasible)

        self.filter_infeasible = filter_infeasible

        # Generalised MCR additions
        self.use_generalised_mcr = use_generalised_mcr
        self.beta1 = 1
        self.beta2 = 1
        self.eta = 1
        self.gamma = 1
        self.alpha = 1

        # Reference vector ranking
        if ref_dirs is not None:
            self.theta_survival = ThetaSurvival(ref_dirs=ref_dirs)

    def _do(self, problem, pop, n_survive, cons_val=None, gen=None, max_gen=None, **kwargs):

        if problem.n_con > 0:

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
            # TODO: Pareto/ Alpha - dominance ranking
            obj_only_fronts = NonDominatedSorting(domination='pareto').do(obj_array, n_stop_if_ranked=len(pop))
            rank_objective_value = self.rank_front_only(obj_only_fronts, (len(pop)))

            # TODO: Theta-dominance ranking
            # rank_objective_value = self.theta_survival._do(problem, pop, n_survive=len(pop), return_rank_only=True)

            # Conduct ranking based on number of violated constraints
            nr_violated_cons_fronts = NonDominatedSorting().do(nr_cons_violated.reshape((len(pop), 1)), n_stop_if_ranked=len(pop))
            rank_nr_violated_cons = self.rank_front_only(nr_violated_cons_fronts, (len(pop)))

            # Modify standard G-MCR coefficients if specified
            if self.use_generalised_mcr:
                zeta = feasible_fraction
                self.beta1 = np.sqrt(1 - (zeta - 1) ** 2)
                self.beta2 = 1 - self.beta1
                self.alpha = nr_cons_violated / len(pop)
                self.eta = 0
                self.gamma = 1 / np.sum(self.alpha)

            # Conduct ranking for each constraint
            rank_for_cons = np.zeros(cons_values.shape)
            for cntr in range(problem.n_con):
                cons_to_be_ranked = cons_values[:, cntr]
                fronts_to_be_ranked = NonDominatedSorting().do(cons_to_be_ranked.reshape((len(pop), 1)), n_stop_if_ranked=len(pop))
                rank_for_cons[:, cntr] = self.rank_front_only(fronts_to_be_ranked, (len(pop)))

            if self.use_generalised_mcr:
                rank_constraints = np.sum(self.alpha[:, None] * rank_for_cons, axis=1)
            else:
                rank_constraints = np.sum(rank_for_cons, axis=1)

            # Create the fitness function for the final ranking
            if feasible_fraction == 0.0:
                self.beta1 = 0.0
            fitness_for_ranking = self.beta1 * rank_objective_value + \
                                  self.beta2 * (self.eta * rank_nr_violated_cons + self.gamma * rank_constraints)

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
