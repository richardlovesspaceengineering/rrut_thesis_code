import numpy as np

from optimisation.model.survival import Survival
from optimisation.util.split_by_feasibility import split_by_feasibility
from optimisation.util.rank_by_front_and_crowding import rank_by_front_and_crowding


class FeasibleRatioEpsilonSurvival(Survival):

    def __init__(self, filter_infeasible=False):

        super().__init__(filter_infeasible=filter_infeasible)

        self.filter_infeasible = filter_infeasible

    def _do(self, problem, pop, n_survive, cons_val=None, gen=None, max_gen=None, **kwargs):

        cons_val_scaled = cons_val

        if problem.n_con > 0:

            # Extract the constraint function values from the population
            cons_array = pop.extract_cons()
            cons_pos = cons_array
            cons_pos[cons_pos <= 0.0] = 0.0

            # Fraction of population that is feasible
            feasible_fraction = np.count_nonzero((cons_val <= 0.0))/len(pop)

            # Find the maximum violation across each of the constraints
            cons_max = np.amax(np.concatenate((cons_array, np.zeros((1, cons_array.shape[1])))), axis=0)
            cons_max[cons_max == 0.0] = 1e-12

            # Scale constraints by the maximum of each constraint
            cons_scaled = cons_pos/cons_max
            cons_val_scaled = np.sum(cons_scaled, axis=1)

            # Mean constraint value
            mean_cv = np.mean(cons_val_scaled)

            # Re-classify solutions with cons_val below threshold as feasible
            cons_val_scaled[cons_val_scaled < feasible_fraction*mean_cv] = 0.0

        # Initialise array of indices of surviving individuals
        survivors = np.zeros(0, dtype=np.int)

        # Split population
        feasible, infeasible = split_by_feasibility(cons_val_scaled, sort_infeasible_by_cv=True)

        # If feasible solutions exist
        if len(feasible) > 0:
            # Calculate survivors using feasible solutions
            survival_idxs = rank_by_front_and_crowding(pop[feasible], min(len(feasible), n_survive), **kwargs)
            survivors = feasible[survival_idxs]

        # Check if infeasible solutions need to be added
        if len(survivors) < n_survive:
            least_infeasible = infeasible[:n_survive - len(feasible)]
            survivors = np.concatenate((survivors, least_infeasible))

        return survivors
