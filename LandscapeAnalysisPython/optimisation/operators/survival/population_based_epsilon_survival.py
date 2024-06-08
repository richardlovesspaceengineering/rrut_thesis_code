import numpy as np
import math

from optimisation.model.survival import Survival
from optimisation.util.split_by_feasibility import split_by_feasibility
from optimisation.util.rank_by_front_and_crowding import rank_by_front_and_crowding


class PopulationBasedEpsilonSurvival(Survival):

    def __init__(self, filter_infeasible=False, cp=6.0, theta_p=0.5):

        super().__init__(filter_infeasible=filter_infeasible)

        self.filter_infeasible = filter_infeasible

        self.cp = np.double(cp)
        self.theta_p = np.double(theta_p)

        self.gen = 0
        self.max_gen = 1
        self.c_gen = 0

        self.theta = 0
        self.epsilon = 0.0

    def _do(self, problem, pop, n_survive, cons_val=None, gen=None, max_gen=None, **kwargs):

        if gen is not None:
            self.gen = np.double(gen)
        if max_gen is not None:
            self.max_gen = np.double(max_gen)

        cons_val_modified = cons_val

        if problem.n_con > 0 and (type(self.gen) == np.double and type(self.max_gen) == np.double):

            # Calculate cut-off generation
            self.c_gen = 0.8*self.max_gen

            # Calculate theta
            self.theta = math.ceil(self.theta_p*len(pop)*(1.0 - self.gen/self.max_gen)**self.cp)

            # Sort solutions by constraint violation (should this only be infeasible solutions?)
            sorted_cons = np.sort(cons_val_modified)

            # Extract the constraint value of the theta-th solution when ranked by constraint violation from least to
            # most
            cons_theta = sorted_cons[self.theta]

            # Calculate epsilon
            if self.gen < self.c_gen:
                self.epsilon = cons_theta
            else:
                self.epsilon = 0.0

            # Re-classify solutions with cons_val below threshold as feasible
            cons_val_modified[cons_val_modified < self.epsilon] = 0.0

        # Initialise array of indices of surviving individuals
        survivors = np.zeros(0, dtype=np.int)

        # Split population
        feasible, infeasible = split_by_feasibility(cons_val_modified, sort_infeasible_by_cv=True)

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

