import numpy as np

from optimisation.model.survival import Survival
from optimisation.util.split_by_feasibility import split_by_feasibility
from optimisation.util.rank_by_front_and_crowding import rank_by_front_and_crowding


class SelfAdaptiveFeasibleRatioEpsilonSurvival(Survival):

    def __init__(self, filter_infeasible=False, cp=6.0):

        super().__init__(filter_infeasible=filter_infeasible)

        self.filter_infeasible = filter_infeasible

        self.cp = np.double(cp)

        self.gen = 0
        self.max_gen = 1

        self.sign_g = 0
        self.g_if = 0

        self.r_f = 0.0
        self.r_f_0 = 0.0
        self.r_d = 0.0

        self.phi_min = 0.0
        self.phi_max = 0.0

        self.epsilon_0 = 0.0
        self.epsilon = 0.0

    def _do(self, problem, pop, n_survive, cons_val=None, gen=None, max_gen=None, **kwargs):

        if gen is not None:
            self.gen = np.double(gen)
        if max_gen is not None:
            self.max_gen = np.double(max_gen)

        cons_val_modified = cons_val

        # Feasibility ratio (fraction of population that is feasible)
        self.r_f = np.count_nonzero((cons_val <= 0.0))/len(pop)
        if self.gen == 0:
            self.r_f_0 = self.r_f

        if problem.n_con > 0 and self.r_f < 1.0 and (type(self.gen) == np.double and type(self.max_gen) == np.double):

            # Calculate minimum and maximum constraint violations (sums) for infeasible individuals
            infeasible_cons_array = cons_val[cons_val > 0.0]
            self.phi_min = np.amin(infeasible_cons_array)
            self.phi_max = np.amax(infeasible_cons_array)

            # Calculate r_d
            self.r_d = self.r_f_0 + (1.0 - self.r_f_0)*(self.gen/self.max_gen)

            if self.r_f == 0.0:
                # Calculate epsilon_0
                if self.epsilon < self.phi_min:
                    self.epsilon_0 = self.phi_min
                elif self.epsilon > self.phi_max:
                    self.epsilon_0 = self.phi_max
                else:
                    self.epsilon_0 = self.epsilon_0

                # Calculate sign_g
                if self.gen == 0:
                    self.sign_g = 0.0
                else:
                    if self.epsilon < self.phi_min:
                        self.sign_g = 1.0
                    elif self.epsilon > self.phi_max:
                        self.sign_g = 0.0
                    else:
                        self.sign_g = self.sign_g

                # Calculate g_if
                if self.gen == 0:
                    self.g_if = 0.0
                else:
                    if self.sign_g == 1.0:
                        self.g_if = self.gen
                    elif self.sign_g == 0.0:
                        self.g_if = self.g_if
            else:
                # Calculate epsilon_0
                if self.r_f < self.r_d:
                    self.epsilon_0 = self.epsilon_0
                else:
                    if self.epsilon < self.phi_min:
                        self.epsilon_0 = (self.r_f/self.r_d)*self.phi_min
                    else:
                        self.epsilon_0 = self.epsilon_0

                # Calculate g_if
                if self.r_f < self.r_d:
                    self.g_if = self.g_if
                else:
                    self.g_if = self.gen

            # Calculate epsilon
            self.epsilon = self.epsilon_0*(1.0 - (self.gen - self.g_if)/(self.max_gen - self.g_if))**self.cp

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
