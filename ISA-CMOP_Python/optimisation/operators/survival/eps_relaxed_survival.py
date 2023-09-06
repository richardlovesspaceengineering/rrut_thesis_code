import copy
import numpy as np

from optimisation.model.survival import Survival


class EpsRelaxedSurvival(Survival):

    def __init__(self, survival, eps0, evaluator, eps_frac=0.15, filter_infeasible=False):
        super().__init__(filter_infeasible=filter_infeasible)

        self.filter_infeasible = filter_infeasible
        
        # Eps-based parameters
        self.eps_frac = eps_frac
        self.eps0 = eps0 * self.eps_frac
        self.eps = self.eps0
        self.cp = 5.0
        
        # Base survival
        self._survival = survival
        self.evaluator = evaluator

    def _do(self, problem, pop, n_survive, cons_val=None, gen=None, max_gen=None, **kwargs):
        # Update eps value
        self.eps = self.reduce_boundary(self.eps0, n_gen=gen, max_gen=max_gen, cp=self.cp)

        # Transform the population
        eps_population = self.transform_population(copy.deepcopy(pop))
        max_abs_con_vals = self.evaluator.calc_max_abs_cons(eps_population, problem)
        eps_population = self.evaluator.sum_normalised_cons(eps_population, problem, max_abs_con_vals=max_abs_con_vals)

        # Conduct survival
        survivors = self._survival._do(problem, eps_population, n_survive, cons_val=eps_population.extract_cons_sum())

        return survivors

    def transform_population(self, population):
        # Create new objectives and transformed constraint
        old_cons = population.extract_cons()
        eps_cons_arr = copy.deepcopy(old_cons) - self.eps
        eps_cons_arr[eps_cons_arr <= 0.0] = 0.0

        # Update new population
        population.assign_cons(eps_cons_arr)

        return population

    @staticmethod
    def reduce_boundary(eps0, n_gen, max_gen, cp, delta=1e-8):
        A = eps0 + delta
        base_num = np.log((eps0 + delta) / delta)
        B = max_gen / np.power(base_num, 1 / cp) + 1e-15
        eps = A * np.exp(-(n_gen / B) ** cp) - delta
        eps[eps < 0.0] = 0.0

        return eps