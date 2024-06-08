import numpy as np

from optimisation.util.split_by_feasibility import split_by_feasibility


class Survival:

    def __init__(self, filter_infeasible=True):
        super().__init__()
        self.filter_infeasible = filter_infeasible

    def do(self, problem, pop, n_survive, gen=None, max_gen=None, **kwargs):

        # Extract the constraint function values from the population
        cons_array = pop.extract_cons_sum()

        # Initialise array of indices of surviving individuals
        survivors = np.zeros(0, dtype=np.int)

        # If population is to be filtered into feasible & infeasible
        if self.filter_infeasible and problem.n_con > 0:

            # Split population
            feasible, infeasible = split_by_feasibility(cons_array, sort_infeasible_by_cv=True)

            if len(feasible) == 1:
                # Domination-based survival methods can only function with > 1 feasible solution
                survivors = feasible
            elif len(feasible) > 1:
                # Calculate survivors using feasible solutions
                survival_idxs = self._do(problem, pop[feasible], min(len(feasible), n_survive), gen=gen, max_gen=max_gen, **kwargs)
                survivors = feasible[survival_idxs]

            # Check if infeasible solutions need to be added
            if len(survivors) < n_survive:
                least_infeasible = infeasible[:int(n_survive) - len(feasible)]
                survivors = np.concatenate((survivors, least_infeasible))

        else:   # Pass entire population (means constraint function sum array must be passed)

            # Converting constraints with no lower bound to be bounded on the lower end by zero (this prevents feasible
            # solutions with lower constraint values as being prioritised over other feasible solutions)
            cons_val = cons_array
            cons_val[cons_val <= 0.0] = 0.0

            # Calculate survivors
            survivors = self._do(problem, pop, n_survive, cons_val=cons_val, gen=gen, max_gen=max_gen, **kwargs)

        return pop[survivors]

    def _do(self, problem, pop, n_survive, gen=None, max_gen=None, **kwargs):
        pass



