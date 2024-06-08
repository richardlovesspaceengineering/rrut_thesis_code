import numpy as np


class Mutation:

    def __init__(self):
        super().__init__()

    def do(self, problem, pop, **kwargs):

        # Extract the design variables from the passed population
        pop_var = pop.extract_var()

        # Conduct mutation
        updated_pop_var = self._do(problem, pop_var, **kwargs)

        # Extract the design variables from the passed population
        pop.assign_var(problem, updated_pop_var)

        return pop

    def _do(self, problem, pop_var, **kwargs):
        pass

