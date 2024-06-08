import numpy as np

from optimisation.model.mutation import Mutation


class NoMutation(Mutation):

    def _do(self, problem, pop_var, **kwargs):
        # used in OMOPSO
        return pop_var

