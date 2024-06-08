import numpy as np
import math

from optimisation.model.infill import InfillCriterion


class Mating(InfillCriterion):

    def __init__(self,
                 selection,
                 crossover,
                 mutation,
                 **kwargs):

        super().__init__(**kwargs)
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation

    def _do(self, problem, pop, n_offspring, parents=None, **kwargs):

        if parents is None:

            # Number of parents to be selected for mating: depends on the number of offspring required
            n_select = math.ceil(n_offspring/self.crossover.n_offspring)

            # Select the parents for mating: produces an index array
            parents = self.selection.do(pop, n_select, self.crossover.n_parents, **kwargs)

        # Conduct crossover using the parents index and the population
        _offspring = self.crossover.do(problem, pop, parents)

        # Conduct mutation on the offspring created through crossover
        _offspring = self.mutation.do(problem, _offspring)

        return _offspring

