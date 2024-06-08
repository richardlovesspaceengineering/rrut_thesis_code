import math
import numpy as np

from optimisation.model.population import Population
from optimisation.model.duplicate import NoDuplicateElimination


class InfillCriterion:

    def __init__(self,
                 eliminate_duplicates=None,
                 **kwargs):

        self.eliminate_duplicates = eliminate_duplicates if eliminate_duplicates is not None else NoDuplicateElimination()
        self.n_infill = 0

    def do(self, problem, pop, n_offspring, **kwargs):

        # Population object to be used
        offspring = Population(problem)

        # Infill counter: counts how often mating must be conducted to fill n_offspring entries
        self.n_infill = 0

        # Iterate until enough offspring are created
        while len(offspring) < n_offspring:

            # Remaining offspring required
            n_remaining = (n_offspring - len(offspring))

            # Conduct mating
            _offspring = self._do(problem, pop, n_remaining, **kwargs)

            # Eliminate duplicates
            _offspring = self.eliminate_duplicates.do(_offspring, pop, offspring)

            # If more offspring have been produced than the number required, truncate them randomly
            if len(offspring) + len(_offspring) > n_offspring:
                n_remaining = n_offspring - len(offspring)
                _offspring = _offspring[:n_remaining]

            # Add calculated offspring to the existing offspring and update the mating counter
            offspring = Population.merge(offspring, _offspring)
            self.n_infill += 1

        offspring.assign_var(problem, offspring.extract_var())

        return offspring

    def _do(self, problem, pop, n_offspring, **kwargs):
        pass
