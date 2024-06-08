import numpy as np
import copy

from optimisation.algorithms.evolutionary_algorithm import EvolutionaryAlgorithm

from optimisation.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from optimisation.operators.selection.uniform_selection import UniformSelection
from optimisation.operators.crossover.frofi_crossover import FROFICrossover
from optimisation.operators.crossover.binomial_crossover import BiasedCrossover

from optimisation.model.population import Population
from optimisation.model.repair import BounceBackBoundsRepair
from optimisation.operators.replacement.frofi_replacement import FROFIReplacement
from optimisation.util.rank_by_front_and_crowding import rank_by_front_and_crowding


class DifferentialEvolutionFROFI(EvolutionaryAlgorithm):

    def __init__(self,
                 n_population=100,
                 sampling=LatinHypercubeSampling(),
                 selection=UniformSelection(),
                 crossover=FROFICrossover(prob=0.5, f=0.3),
                 mutation=BiasedCrossover(bias=0.5),
                 m=4,
                 **kwargs):

        # Subdivisions of population for archive replacement
        self.m = m

        super().__init__(n_population=n_population,
                         sampling=sampling,
                         selection=selection,
                         crossover=crossover,
                         mutation=mutation,
                         survival=None,
                         n_offspring=n_population,
                         **kwargs)

    def _next(self):

        # Conduct selection
        parents = self.mating.selection.do(self.population, self.n_population, 1)

        # Conduct crossover
        _offspring, type_mask = self.mating.crossover.do(self.problem, self.population, parents, opt=self.opt)

        # Conduct mutation (which is actually binomial crossover on ~crossover_type)
        self.offspring = copy.copy(_offspring)
        _population = Population.merge(self.population[~type_mask], _offspring[~type_mask])
        _parents = np.column_stack([np.arange(np.count_nonzero(~type_mask)),
                                    np.arange(np.count_nonzero(~type_mask)) + np.count_nonzero(~type_mask)])
        self.offspring[~type_mask] = self.mating.mutation.do(self.problem, _population, _parents)[:np.count_nonzero(~type_mask)]

        # Repair variables outside bounds
        self.offspring = BounceBackBoundsRepair().do(self.problem, self.offspring)

        # Evaluate offspring
        self.evaluator.do(self.problem.obj_func, self.problem, self.offspring, self.max_abs_con_vals)

        # Conduct replacement via domination & archive (this is effectively the survival method)
        self.population = FROFIReplacement().do(self.problem, self.population, self.offspring, m=self.m)

        # Update maximum constraint values across the population
        self.max_abs_con_vals = self.evaluator.calc_max_abs_cons(self.population, self.problem)

        # Updating population normalised constraint function values
        if self.problem.n_con > 0:
            self.population = self.evaluator.sum_normalised_cons(self.population, self.problem, max_abs_con_vals=self.max_abs_con_vals)

        # Update optimum
        opt_idx = rank_by_front_and_crowding(self.population, 1, cons_val=None)
        self.opt = self.population[opt_idx][0]
