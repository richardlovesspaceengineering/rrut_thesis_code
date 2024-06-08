import numpy as np

from optimisation.algorithms.evolutionary_algorithm import EvolutionaryAlgorithm

from optimisation.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from optimisation.operators.selection.random_selection import RandomSelection
from optimisation.operators.crossover.differential_evolution_crossover import DifferentialEvolutionCrossover
from optimisation.operators.crossover.binomial_crossover import BiasedCrossover
from optimisation.operators.crossover.exponential_crossover import ExponentialCrossover

from optimisation.model.population import Population
from optimisation.model.repair import BounceBackBoundsRepair
from optimisation.operators.replacement.improvement_replacement import ImprovementReplacement
from optimisation.operators.survival.rank_and_crowding_survival import RankAndCrowdingSurvival


class DifferentialEvolution(EvolutionaryAlgorithm):

    def __init__(self,
                 n_population=100,
                 sampling=LatinHypercubeSampling(),
                 selection=RandomSelection(),
                 crossover=None,
                 mutation=None,
                 var_selection='rand',
                 var_n=1,
                 var_mutation='bin',
                 cr=0.5,
                 f=0.3,
                 dither='vector',
                 jitter=False,
                 **kwargs):

        # DE parameters
        self.var_selection = var_selection
        self.var_n = var_n
        self.var_mutation = var_mutation

        # DE crossover
        if crossover is None:
            crossover = DifferentialEvolutionCrossover(weight=f, dither=dither, jitter=jitter)

        # DE mutation
        if self.var_mutation == 'exp':
            mutation = ExponentialCrossover(cr)
        elif self.var_mutation == 'bin':
            mutation = BiasedCrossover(cr)

        super().__init__(n_population=n_population,
                         sampling=sampling,
                         selection=selection,
                         crossover=crossover,
                         mutation=mutation,
                         survival=None,
                         n_offspring=n_population,
                         **kwargs)

    def _next(self):

        # Extract objective & constraint arrays from population
        obj_array, cons_array = self.population.extract_obj(), self.population.extract_cons_sum()

        # Conduct selection
        if self.var_selection == 'rand':
            parents = self.mating.selection.do(self.population, self.n_population, self.mating.crossover.n_parents)
        elif self.var_selection == 'best':
            best = np.argmin(obj_array[:, 0])
            parents = self.mating.selection.do(self.population, self.n_population, self.mating.crossover.n_parents - 1)
            parents = np.column_stack([np.full(self.n_population, best), parents])
        elif self.var_selection == 'rand+best':
            best = np.argmin(obj_array[:, 0])
            parents = self.mating.selection.do(self.population, self.n_population, self.mating.crossover.n_parents)
            use_best = np.random.random(self.n_population) < 0.3
            parents[use_best, 0] = best
        else:
            raise Exception('Unknown selection method')

        # Conduct crossover
        _offspring = self.mating.crossover.do(self.problem, self.population, parents)

        # Conduct mutation (which is actually another crossover operation)
        _population = Population.merge(self.population, _offspring)
        _parents = np.column_stack([np.arange(self.n_population), np.arange(self.n_population) + self.n_population])
        self.offspring = self.mating.mutation.do(self.problem, _population, _parents)[:self.n_population]

        # Repair variables outside bounds
        self.offspring = BounceBackBoundsRepair().do(self.problem, self.offspring)

        # Evaluate offspring
        self.evaluator.do(self.problem.obj_func, self.problem, self.offspring, self.max_abs_con_vals)

        # Replace individuals that have improved (this is effectively the survival method)
        self.population = ImprovementReplacement().do(self.problem, self.population, self.offspring)

        # Update maximum constraint values across the population
        self.max_abs_con_vals = self.evaluator.calc_max_abs_cons(self.population, self.problem)

        # Updating population normalised constraint function values
        if self.problem.n_con > 0:
            self.population = self.evaluator.sum_normalised_cons(self.population, self.problem, max_abs_con_vals=self.max_abs_con_vals)

        # Update optimum
        opt = RankAndCrowdingSurvival().do(self.problem, self.population, 1, None, None)
        self.opt = opt[0]

