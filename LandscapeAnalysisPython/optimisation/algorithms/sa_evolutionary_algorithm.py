import copy

import numpy as np

from optimisation.model.algorithm import Algorithm
from optimisation.model.duplicate import DefaultDuplicateElimination
from optimisation.model.mating import Mating
from optimisation.model.population import Population
from optimisation.operators.survival.rank_and_crowding_survival import RankAndCrowdingSurvival
from optimisation.util.non_dominated_sorting import NonDominatedSorting


class SAEvolutionaryAlgorithm(Algorithm):

    def __init__(self,
                 n_population=None,
                 sampling=None,
                 selection=None,
                 crossover=None,
                 mutation=None,
                 survival=None,
                 n_offspring=None,
                 eliminate_duplicates=DefaultDuplicateElimination(),
                 mating=None,
                 surrogate=None,
                 save_name=None,
                 **kwargs):

        super().__init__(**kwargs)

        # Population parameters
        if n_population is None:
            n_population = 100
        self.n_population = n_population
        if n_offspring is None:
            n_offspring = self.n_population
        self.n_offspring = n_offspring

        # Generation parameters
        self.max_f_eval = (self.max_gen+1)*self.n_population
        self.save_name = save_name

        # Population and offspring
        self.population = None
        self.offspring = None

        # Surrogate strategy instance
        self.surrogate = surrogate
        if self.surrogate is None:
            raise Exception('A surrogate strategy instance must be passed!')

        # Max constraints array
        self.max_abs_con_vals = None

        # Sampling
        self.sampling = sampling

        # Mating
        if mating is None:
            mating = Mating(selection,
                            crossover,
                            mutation,
                            eliminate_duplicates=eliminate_duplicates)
        self.mating = mating

        # Survival
        self.survival = survival

        # Duplicate elimination
        self.eliminate_duplicates = eliminate_duplicates

    def _initialise(self):

        # Instantiate population
        self.population = Population(self.problem, self.n_population)

        # Population initialisation
        if self.hot_start:
            # Initialise population using hot-start
            self.hot_start_initialisation()
        else:
            # Initialise surrogate modelling strategy
            self.surrogate.initialise(self.problem, self.sampling)
            self.evaluator.n_eval += len(self.surrogate.population)

            # Assign sampled design variables to population
            self.population = copy.deepcopy(self.surrogate.population)

        # # Dummy survival call to ensure population is ranked prior to mating selection
        # if self.survival:
        #     self.population = self.survival.do(self.problem, self.population, self.n_population, self.n_gen, self.max_gen)

        if self.problem.n_con > 0:
            # Calculate maximum constraint values across the population
            self.max_abs_con_vals = self.evaluator.calc_max_abs_cons(self.population, self.problem)

        # Assign rank and crowding to population
        self.population.assign_rank_and_crowding()

        # Update optimum
        if self.problem.n_con == 0:
            fronts = NonDominatedSorting().do(self.population.extract_obj(), return_rank=False)
            self.opt = copy.deepcopy(self.population[fronts[0]])

    def _next(self):
        pass

