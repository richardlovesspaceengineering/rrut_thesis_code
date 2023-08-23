import numpy as np

from optimisation.model.algorithm import Algorithm
from optimisation.model.duplicate import DefaultDuplicateElimination
from optimisation.model.mating import Mating
from optimisation.model.population import Population
from optimisation.operators.survival.rank_and_crowding_survival import RankAndCrowdingSurvival


class EvolutionaryAlgorithm(Algorithm):

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

        # Population and offspring
        self.population = None
        self.offspring = None

        # Surrogate strategy instance
        self.surrogate = surrogate

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
            if self.surrogate is not None:
                self.surrogate.initialise(self.problem, self.sampling)

            # Compute sampling
            self.sampling.do(self.n_population, self.problem.x_lower, self.problem.x_upper)

            # Assign sampled design variables to population
            self.population.assign_var(self.problem, self.sampling.x)
            if self.x_init:
                self.population[0].set_var(self.problem, self.problem.x_value)
            if self.x_init_additional and self.problem.x_value_additional is not None:
                for i in range(len(self.problem.x_value_additional)):
                    self.population[i+1].set_var(self.problem, self.problem.x_value_additional[i, :])

        # Evaluate initial population
        if self.surrogate is not None:
            self.population = self.evaluator.do(self.surrogate.obj_func, self.problem, self.population)
        else:
            self.population = self.evaluator.do(self.problem.obj_func, self.problem, self.population)

        # Dummy survival call to ensure population is ranked prior to mating selection
        if self.survival:
            self.population = self.survival.do(self.problem, self.population, self.n_population, self.n_gen, self.max_gen)

        # Calculate maximum constraint values across the population
        self.max_abs_con_vals = self.evaluator.calc_max_abs_cons(self.population, self.problem)

        # Assign rank and crowding to population
        self.population.assign_rank_and_crowding()

        # Update optimum
        opt = RankAndCrowdingSurvival().do(self.problem, self.population, 1, None, None)
        self.opt = opt[0]

    def _next(self):

        # Conduct surrogate model refinement (adaptive sampling)
        if self.surrogate is not None:
            self.surrogate.run(self.problem)

        # Conduct mating using the current population
        self.offspring = self.mating.do(self.problem, self.population, self.n_offspring)

        # Evaluate offspring
        if self.surrogate is not None:
            self.offspring = self.evaluator.do(self.surrogate.obj_func, self.problem, self.offspring, self.max_abs_con_vals)
        else:
            self.offspring = self.evaluator.do(self.problem.obj_func, self.problem, self.offspring, self.max_abs_con_vals)

        # Merge the offspring with the current population
        self.population = Population.merge(self.population, self.offspring)

        # Conduct survival selection
        self.population = self.survival.do(self.problem, self.population, self.n_population, self.n_gen, self.max_gen)

        # Update maximum constraint values across the population
        self.max_abs_con_vals = self.evaluator.calc_max_abs_cons(self.population, self.problem)

        # Updating population normalised constraint function values
        if self.problem.n_con > 0:
            self.population = self.evaluator.sum_normalised_cons(self.population, self.problem, max_abs_con_vals=self.max_abs_con_vals)

        # Assign rank and crowding to population
        self.population.assign_rank_and_crowding()

        # Update optimum
        opt = RankAndCrowdingSurvival().do(self.problem, self.population, 1, None, None)
        self.opt = opt[0]

