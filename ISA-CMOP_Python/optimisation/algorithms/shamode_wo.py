import numpy as np
import random
from scipy.stats import cauchy, norm

from optimisation.algorithms.evolutionary_algorithm import EvolutionaryAlgorithm

from optimisation.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from optimisation.operators.selection.random_selection import RandomSelection
from optimisation.operators.crossover.differential_evolution_crossover import DifferentialEvolutionCrossover
from optimisation.operators.crossover.binomial_crossover import BiasedCrossover
from optimisation.operators.crossover.exponential_crossover import ExponentialCrossover

from optimisation.model.population import Population
from optimisation.model.repair import BounceBackBoundsRepair, BasicBoundsRepair
from optimisation.operators.replacement.improvement_replacement import ImprovementReplacement
from optimisation.operators.survival.rank_and_crowding_survival import RankAndCrowdingSurvival

from optimisation.util.non_dominated_sorting import NonDominatedSorting
from optimisation.util.rank_fronts import rank_fronts


class SHAMODE(EvolutionaryAlgorithm):
    """
    success history–based adaptive multi-objective differential evolution with whale optimisation
    https://doi.org/10.1007/s00158-019-02302-x
    """

    def __init__(self,
                 n_population=100,
                 sampling=LatinHypercubeSampling(),
                 selection=RandomSelection(),
                 survival=RankAndCrowdingSurvival(),
                 crossover=None,
                 mutation=None,
                 var_selection='best',
                 var_n=1,
                 var_mutation='exp',
                 cr=0.5,
                 f=0.5,
                 dither='vector',
                 jitter=False,
                 **kwargs):

        # DE parameters
        self.var_selection = var_selection
        self.var_n = var_n
        self.var_mutation = var_mutation

        # archives
        self.pareto_archive = None
        self.adaptive_archive = None
        self.adaptive_archive_size = int(1.4 * n_population)

        # memory for f and CR
        self.f = f
        self.cr = cr
        self.memory_length = 5
        self.memory_index = None

        # survival
        self.survival = survival

        # flag to use spiral movement operator of WOA
        self.use_wo = True
        # flag to use improved spiral movement operator from TLBO
        self.use_spiral = True

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
                         survival=survival,
                         n_offspring=n_population,
                         **kwargs)

    def _initialise(self):

        # Set up the memory index
        self.memory_index = 1

        # Set up the archives
        self.pareto_archive = Population(self.problem, 0)
        self.adaptive_archive = Population(self.problem, 0)

        # Initialise from the evolutionary algorithm class
        super()._initialise()

        # update pareto archive as non-dominated solutions from pareto archive and population
        self.update_pareto_archive()

        # set memory index to 0
        self.memory_index = 0
        # set the f and cr memory

        self.f_memory = self.f * np.ones(self.memory_length)
        self.cr_memory = self.cr * np.ones(self.memory_length)

    def _next(self):

        # Conduct surrogate model refinement (adaptive sampling)
        if self.surrogate is not None:
            self.surrogate.run(self.problem)

        # create offspring and arrays
        cr_array, f_array, offspring, trial_array = self.create_shamode_offspring()

        # merge population and find which offspring survived
        survived = self.create_merged_population(offspring, trial_array)

        # updated pareto archive and adaptive archive
        self.update_pareto_archive()
        self.update_adaptive_archive(offspring[survived])

        # update the memory for f and cr
        self.update_f_and_cr_memory(cr_array, f_array, survived)

    def update_f_and_cr_memory(self, cr_array, f_array, survived):
        if survived.sum() > 0:
            f_memory = np.sum(f_array[survived] * f_array[survived]) / np.sum(f_array[survived])
            cr_memory = np.sum(cr_array[survived] * cr_array[survived]) / np.sum(cr_array[survived])
        else:
            f_memory = self.f_memory[self.memory_index - 1]
            cr_memory = self.cr_memory[self.memory_index - 1]
        self.f_memory[self.memory_index] = f_memory
        self.cr_memory[self.memory_index] = cr_memory
        self.memory_index += 1
        if self.memory_index > self.memory_length - 1:
            self.memory_index = 0

    def create_merged_population(self, offspring, trial_array):
        merged_population = Population.merge(self.population, offspring)
        merged_population = self.survival.do(self.problem, merged_population, self.n_population,
                                             gen=self.n_gen, max_gen=self.max_gen)
        # find which offspring survived
        merged_population_var_array = merged_population.extract_var()
        survived = np.zeros(self.n_population, dtype=bool)
        for idx in range(self.n_population):
            test_array = trial_array[idx, :]
            survived[idx] = (merged_population_var_array == test_array).all(1).any()
        self.population = merged_population
        return survived

    def create_shamode_offspring(self):
        # extract variables and create union of external archive and population
        var_array = self.population.extract_var()
        personal_best_array = self.pareto_archive.extract_var()
        # merge external archive and current population
        external_archive = self.create_external_archive()
        external_array = external_archive.extract_var()
        # create arrays needed for shamode steps
        mutant_array = np.zeros(var_array.shape)
        trial_array = np.zeros(var_array.shape)
        f_array = np.zeros(self.n_population)
        cr_array = np.zeros(self.n_population)
        # loop through the population
        for idx in range(len(var_array)):
            # create f and cr
            f_base = self.f_memory[self.memory_index]
            cr_base = self.cr_memory[self.memory_index]
            f = cauchy.rvs(f_base, 0.1, 1)
            cr = norm.rvs(cr_base, 0.1, 1)
            f_array[idx] = f
            cr_array[idx] = cr
            # select best solutions
            if self.use_spiral:
                best_indices = self._select_random_indices(len(var_array), 2, current_index=idx)
            elif self.use_wo:
                best_indices = self._select_random_indices(len(var_array), 2, current_index=idx)
            else:
                best_indices = self._select_random_indices(len(var_array), 1, current_index=idx)
            archive_indices = self._select_random_indices(len(external_archive), 2)
            mutant_array[idx, :] = var_array[idx, :] + f * (personal_best_array[best_indices[0], :] - var_array[idx, :]) \
                                   + f * (external_array[archive_indices[0], :] - external_array[archive_indices[1], :])
            if self.use_wo:
                rand = np.random.uniform(0, 1, 1)
                if rand < 0.5:
                    l = np.random.uniform(-1, 1, 1)
                    # use euclidean distance for now
                    distance = np.linalg.norm(personal_best_array[best_indices[1], :] - mutant_array[idx, :], 2)
                    mutant_array[idx, :] = np.exp(l) * np.cos(2 * np.pi * l) * distance + personal_best_array[
                                                                                          best_indices[1], :]
            if self.use_spiral:
                rand = np.random.uniform(0, 1, 1)
                if rand < 0.5:
                    # use euclidean distance for now
                    distance = np.linalg.norm(personal_best_array[best_indices[1], :] - mutant_array[idx, :], 2)
                    theta = 2 * (1 - self.n_gen / self.max_gen) - 1
                    mutant_array[idx, :] = np.exp(theta) * np.cos(2 * np.pi * theta) * distance \
                                           + personal_best_array[best_indices[1], :]

            for var_idx in range(self.problem.n_var):
                rand = np.random.random(1)
                j_rand = np.random.randint(0, self.problem.n_var)
                if rand < cr or var_idx == j_rand:
                    trial_array[idx, var_idx] = mutant_array[idx, var_idx]
                else:
                    trial_array[idx, var_idx] = var_array[idx, var_idx]
        offspring = Population(self.problem, self.n_population)
        offspring.assign_var(self.problem, trial_array)
        offspring = BasicBoundsRepair().do(self.problem, offspring)
        if self.surrogate is not None:
            offspring = self.evaluator.do(self.surrogate.obj_func, self.problem, offspring)
        else:
            offspring = self.evaluator.do(self.problem.obj_func, self.problem, offspring)


        return cr_array, f_array, offspring, trial_array

    def update_pareto_archive(self):
        # Merge the pareto archive with the current population
        updated_pareto = Population.merge(self.population, self.pareto_archive)
        if len(updated_pareto) > self.n_population:
            index_list = list(range(len(updated_pareto)))
            selected_indices = random.sample(index_list, self.n_population)
            updated_pareto = updated_pareto[selected_indices]
        self.pareto_archive = updated_pareto

    def update_adaptive_archive(self, offspring):
        self.adaptive_archive = Population.merge(self.adaptive_archive, offspring)
        # trim to the right size
        if len(self.adaptive_archive) > self.adaptive_archive_size:
            index_list = list(range(len(self.adaptive_archive)))
            selected_indices = random.sample(index_list, self.adaptive_archive_size)
            self.adaptive_archive = self.adaptive_archive[selected_indices]

    def create_external_archive(self):
        # Merge the pareto archive with the current population
        external_archive = Population.merge(self.population, self.adaptive_archive)
        return external_archive

    def _select_random_indices(self, population_size, nr_indices, current_index=None):
        index_list = list(range(population_size))
        if current_index is not None:
            index_list.pop(current_index)
        selected_indices = random.sample(index_list, nr_indices)
        return selected_indices

