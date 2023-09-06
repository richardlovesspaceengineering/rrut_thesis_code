import copy
import random
import numpy as np

from scipy.stats import cauchy, norm
from scipy.spatial.distance import cdist

from optimisation.algorithms.evolutionary_algorithm import EvolutionaryAlgorithm

from optimisation.operators.sampling.random_sampling import RandomSampling
from optimisation.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from optimisation.operators.selection.tournament_selection import TournamentSelection, comp_by_cv_then_random
from optimisation.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from optimisation.operators.mutation.polynomial_mutation import PolynomialMutation
from optimisation.operators.survival.rank_and_crowding_survival import RankAndCrowdingSurvival
from optimisation.operators.survival.two_ranking_survival import TwoRankingSurvival

from optimisation.model.duplicate import DefaultDuplicateElimination
from optimisation.model.population import Population
from optimisation.model.repair import BasicBoundsRepair
from optimisation.model.normalisation import Normalisation

from optimisation.util import dominator

"""
"Utilizing the Relationship Between Unconstrained and Constrained Pareto Fronts for Constrained 
Multiobjective Optimization" - Liang2022ab

NOTE: This is a variation of CCMO using the ideas from URCMO, so not an exact implementation from the original paper!
"""


class URCMO(EvolutionaryAlgorithm):

    def __init__(self,
                 ref_dirs=None,
                 n_population=100,
                 sampling=LatinHypercubeSampling(),
                 selection=TournamentSelection(comp_func=comp_by_cv_then_random),
                 crossover=SimulatedBinaryCrossover(eta=30, prob=1.0),
                 mutation=PolynomialMutation(eta=20, prob=None),
                 eliminate_duplicates=DefaultDuplicateElimination(),
                 survival1=TwoRankingSurvival(filter_infeasible=False),
                 survival2=RankAndCrowdingSurvival(),
                 **kwargs):

        self.ref_dirs = ref_dirs

        if 'save_results' in kwargs:
            self.save_results = kwargs['save_results']

        if 'save_name' in kwargs:
            self.save_name = kwargs['save_name']

        self.survival1 = survival1
        self.survival2 = survival2

        super().__init__(n_population=n_population,
                         sampling=sampling,
                         selection=selection,
                         crossover=crossover,
                         mutation=mutation,
                         survival=survival1,
                         eliminate_duplicates=eliminate_duplicates,
                         **kwargs)

    def _initialise(self):
        # Initialise from the evolutionary algorithm class -------------------------------------------------------------
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
            self.sampling.do(self.n_population, self.problem.x_lower, self.problem.x_upper, self.seed)

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

        # Create co-evolution population sets
        self.population2 = copy.deepcopy(self.population)

        # Dummy survival call to ensure population is ranked prior to mating selection
        self.population = self.survival1.do(self.problem, self.population, self.n_population, self.n_gen, self.max_gen)
        self.population2 = self.survival2.do(self.problem, self.population2, self.n_population, self.n_gen, self.max_gen)

        # Calculate maximum constraint values across the population
        self.max_abs_con_vals = self.evaluator.calc_max_abs_cons(self.population, self.problem)

        # Assign rank and crowding to population
        self.population.assign_rank_and_crowding()

        # Number of offspring to generate by individual co-evolution
        self.n_half_offspring = int(self.n_offspring / 2)
        
        # DE parameters
        self.repair = BasicBoundsRepair()
        self.f = [0.6, 0.8, 1.0]
        self.cr = [0.1, 0.2, 1.0]

    def _next(self):
        # Generate offspring (GA)
        self.offspring1 = self.mating.do(self.problem, self.population, self.n_half_offspring)
        self.offspring2 = self.mating.do(self.problem, self.population2, self.n_half_offspring)

        # Generate offspring (DE)
        self.offspring3 = self.create_de_offspring()
        self.offspring4 = self.create_de_offspring()

        # Evaluate offspring
        if self.surrogate is not None:
            self.offspring1 = self.evaluator.do(self.surrogate.obj_func, self.problem, self.offspring1, self.max_abs_con_vals)
            self.offspring2 = self.evaluator.do(self.surrogate.obj_func, self.problem, self.offspring2, self.max_abs_con_vals)
        else:
            self.offspring1 = self.evaluator.do(self.problem.obj_func, self.problem, self.offspring1, self.max_abs_con_vals)
            self.offspring2 = self.evaluator.do(self.problem.obj_func, self.problem, self.offspring2, self.max_abs_con_vals)

        # Merge the offspring with the current population
        self.population = Population.merge(self.population, self.offspring1)
        self.population = Population.merge(self.population, self.offspring2)
        self.population = Population.merge(self.population, self.offspring3)
        self.population2 = Population.merge(self.population2, self.offspring1)
        self.population2 = Population.merge(self.population2, self.offspring2)
        self.population2 = Population.merge(self.population2, self.offspring4)

        # Conduct environmental control
        self.population = self.survival1.do(self.problem, self.population, self.n_population, self.n_gen, self.max_gen)

        # Set constraints to None to avoid using in survival of second population
        self.population2 = copy.deepcopy(self.population2)
        self.population2.assign_cons(np.full(self.population2.extract_cons().shape, None))
        self.population2 = self.survival2.do(self.problem, self.population2, self.n_population, self.n_gen, self.max_gen)

    def create_de_offspring(self):
        # Extract variables from populations
        var1_array = self.population.extract_var()
        var2_array = self.population2.extract_var()

        # Obtain F and CR values from list
        f, cr = self.assign_f_and_cr_from_pool()
        f = cauchy.rvs(f, 0.1, 1)
        cr = norm.rvs(cr, 0.1, 1)

        # Select DE operator
        rand_int = np.random.uniform(0.0, 1.0, 1)

        if rand_int < 0.5:
            # Generate offspring from DE/transfer/1
            offspring = self.transfer_to_1_bin(var2_array, var1_array, f, cr, self.n_half_offspring)
        else:
            # Select one random individual from top 100p% individuals in population1
            fitness = self.spea2_fitness(self.population)
            p = np.random.randint(1, self.n_population)
            sorted_indices = np.argpartition(fitness, p)[:p]
            opbest_indices = np.random.choice(sorted_indices, 1)
            opbest_arr = var1_array[opbest_indices]

            # Generate offspring from DE/current-to-opbest/1
            offspring = self.current_to_opbest_1_bin(var2_array, opbest_arr, f, cr, self.n_half_offspring)

        return offspring

    def transfer_to_1_bin(self, var1_array, var2_array, f, cr, population_size):
        mutant_array = np.zeros(var1_array.shape)
        trial_array = np.zeros(var1_array.shape)

        # Randomly permutate arrays to ensure fair chance of all individuals
        perm_mask = np.random.permutation(list(range(len(var1_array))))
        var1_array = var1_array[perm_mask]
        var2_array = var2_array[perm_mask]

        for idx in range(population_size):
            rand_indices = self._select_random_indices(population_size, 1, current_index=idx)
            mutant_array[idx, :] = var1_array[rand_indices, :]

            for var_idx in range(self.problem.n_var):
                rand = np.random.random(1)
                j_rand = np.random.randint(0, self.problem.n_var)
                if rand < cr or var_idx == j_rand:
                    trial_array[idx, var_idx] = mutant_array[idx, var_idx]
                else:
                    trial_array[idx, var_idx] = var2_array[idx, var_idx]

        offspring = Population(self.problem, population_size)
        offspring.assign_var(self.problem, trial_array)
        offspring = self.repair.do(self.problem, offspring)

        return offspring

    def current_to_opbest_1_bin(self, var2_array, opbest_array, f, cr, population_size):
        mutant_array = np.zeros((population_size, self.problem.n_var))
        trial_array = np.zeros((population_size, self.problem.n_var))

        # Randomly permutate arrays to ensure fair chance of all individuals
        perm_mask = np.random.permutation(list(range(len(var2_array))))
        var2_array = var2_array[perm_mask]

        # loop through the population
        for idx in range(population_size):
            rand_indices = self._select_random_indices(population_size, 2, current_index=idx)
            mutant_array[idx, :] = var2_array[idx, :] + \
                                   f * (opbest_array - var2_array[idx, :]) + \
                                   f * (var2_array[rand_indices[0], :] - var2_array[rand_indices[1], :])

            # Do not perform crossover
            trial_array = mutant_array

        offspring = Population(self.problem, population_size)
        offspring.assign_var(self.problem, trial_array)
        offspring = self.repair.do(self.problem, offspring)

        return offspring

    def assign_f_and_cr_from_pool(self):
        ind = random.randint(0, len(self.f) - 1)
        f = self.f[ind]
        cr = self.cr[ind]

        return f, cr

    @staticmethod
    def spea2_fitness(pop):
        # Extract objectives
        obj_val = np.atleast_2d(pop.extract_obj())
        cons_val = pop.extract_cons_sum()

        # Calculate domination matrix
        M = dominator.calculate_domination_matrix(obj_val, cons_val, domination_type="pareto")

        # Number of solutions each individual dominates
        S = np.sum(M == 1, axis=0)

        # The raw fitness of each solution (strength of its dominators)
        R = np.sum(((M == -1) * S), axis=1)

        # Determine the k-th nearest neighbour
        k = int(np.sqrt(len(pop)))
        if k >= len(pop):
            k -= 1

        _pop = Normalisation().do(copy.deepcopy(pop), recalculate=True)
        _obj_val = np.atleast_2d(_pop.extract_obj())

        # Calculate distance matrix and sort by nearest neighbours
        dist_mat = cdist(_obj_val, _obj_val)
        np.fill_diagonal(dist_mat, np.inf)
        sorted_dist_mat = np.sort(dist_mat, axis=1)

        # Inverse distance metric
        D = 1.0 / (sorted_dist_mat[:, k] + 2.0)

        # SPEA2 fitness
        fitness = R + D

        return fitness

    @staticmethod
    def _select_random_indices(population_size, nr_indices, current_index=None):
        index_list = list(range(population_size))
        if current_index is not None:
            index_list.pop(current_index)
        selected_indices = random.sample(index_list, nr_indices)

        return selected_indices
