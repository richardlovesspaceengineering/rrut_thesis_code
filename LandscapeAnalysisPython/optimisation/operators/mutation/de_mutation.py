import copy

import numpy as np
import random

from optimisation.model.repair import BasicBoundsRepair
# from optimisation.model.repair import BounceBackBoundsRepair
from optimisation.model.population import Population

METHODS = ['convergence', 'diversity', 'balanced']


class DifferentialEvolutionMutation(object):
    def __init__(self, F=None, Cr=None, problem=None, method='diversity'):

        # Store F and Cr memory list
        if F or Cr is None:
            F = [0.8, 1.0, 1.0, 0.8, 0.8, 0.9, 0.4, 0.4]
            Cr = [0.2, 0.1, 0.9, 0.8, 0.6, 0.2, 0.2, 0.9]
        self.f = F
        self.cr = Cr

        # Problem instance
        self.problem = problem

        # Repair
        self.repair = BasicBoundsRepair()

        # Offspring strategy: 'convergence' / 'diversity' / 'balanced'
        self.method = method.lower()
        assert self.method in METHODS

        # Initialise parent class
        super().__init__()

    def do(self, initial_population, pareto_archive, n_offspring):
        n_offspring_to_gen = min(n_offspring, len(initial_population))
        offspring_population = self._do(initial_population, pareto_archive, n_offspring_to_gen)

        return offspring_population

    def _do(self, initial_population, pareto_archive, n_offspring):
        # Create a large offspring population
        var_array = np.atleast_2d(initial_population.extract_var())
        personal_best_array = np.atleast_2d(pareto_archive.extract_var())

        # Created merged population with initial population
        # merged_population = copy.deepcopy(initial_population)
        merged_population = Population(self.problem)

        # create the various offspring
        if self.method == 'balanced' or self.method == 'convergence':
            f, cr = self.assign_f_and_cr_from_pool()
            offspring1 = self.best_1_bin(var_array, personal_best_array, f, cr, n_offspring)
            f1, cr = self.assign_f_and_cr_from_pool()
            f2, _ = self.assign_f_and_cr_from_pool()
            offspring2 = self.best_2_bin(var_array, personal_best_array, f1, f2, cr, n_offspring)
            f, cr = self.assign_f_and_cr_from_pool()
            offspring3 = self.current_to_rand_1_bin(var_array, f, cr, n_offspring)
            f, cr = self.assign_f_and_cr_from_pool()
            offspring4 = self.modified_rand_to_best_1_bin(var_array, personal_best_array, f, cr, n_offspring)
            f, cr = self.assign_f_and_cr_from_pool()
            offspring5 = self.current_to_best_1_bin(var_array, personal_best_array, f, cr, n_offspring)

            # Merge all offspring into a very large population
            merged_population = Population.merge(merged_population, offspring1)
            merged_population = Population.merge(merged_population, offspring2)
            merged_population = Population.merge(merged_population, offspring3)
            merged_population = Population.merge(merged_population, offspring4)
            merged_population = Population.merge(merged_population, offspring5)

        if self.method == 'balanced' or self.method == 'diversity':
            f, cr = self.assign_f_and_cr_from_pool()
            offspring6 = self.rand_1_bin(var_array, f, cr, n_offspring)
            f, cr = self.assign_f_and_cr_from_pool()
            offspring7 = self.current_to_randbest_1_bin(var_array, personal_best_array, f, cr, n_offspring)
            merged_population = Population.merge(merged_population, offspring6)
            merged_population = Population.merge(merged_population, offspring7)

        return merged_population

    def rand_1_bin(self, var_array, f, cr, population_size):
        # create arrays needed for steps
        mutant_array = np.zeros(var_array.shape)
        trial_array = np.zeros(var_array.shape)

        # Randomly permutate arrays to ensure fair chance of all individuals
        perm_mask = np.random.permutation(list(range(len(var_array))))
        var_array = var_array[perm_mask]

        # loop through the population
        for idx in range(population_size):
            archive_indices = self._select_random_indices(population_size, 3, current_index=idx)
            mutant_array[idx, :] = var_array[archive_indices[0], :] + \
                                   f * (var_array[archive_indices[1], :] - var_array[archive_indices[2], :])

            for var_idx in range(self.problem.n_var):
                rand = np.random.random(1)
                j_rand = np.random.randint(0, self.problem.n_var)
                if rand < cr or var_idx == j_rand:
                    trial_array[idx, var_idx] = mutant_array[idx, var_idx]
                else:
                    trial_array[idx, var_idx] = var_array[idx, var_idx]

        offspring = Population(self.problem, population_size)
        offspring.assign_var(self.problem, trial_array)
        offspring = self.repair.do(self.problem, offspring)

        return offspring

    def best_1_bin(self, var_array, personal_best_array, f, cr, population_size):

        # create arrays needed for steps
        mutant_array = np.zeros(var_array.shape)
        trial_array = np.zeros(var_array.shape)

        # Randomly permutate arrays to ensure fair chance of all individuals
        perm_mask = np.random.permutation(list(range(len(var_array))))
        var_array = var_array[perm_mask]

        # loop through the population
        for idx in range(population_size):
            archive_indices = self._select_random_indices(population_size, 2, current_index=idx)
            best_indices = self._select_random_indices(len(personal_best_array), 1)
            mutant_array[idx, :] = personal_best_array[best_indices[0], :] + \
                                   f * (var_array[archive_indices[0], :] - var_array[archive_indices[1], :])

            for var_idx in range(self.problem.n_var):
                rand = np.random.random(1)
                j_rand = np.random.randint(0, self.problem.n_var)
                if rand < cr or var_idx == j_rand:
                    trial_array[idx, var_idx] = mutant_array[idx, var_idx]
                else:
                    trial_array[idx, var_idx] = var_array[idx, var_idx]

        offspring = Population(self.problem, population_size)
        offspring.assign_var(self.problem, trial_array)
        offspring = self.repair.do(self.problem, offspring)

        return offspring

    def best_2_bin(self, var_array, personal_best_array, f1, f2, cr, population_size):

        # create arrays needed for steps
        mutant_array = np.zeros(var_array.shape)
        trial_array = np.zeros(var_array.shape)

        # Randomly permutate arrays to ensure fair chance of all individuals
        perm_mask = np.random.permutation(list(range(len(var_array))))
        var_array = var_array[perm_mask]

        # loop through the population
        for idx in range(population_size):
            archive_indices = self._select_random_indices(population_size, 4, current_index=idx)
            best_indices = self._select_random_indices(len(personal_best_array), 1)
            mutant_array[idx, :] = personal_best_array[best_indices[0], :] + \
                                   f1 * (var_array[archive_indices[0], :] - var_array[archive_indices[1], :]) + \
                                   f2 * (var_array[archive_indices[2], :] - var_array[archive_indices[3], :])

            for var_idx in range(self.problem.n_var):
                rand = np.random.random(1)
                j_rand = np.random.randint(0, self.problem.n_var)
                if rand < cr or var_idx == j_rand:
                    trial_array[idx, var_idx] = mutant_array[idx, var_idx]
                else:
                    trial_array[idx, var_idx] = var_array[idx, var_idx]

        offspring = Population(self.problem, population_size)
        offspring.assign_var(self.problem, trial_array)
        offspring = self.repair.do(self.problem, offspring)

        return offspring

    def current_to_randbest_1_bin(self, var_array, personal_best_array, f, cr,
                                  population_size):

        # create arrays needed for shamode steps
        mutant_array = np.zeros(var_array.shape)
        trial_array = np.zeros(var_array.shape)

        # Randomly permutate arrays to ensure fair chance of all individuals
        perm_mask = np.random.permutation(list(range(len(var_array))))
        var_array = var_array[perm_mask]

        # loop through the population
        for idx in range(population_size):
            archive_indices = self._select_random_indices(population_size, 2, current_index=idx)
            best_indices = self._select_random_indices(len(personal_best_array), 1)
            rand = np.random.random(self.problem.n_var)
            mutant_array[idx, :] = var_array[idx, :] + \
                                   rand * (personal_best_array[best_indices[0], :] - var_array[idx, :]) + \
                                   f * (var_array[archive_indices[0], :] - var_array[archive_indices[1], :])

            for var_idx in range(self.problem.n_var):
                rand = np.random.random(1)
                j_rand = np.random.randint(0, self.problem.n_var)
                if rand < cr or var_idx == j_rand:
                    trial_array[idx, var_idx] = mutant_array[idx, var_idx]
                else:
                    trial_array[idx, var_idx] = var_array[idx, var_idx]

        offspring = Population(self.problem, population_size)
        offspring.assign_var(self.problem, trial_array)
        offspring = self.repair.do(self.problem, offspring)

        return offspring

    def current_to_best_1_bin(self, var_array, personal_best_array, f, cr,
                              population_size):

        # create arrays needed for shamode steps
        mutant_array = np.zeros(var_array.shape)
        trial_array = np.zeros(var_array.shape)

        # Randomly permutate arrays to ensure fair chance of all individuals
        perm_mask = np.random.permutation(list(range(len(var_array))))
        var_array = var_array[perm_mask]

        # loop through the population
        for idx in range(population_size):
            archive_indices = self._select_random_indices(population_size, 2, current_index=idx)
            best_indices = self._select_random_indices(len(personal_best_array), 1)
            rand = np.random.random(self.problem.n_var)
            mutant_array[idx, :] = var_array[idx, :] + \
                                   f * (personal_best_array[best_indices[0], :] - var_array[idx, :]) + \
                                   f * (var_array[archive_indices[0], :] - var_array[archive_indices[1], :])

            for var_idx in range(self.problem.n_var):
                rand = np.random.random(1)
                j_rand = np.random.randint(0, self.problem.n_var)
                if rand < cr or var_idx == j_rand:
                    trial_array[idx, var_idx] = mutant_array[idx, var_idx]
                else:
                    trial_array[idx, var_idx] = var_array[idx, var_idx]

        offspring = Population(self.problem, population_size)
        offspring.assign_var(self.problem, trial_array)
        offspring = self.repair.do(self.problem, offspring)

        return offspring

    def modified_rand_to_best_1_bin(self, var_array, personal_best_array, f, cr,
                                    population_size):
        # create arrays needed for shamode steps
        mutant_array = np.zeros(var_array.shape)
        trial_array = np.zeros(var_array.shape)

        # Randomly permutate arrays to ensure fair chance of all individuals
        perm_mask = np.random.permutation(list(range(len(var_array))))
        var_array = var_array[perm_mask]

        # loop through the population
        for idx in range(population_size):
            archive_indices = self._select_random_indices(population_size, 4, current_index=idx)
            best_indices = self._select_random_indices(len(personal_best_array), 1)
            mutant_array[idx, :] = var_array[archive_indices[0], :] + \
                                   f * (personal_best_array[best_indices[0], :] - var_array[archive_indices[1], :]) + \
                                   f * (var_array[archive_indices[2], :] - var_array[archive_indices[3], :])

            for var_idx in range(self.problem.n_var):
                rand = np.random.random(1)
                j_rand = np.random.randint(0, self.problem.n_var)
                if rand < cr or var_idx == j_rand:
                    trial_array[idx, var_idx] = mutant_array[idx, var_idx]
                else:
                    trial_array[idx, var_idx] = var_array[idx, var_idx]

        offspring = Population(self.problem, population_size)
        offspring.assign_var(self.problem, trial_array)
        offspring = self.repair.do(self.problem, offspring)

        return offspring

    def current_to_rand_1_bin(self, var_array, f, cr, population_size):
        # create arrays needed for shamode steps
        mutant_array = np.zeros(var_array.shape)
        trial_array = np.zeros(var_array.shape)

        # Randomly permutate arrays to ensure fair chance of all individuals
        perm_mask = np.random.permutation(list(range(len(var_array))))
        var_array = var_array[perm_mask]

        # loop through the population
        for idx in range(population_size):
            archive_indices = self._select_random_indices(population_size, 3, current_index=idx)
            rand = np.random.random(self.problem.n_var)
            mutant_array[idx, :] = var_array[idx, :] + \
                                   f * (var_array[archive_indices[0], :] - var_array[idx, :]) + \
                                   f * (var_array[archive_indices[1], :] - var_array[archive_indices[2], :])

            for var_idx in range(self.problem.n_var):
                rand = np.random.random(1)
                j_rand = np.random.randint(0, self.problem.n_var)
                if rand < cr or var_idx == j_rand:
                    trial_array[idx, var_idx] = mutant_array[idx, var_idx]
                else:
                    trial_array[idx, var_idx] = var_array[idx, var_idx]

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
    def _select_random_indices(population_size, nr_indices, current_index=None):
        index_list = list(range(population_size))
        if current_index is not None:
            index_list.pop(current_index)
        selected_indices = random.sample(index_list, nr_indices)
        return selected_indices
