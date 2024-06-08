import numpy as np
from scipy.spatial import distance
import copy
import random

# from optimisation.model.algorithm import Algorithm
from optimisation.algorithms.evolutionary_algorithm import EvolutionaryAlgorithm

# from optimisation.surrogate.models.rbf_kernel_ensemble import RBFKernelEnsembleSurrogate
# from optimisation.surrogate.models.ensemble import EnsembleSurrogate
# from optimisation.surrogate.models.rbf import RadialBasisFunctions
# from optimisation.surrogate.models.mars import MARSRegression

from optimisation.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from optimisation.operators.survival.rank_and_crowding_survival import RankAndCrowdingSurvival
from optimisation.model.population import Population
# from optimisation.model.repair import BasicBoundsRepair
from optimisation.model.repair import BounceBackBoundsRepair

import matplotlib
# matplotlib.use('TkAgg')


class MDE_SATLBO(EvolutionaryAlgorithm):
    """
    Surrogate-Assisted Teaching-Learning-Based Optimisation (SATLBO)
    Dong 2021
    """

    def __init__(self,
                 n_population=None,
                 sampling=LatinHypercubeSampling(),
                 survival=RankAndCrowdingSurvival(),
                 **kwargs):

        # Population parameters
        self.n_population = n_population

        # # Surrogate strategy instance
        # self.surrogate = surrogate

        # Sampling
        self.sampling = sampling

        # Survival
        self.survival = survival

        # Optimum position
        self.opt = None

        # Repair strategy
        self.repair = BounceBackBoundsRepair()

        super().__init__(n_population=n_population,
                         sampling=sampling,
                         survival=survival,
                         n_offspring=n_population,
                         **kwargs)

    def _initialise(self):

        # Generation parameters
        self.max_f_eval = self.max_gen + self.surrogate.n_training_pts
        print(self.max_f_eval)
        self.duplication_tolerance = 1e-2

        # Infill Points
        self.n_infill_exploitation = 3
        self.n_infill_total = 4

        # MODE Parameters (memory for f and CR)
        self.f = [0.8, 1.0, 1.0, 0.8, 0.8, 0.9, 0.4, 0.4]
        self.cr = [0.2, 0.1, 0.9, 0.8, 0.6, 0.2, 0.2, 0.9]

        # SATLBO Parameters
        self.tlbo_max_gen = self.problem.n_var
        self.g = -2
        self.h = 6

        # Initialise from the evolutionary algorithm class
        super()._initialise()

        # Initialise TLBO population only at beginning after LHS
        self.population = copy.deepcopy(self.surrogate.population)

        # Extract surrogate model from surrogate strategy
        self.obj_surrogate = self.surrogate.obj_surrogates[0]

        print('mode_satlbo1')

    def _next(self):

        # Conduct next iteration
        self._step()

    def _step(self):

        ## 1. Surrogate-Based Multi-Offspring DE Exploitation
        top_ranked_population = self.survival.do(self.problem, self.surrogate.population, self.n_population)
        exploit_population = self.surrogate_exploitation(top_ranked_population, n_population=self.n_population,
                                                            n_generations=self.problem.n_var, n_return=1)
        n_exploit = len(exploit_population)
        print('Exploit infill: ', n_exploit)

        ## 2. Surrogate-Assisted TLBO (SATLBO) Exploration
        n_explore = self.n_infill_total - n_exploit
        exploration_population = self.satlbo(top_ranked_population, n_generations=self.tlbo_max_gen,
                                             n_population=self.n_population, n_survive=n_explore)
        print('Explore infill: ', len(exploration_population))

        # Evaluate Infill Population and update Surrogate Population
        infill_population = Population.merge(exploit_population, exploration_population)
        infill_population = self.evaluator.do(self.problem.obj_func, self.problem, infill_population)
        self.surrogate.population = Population.merge(self.surrogate.population, infill_population)

        ## 3. Update Optimum
        old_opt = copy.deepcopy(self.opt)
        opt = RankAndCrowdingSurvival().do(self.problem, self.surrogate.population, 1, None, None)
        self.opt = opt[0]

        ## 4. Kernel Ranking
        improvement = self.opt.obj - old_opt.obj
        if improvement == 0:
            # kernels_to_keep = self.obj_surrogate.predict_random_accuracy()
            kernels_to_keep = self.obj_surrogate.predict_sep_accuracy(infill_population)
            # kernels_to_keep = self.obj_surrogate.predict_mse_accuracy(infill_population)
            # kernels_to_keep = self.obj_surrogate.predict_rank_accuracy(infill_population)
            print('Using kernels: ', kernels_to_keep)
        else:
            self.obj_surrogate.clear_ranks()

        ## 5. Update Surrogate
        self.obj_surrogate.add_points(infill_population.extract_var(), infill_population.extract_obj().flatten())
        self.obj_surrogate.train()

    def surrogate_exploitation(self, init_population, n_population=30, n_generations=30, n_return=1):

        # Calculate each of the surrogates
        optima_population = Population(self.problem, 0)

        for cntr in range(self.obj_surrogate.n_kernels):

            # Initialise MODE Population with LHS
            seed = np.random.randint(low=0, high=10000, size=(1,))
            lhs_de = LatinHypercubeSampling(iterations=100)
            lhs_de.do(2*n_population, x_lower=self.problem.x_lower, x_upper=self.problem.x_upper, seed=seed)
            x_var = lhs_de.x
            de_population = Population(self.problem, n_individuals=len(x_var))
            de_population.assign_var(self.problem, x_var)

            # Reduce population to n_population size by selecting most diverse individuals
            de_population = self._select_diverse_individuals(de_population, init_population, n_population)

            # Merge with Top Ranked Surrogate Population
            de_population = Population.merge(de_population, copy.deepcopy(init_population))

            # Assign Surrogate Predictions
            x_var = de_population.extract_var()
            y_var = np.zeros((len(de_population), 1))
            for cntr1 in range(len(de_population)):
                y_var[cntr1, 0] = self.obj_surrogate.predict_model(x_var[cntr1], cntr)
            de_population.assign_obj(y_var)

            # Initialise Pareto Archive
            pareto_archive = copy.deepcopy(de_population)

            for iteration_counter in range(n_generations):

                # Create Multiple Offspring
                offspring_population = self.create_offspring(de_population, pareto_archive)

                # Repair out-of-bounds
                offspring_population = self.repair.do(self.problem, offspring_population)

                # Surrogate Screening
                offspring_variables = offspring_population.extract_var()
                offspring_objectives = np.zeros((len(offspring_population), 1))
                for cntr2 in range(len(offspring_population)):
                    offspring_objectives[cntr2, 0] = self.obj_surrogate.predict_model(offspring_variables[cntr2], cntr)
                offspring_population.assign_obj(offspring_objectives)

                # Rank and Crowding Survival
                offspring_population = self.survival.do(self.problem, offspring_population, self.n_population)
                de_population = copy.deepcopy(offspring_population)

                # Update Archive
                pareto_archive = self.update_archive(pareto_archive, offspring_population,
                                                     population_size=n_population)

            # Final Selection for each Kernel Surrogate
            infill_population = self.survival.do(self.problem, pareto_archive, n_return, gen=self.n_gen,
                                                 max_gen=self.max_gen)

            # Select n_return points (per kernel)
            if len(infill_population) > 0:
                optima_population = Population.merge(optima_population, infill_population[:n_return])

        # Duplicates within the population
        is_duplicate = self.check_duplications(optima_population, other=None, epsilon=self.duplication_tolerance)
        optima_population = optima_population[np.invert(is_duplicate)[0]]

        # Duplicates with the surrogate population
        is_duplicate = self.check_duplications(optima_population, self.surrogate.population,
                                               epsilon=self.duplication_tolerance)
        optima_population = optima_population[np.invert(is_duplicate)[0]]

        return optima_population

    def satlbo(self, init_population, n_generations=30, n_population=30, n_survive=1):

        # Initialise TLBO Population with LHS and merge with Top Ranked Individuals
        seed = np.random.randint(low=0, high=10000, size=(1,))
        lhs_de = LatinHypercubeSampling(iterations=100)
        lhs_de.do(n_population, x_lower=self.problem.x_lower, x_upper=self.problem.x_upper, seed=seed)
        x_var = lhs_de.x
        tlbo_population = Population(self.problem, n_individuals=n_population)
        tlbo_population.assign_var(self.problem, x_var)
        tlbo_population = Population.merge(tlbo_population, copy.deepcopy(init_population))
        x_var = tlbo_population.extract_var()
        y_var = np.zeros((len(tlbo_population), 1))
        for cntr1 in range(len(tlbo_population)):
            y_var[cntr1, 0] = self.obj_surrogate.predict_model(x_var[cntr1], 0)  # Best kernel
        tlbo_population.assign_obj(y_var)

        # Perform Internal Iterations of SATLBO
        for iteration_counter in range(n_generations):
            new_teacher = self.survival.do(self.problem, tlbo_population, 1)
            tlbo_population = self.teachingPhase(tlbo_population, new_teacher)
            tlbo_population = self.learningPhase(tlbo_population, g=self.g, h=self.h)

        # Duplicates within the population
        is_duplicate = self.check_duplications(tlbo_population, other=None, epsilon=self.duplication_tolerance)
        tlbo_population = tlbo_population[np.invert(is_duplicate)[0]]

        # Duplicates with the surrogate population
        is_duplicate = self.check_duplications(tlbo_population, self.surrogate.population,
                                               epsilon=self.duplication_tolerance)
        tlbo_population = tlbo_population[np.invert(is_duplicate)[0]]

        # Select Points to Survive
        exploration_population = self.survival.do(self.problem, tlbo_population, n_survive)

        return exploration_population

    def teachingPhase(self, temp_population, new_teacher):

        teacher_var = new_teacher.extract_var()

        temp_pop = copy.deepcopy(temp_population)
        temp_var = temp_pop.extract_var()
        temp_obj = temp_pop.extract_obj()
        new_var = np.zeros(np.shape(temp_var))

        # Compute mean
        mean_var = np.mean(temp_var, axis=0)  ## Check axis

        # Compute weighted mean
        # mean_var = np.sum(temp_obj * temp_var, axis=0) / np.sum(temp_obj)

        # a = np.random.uniform(0, 0.5, self.problem.n_var)  ## SINGLE RANDOM VAR OR PER DIMENSION ??
        # b = np.random.uniform(0.5, 1, self.problem.n_var)  ## GENERATED ONCE, OR PER INDIVIDUAL POPULATION

        for idx in range(self.n_population):
            # Teaching Factors
            # Tf1 = np.round(1 + np.random.uniform(0, 1, 1))
            # Tf2 = np.round(1 + np.random.uniform(0, 1, 1))
            Tf1 = np.random.uniform(0, 1, 1) + 1
            Tf2 = np.random.uniform(0, 1, 1) + 1

            # Random Factors
            a = np.random.uniform(0, 0.5, 1)
            b = np.random.uniform(0.5, 1, 1)

            # Calculate new position
            new_var[idx, :] = temp_var[idx, :] + a * (teacher_var - Tf1 * mean_var) + b * (
                        teacher_var - Tf2 * temp_var[idx, :])

        # Restore Bounds on new positions
        new_var = self.repair_out_of_bounds(self.problem.x_lower, self.problem.x_upper, new_var)

        # Create new population
        new_population = Population(self.problem, n_individuals=self.n_population)
        new_population.assign_var(self.problem, new_var)
        new_objectives = np.zeros((len(new_var), 1))
        for cntr in range(len(new_var)):
            new_objectives[cntr, 0] = self.obj_surrogate.predict_model(new_var[cntr], 0)  # Best kernel
        new_population.assign_obj(new_objectives)

        return new_population

    def learningPhase(self, population, g=-2, h=6):

        temp_pop = copy.deepcopy(population)
        temp_var = temp_pop.extract_var()
        new_var = np.zeros(np.shape(temp_var))

        # Random variable
        # r = np.random.uniform(0, 1, self.problem.n_var)

        for j_idx in range(self.n_population):

            # Random variable
            r = np.random.uniform(0, 1, 1)

            # Select random individual from the rest of the (passed) population
            k_idx = j_idx
            while k_idx == j_idx:
                k_idx = np.random.randint(1, self.n_population)

            # Calculate new position
            if temp_pop[j_idx].obj < temp_pop[k_idx].obj:  # The current individual dominates the other individual
                new_var[j_idx, :] = temp_var[j_idx, :] + (g + h * r) * (temp_var[j_idx, :] - temp_var[k_idx, :])

            else:  # The other individual dominates the current individual
                new_var[j_idx, :] = temp_var[j_idx, :] + (g + h * r) * (temp_var[k_idx, :] - temp_var[j_idx, :])

        # Restore Bounds on new positions
        new_var = self.repair_out_of_bounds(self.problem.x_lower, self.problem.x_upper, new_var)

        # Create new population
        new_population = Population(self.problem, n_individuals=self.n_population)
        new_population.assign_var(self.problem, new_var)
        new_objectives = np.zeros((len(new_var), 1))
        for cntr in range(len(new_var)):
            new_objectives[cntr, 0] = self.obj_surrogate.predict_model(new_var[cntr], 0)  # Best kernel
        new_population.assign_obj(new_objectives)

        return new_population

    def create_offspring(self, initial_population, pareto_archive):
        var_array = initial_population.extract_var()
        personal_best_array = pareto_archive.extract_var()

        # create the various offspring
        f, cr = self.assign_f_and_cr_from_pool()
        offspring1 = self.best_1_bin(var_array, personal_best_array, f, cr,
                                     population_size=len(var_array))
        f1, cr = self.assign_f_and_cr_from_pool()
        f2, _ = self.assign_f_and_cr_from_pool()
        offspring2 = self.best_2_bin(var_array, personal_best_array, f1, f2, cr,
                                     population_size=len(var_array))
        f, cr = self.assign_f_and_cr_from_pool()
        offspring3 = self.current_to_rand_1_bin(var_array, f, cr,
                                                population_size=len(var_array))
        f, cr = self.assign_f_and_cr_from_pool()
        offspring4 = self.modified_rand_to_best_1_bin(var_array, personal_best_array, f, cr,
                                                      population_size=len(var_array))
        f, cr = self.assign_f_and_cr_from_pool()
        offspring5 = self.current_to_best_1_bin(var_array, personal_best_array, f, cr,
                                                population_size=len(var_array))
        f, cr = self.assign_f_and_cr_from_pool()
        offspring6 = self.rand_1_bin(var_array, f, cr, population_size=len(var_array))
        f, cr = self.assign_f_and_cr_from_pool()
        offspring7 = self.current_to_randbest_1_bin(var_array, personal_best_array, f, cr,
                                                    population_size=len(var_array))

        # merge everything into a very large population
        merged_population = Population.merge(initial_population, offspring1)
        merged_population = Population.merge(merged_population, offspring2)
        merged_population = Population.merge(merged_population, offspring3)
        merged_population = Population.merge(merged_population, offspring4)
        merged_population = Population.merge(merged_population, offspring5)
        merged_population = Population.merge(merged_population, offspring6)
        merged_population = Population.merge(merged_population, offspring7)

        return merged_population

    def rand_1_bin(self, var_array, f, cr, population_size=None):
        if population_size is None:
            population_size = self.n_population
        # create arrays needed for steps
        mutant_array = np.zeros(var_array.shape)
        trial_array = np.zeros(var_array.shape)

        # loop through the population
        for idx in range(len(var_array)):
            archive_indices = self._select_random_indices(len(var_array), 3, current_index=idx)
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

    def best_1_bin(self, var_array, personal_best_array, f, cr, population_size=None):
        if population_size is None:
            population_size = self.n_population
        # create arrays needed for steps
        mutant_array = np.zeros(var_array.shape)
        trial_array = np.zeros(var_array.shape)

        # loop through the population
        for idx in range(len(var_array)):
            archive_indices = self._select_random_indices(len(var_array), 2, current_index=idx)
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

    def best_2_bin(self, var_array, personal_best_array, f1, f2, cr, population_size=None):
        if population_size is None:
            population_size = self.n_population
        # create arrays needed for steps
        mutant_array = np.zeros(var_array.shape)
        trial_array = np.zeros(var_array.shape)

        # loop through the population
        for idx in range(len(var_array)):
            archive_indices = self._select_random_indices(len(var_array), 4, current_index=idx)
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
                                  population_size=None):
        if population_size is None:
            population_size = self.n_population
        # create arrays needed for shamode steps
        mutant_array = np.zeros(var_array.shape)
        trial_array = np.zeros(var_array.shape)

        # loop through the population
        for idx in range(len(var_array)):
            archive_indices = self._select_random_indices(len(var_array), 2, current_index=idx)
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
                              population_size=None):
        if population_size is None:
            population_size = self.n_population
        # create arrays needed for shamode steps
        mutant_array = np.zeros(var_array.shape)
        trial_array = np.zeros(var_array.shape)

        # loop through the population
        for idx in range(len(var_array)):
            archive_indices = self._select_random_indices(len(var_array), 2, current_index=idx)
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
                                    population_size=None):
        if population_size is None:
            population_size = self.n_population
        # create arrays needed for shamode steps
        mutant_array = np.zeros(var_array.shape)
        trial_array = np.zeros(var_array.shape)

        # loop through the population
        for idx in range(len(var_array)):
            archive_indices = self._select_random_indices(len(var_array), 4, current_index=idx)
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

    def current_to_rand_1_bin(self, var_array, f, cr, population_size=None):
        if population_size is None:
            population_size = self.n_population
        # create arrays needed for shamode steps
        mutant_array = np.zeros(var_array.shape)
        trial_array = np.zeros(var_array.shape)

        # loop through the population
        for idx in range(len(var_array)):
            archive_indices = self._select_random_indices(len(var_array), 3, current_index=idx)
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

    def update_archive(self, population, offspring, population_size=None):
        if population_size is None:
            population_size = self.n_population
        merged_population = Population.merge(population, offspring)
        merged_population = self.survival.do(self.problem, merged_population, population_size,
                                             gen=self.n_gen, max_gen=self.max_gen)

        return merged_population

    def select_individuals(self, lower_bound, upper_bound, population, **kwargs):

        var_array = population.extract_var()
        # Upper and lower bounds masks
        upper_mask = var_array <= upper_bound
        lower_mask = var_array >= lower_bound

        # create mask for selected
        selected = [False for i in range(len(upper_mask))]
        for i in range(len(upper_mask)):
            if lower_mask[i].all() and upper_mask[i].all():
                selected[i] = True

        return selected

    def check_duplications(self, pop, other, epsilon=1e-3):

        dist = self.calc_dist(pop, other)
        dist[np.isnan(dist)] = np.inf

        is_duplicate = [np.any(dist < epsilon, axis=1)]

        return is_duplicate

    @staticmethod
    def repair_out_of_bounds(lower_bound, upper_bound, var_array, **kwargs):

        # Upper and lower bounds masks
        upper_mask = var_array > upper_bound
        lower_mask = var_array < lower_bound

        # Repair variables lying outside bounds
        var_array[upper_mask] = np.tile(upper_bound, (len(var_array), 1))[upper_mask]
        var_array[lower_mask] = np.tile(lower_bound, (len(var_array), 1))[lower_mask]

        return var_array

    @staticmethod
    def calc_dist(pop, other=None):

        pop_var = pop.extract_var()

        if other is None:
            dist = distance.cdist(pop_var, pop_var)
            dist[np.triu_indices(len(pop_var))] = np.inf
        else:
            other_var = other.extract_var()
            if pop_var.ndim == 1:
                pop_var = pop_var[None, :]
            if other_var.ndim == 1:
                other_var = other_var[None, :]
            dist = distance.cdist(pop_var, other_var)

        return dist

    @staticmethod
    def _select_random_indices(population_size, nr_indices, current_index=None):
        index_list = list(range(population_size))
        if current_index is not None:
            index_list.pop(current_index)
        selected_indices = random.sample(index_list, nr_indices)
        return selected_indices

    @staticmethod
    def _select_diverse_individuals(population, comp_population, n_survive=None):

        if n_survive is None:
            n_survive = len(population)

        # Extract population individuals
        x_vars = population.extract_var()
        comp_vars = comp_population.extract_var()

        # Calculate distance matrix of population
        dist_mat = distance.cdist(x_vars, comp_vars)
        dist_mat[dist_mat == 0.] = np.nan

        # Minimum distance from surrounding points
        min_dist = np.nanmin(dist_mat, axis=1)

        # Return first n_survive points with largest distances
        diverse_individuals = np.argpartition(-min_dist, n_survive)[:n_survive]

        return population[diverse_individuals]
