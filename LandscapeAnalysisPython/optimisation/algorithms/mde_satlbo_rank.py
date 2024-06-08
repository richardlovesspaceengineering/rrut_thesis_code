import numpy as np
from scipy.spatial import distance
import copy
import random

# from optimisation.model.algorithm import Algorithm
from optimisation.algorithms.evolutionary_algorithm import EvolutionaryAlgorithm

# from optimisation.surrogate.models.rbf_kernel_ensemble import RBFKernelEnsembleSurrogate
from optimisation.surrogate.models.ensemble import EnsembleSurrogate
from optimisation.surrogate.models.rbf import RadialBasisFunctions
from optimisation.surrogate.models.mars import MARSRegression

from optimisation.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from optimisation.operators.survival.rank_and_crowding_survival import RankAndCrowdingSurvival
from optimisation.model.population import Population
from optimisation.model.repair import BasicBoundsRepair

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

        # TLBO Parameters
        self.g = -2
        self.h = 6
        self.T = np.array([1, 1, 1])

        self.n_infill_exploitation = 3
        self.n_infill_total = 4
        self.tlbo_max_gen = self.problem.n_var

        # MODE Parameters
        # memory for f and CR
        self.f = [0.8, 1.0, 1.0, 0.8, 0.8, 0.9, 0.4, 0.4]
        self.cr = [0.2, 0.1, 0.9, 0.8, 0.6, 0.2, 0.2, 0.9]

        # Initialise from the evolutionary algorithm class
        super()._initialise()

        # Initialise TLBO population only at beginning after LHS
        self.population = copy.deepcopy(self.surrogate.population)

        # Ali Ahrari Trust Region Parameters
        self.tau_r = 3
        self.e_prox = 1e-6*np.linalg.norm(self.problem.x_upper - self.problem.x_lower)
        self.R_prox = 0.0

        population_vars = self.population.extract_var()
        lhs_distances = distance.cdist(np.atleast_2d(population_vars[0]), population_vars[1:])
        self.R_init_0 = np.min(lhs_distances)
        self.n_init = len(self.surrogate.population)

        # Extract surrogate model from surrogate strategy
        self.obj_surrogate = self.surrogate.obj_surrogates[0]

    def _next(self):

        ## 0. Adaptively resize the Trust Region R_approx
        trust_region = self.R_init_0 * (1 - len(self.surrogate.population) / self.max_f_eval)**self.tau_r
        self.R_prox = np.max((trust_region, self.e_prox))
        # print('Trust Region R: ', self.R_prox)

        ## 1. Surrogate-Based Multimodal DE Exploitation
        top_ranked_population = self.survival.do(self.problem, self.surrogate.population, self.n_population)
        exploit_population = self.local_minima_exploitation(top_ranked_population, n_population=self.n_population,
                                                            n_generations=50, n_return=1)
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
            kernels_to_keep = self.obj_surrogate.predict_mse_accuracy(infill_population)
            # kernels_to_keep = self.obj_surrogate.predict_rank_accuracy(infill_population)
            print('Using kernels: ', kernels_to_keep)

        ## 5. Update Surrogate
        self.obj_surrogate.add_points(infill_population.extract_var(), infill_population.extract_obj().flatten())
        self.obj_surrogate.train()

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
            y_var[cntr1, 0] = self.obj_surrogate.predict_model(x_var[cntr1], 0)
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

    def local_minima_exploitation(self, init_population, n_population=30, n_generations=50, n_survive=6, n_return=1):

        # Calculate each of the surrogates
        optima_population = Population(self.problem, 0)

        for cntr in range(self.obj_surrogate.n_kernels):

            # Initialise MDE Population with LHS
            seed = np.random.randint(low=0, high=10000, size=(1,))
            lhs_de = LatinHypercubeSampling(iterations=100)
            lhs_de.do(n_population, x_lower=self.problem.x_lower, x_upper=self.problem.x_upper, seed=seed)
            x_var = lhs_de.x
            de_population = Population(self.problem, n_individuals=len(x_var))
            de_population.assign_var(self.problem, x_var)
            de_population = Population.merge(de_population, copy.deepcopy(init_population))
            x_var = de_population.extract_var()
            y_var = np.zeros((len(de_population), 1))
            for cntr1 in range(len(de_population)):
                y_var[cntr1, 0] = self.obj_surrogate.predict_model(x_var[cntr1], cntr)
            de_population.assign_obj(y_var)

            for iteration_counter in range(n_generations):

                # Create Offspring
                offspring_population = self.create_offspring(de_population, de_population)
                offspring_variables = offspring_population.extract_var()

                # Repair out-of-bounds
                offspring_variables = self.repair_out_of_bounds(self.problem.x_lower, self.problem.x_upper, offspring_variables)

                # Surrogate screening
                offspring_objectives = np.zeros((len(offspring_population), 1))
                for cntr2 in range(len(offspring_population)):
                    offspring_objectives[cntr2, 0] = self.obj_surrogate.predict_model(offspring_variables[cntr2, :], cntr)
                offspring_population.assign_obj(offspring_objectives)

                # Modality Clustering Survival
                offspring_population = self.modal_sort_and_ranking(offspring_population, n_population, cosine=True)
                de_population = copy.deepcopy(offspring_population)

            # Final survival for infill points
            infill_population = self.modal_sort_and_ranking(de_population, n_survive, cosine=False)

            # print(cntr, 'n_modal after R_trust: ', len(infill_population))

            # Select n_return points (per kernel)
            if len(infill_population) > 0:
                optima_population = Population.merge(optima_population, infill_population[:n_return])

        # Duplicates within the population
        is_duplicate = self.check_duplications(optima_population, other=None, epsilon=self.duplication_tolerance)
        optima_population = optima_population[np.invert(is_duplicate)[0]]

        # Duplicates with the surrogate population
        is_duplicate = self.check_duplications(optima_population, self.surrogate.population,
                                               epsilon=self.R_prox)  # Use Trust Region as distance threshold
        optima_population = optima_population[np.invert(is_duplicate)[0]]

        return optima_population

    def modal_sort_and_ranking(self, population, n_survived, stop_at_cluster=10, cosine=True):
        # Generate 30 interior points
        samples = np.linspace(0.0, 1.0, 32)[1:-1]

        # Rank by objectives first
        obj_ranked_population = self.survival.do(self.problem, population, len(population))

        # Extract Population
        obj_ranked_var = obj_ranked_population.extract_var()
        survived_labels = -np.ones(len(obj_ranked_var))
        assigned_mask = np.full(len(obj_ranked_var), False)

        # Perform modal sorting until no variables left to sort
        cluster_label = 0
        while cluster_label < stop_at_cluster:

            # Break loop if clustering predicts less clusters than maximum specified
            if np.all(assigned_mask):
                break

            # Select current best and rank by distance
            current_best = obj_ranked_var[np.invert(assigned_mask)][0]
            distances = distance.cdist(np.atleast_2d(current_best), obj_ranked_var)[0]

            # Set a large distance if points already selected
            distances[assigned_mask] = np.NaN

            # Sort distances in order of closest to furthest
            indices = np.argsort(distances)

            # Compute Hill Valley Test until modality between individuals does not match
            modal_indices = []
            for idx in indices:
                x_q = obj_ranked_var[idx]
                flag = self.HVT(current_best, x_q, samples=samples)
                if flag == 0:
                    modal_indices.append(idx)
                else:
                    break

            # Assign correct cluster label to identified individuals and allocate mask of already-assigned individuals
            assigned_mask[modal_indices] = True
            survived_labels[modal_indices] = cluster_label

            # Update label counter for next modal cluster
            if modal_indices != []:
                cluster_label += 1

        # Determine n_survived split among modal clusters
        n_clust = len(np.unique(survived_labels))
        if cosine:
            area = np.sum(np.cos(np.pi * np.arange(n_clust) / (2 * n_clust)))
            scale = n_survived / area
            n_split = np.round(scale * np.cos(np.pi * np.arange(n_clust) / (2 * n_clust))).astype(int)
        else:
            n_split = [len(item) for item in np.array_split(np.arange(n_survived), n_clust)]

        # Conduct survival for each cluster and merge populations
        count = 0
        n_assigned = 0
        survived_population = None
        for i in np.unique(survived_labels):

            # Identify indices and number of points to be assigned
            cluster_idx = survived_labels == i
            n_per_cluster = np.count_nonzero(cluster_idx)
            n_to_survive = n_split[count]
            n_assigned += n_to_survive

            # Assign one individual directly and initialise population if necessary
            if n_per_cluster == 1:
                if survived_population is None:
                    survived_population = obj_ranked_population[cluster_idx]
                else:
                    survived_population = Population.merge(survived_population, obj_ranked_population[cluster_idx])

            # Perform survival and initialise population if necessary
            else:
                temp_pop = self.survival.do(self.problem, obj_ranked_population[cluster_idx], n_to_survive)
                if survived_population is None:
                    survived_population = copy.deepcopy(temp_pop)
                else:
                    survived_population = Population.merge(survived_population, temp_pop)
            count += 1

        return survived_population

    def HVT(self, x_p, x_q, samples=[0.2, 0.5, 0.75]):
        # Handle the case when x_p and x_q are the same
        if np.all(x_p == x_q):
            return 0

        # Function values at points p and q
        y_p = self.obj_surrogate.predict_model(x_p, 0)
        y_q = self.obj_surrogate.predict_model(x_q, 0)
        y_max = np.max((y_p, y_q))

        x_init = np.zeros((len(samples), self.problem.n_var))
        modality = np.zeros(len(samples))
        for i in range(len(samples)):
            # Calculate minimum fitness interior points between p and q
            x_init[i, :] = x_p + (x_q - x_p) * samples[i]

            # Determine modality using interior points
            dy = y_max - self.obj_surrogate.predict_model(x_init[i, :], 0)
            # print(dy)
            if dy < 0.0:
                modality[i] = 1

        # Return 0 if modality corresponds, or 1 if not of the same modality
        if np.sum(modality) == 0.0:
            return 0
        else:
            return 1

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
        offspring = BasicBoundsRepair().do(self.problem, offspring)

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
        offspring = BasicBoundsRepair().do(self.problem, offspring)

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
        offspring = BasicBoundsRepair().do(self.problem, offspring)

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
        offspring = BasicBoundsRepair().do(self.problem, offspring)

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
        offspring = BasicBoundsRepair().do(self.problem, offspring)

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
        offspring = BasicBoundsRepair().do(self.problem, offspring)

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
        offspring = BasicBoundsRepair().do(self.problem, offspring)

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

    def repair_out_of_bounds(self, lower_bound, upper_bound, var_array, **kwargs):

        # Upper and lower bounds masks
        upper_mask = var_array > upper_bound
        lower_mask = var_array < lower_bound

        # Repair variables lying outside bounds
        var_array[upper_mask] = np.tile(upper_bound, (len(var_array), 1))[upper_mask]
        var_array[lower_mask] = np.tile(lower_bound, (len(var_array), 1))[lower_mask]

        return var_array

    def calc_dist(self, pop, other=None):

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

    def _select_random_indices(self, population_size, nr_indices, current_index=None):
        index_list = list(range(population_size))
        if current_index is not None:
            index_list.pop(current_index)
        selected_indices = random.sample(index_list, nr_indices)
        return selected_indices