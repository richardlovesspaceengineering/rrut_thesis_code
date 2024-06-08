import numpy as np
from scipy.spatial import distance
import copy
import random

# from optimisation.model.algorithm import Algorithm
from optimisation.algorithms.evolutionary_algorithm import EvolutionaryAlgorithm

from optimisation.surrogate.models.ensemble import EnsembleSurrogate
from optimisation.surrogate.models.rbf import RadialBasisFunctions
from optimisation.surrogate.models.mars import MARSRegression

from optimisation.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from optimisation.operators.survival.rank_and_crowding_survival import RankAndCrowdingSurvival
from optimisation.model.population import Population
from optimisation.model.repair import BasicBoundsRepair


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
        self.duplication_tolerance = 1e-4

        # TLBO Parameters
        self.g = -2
        self.h = 6
        self.T = np.array([1, 1, 1])   ## SATLBO10

        self.n_infill_exploitation = 3
        self.n_infill_total = 4    ## SATLBO_MODE_LHS3
        self.n_lhs_split = int(self.n_population / 2)
        self.tlbo_max_gen = self.problem.n_var
        self.rs_distance = np.linalg.norm(self.problem.x_upper - self.problem.x_lower) / 4

        # MODE Parameters
        # memory for f and CR
        self.f = [0.8, 1.0, 1.0, 0.8, 0.8, 0.9, 0.4, 0.4]
        self.cr = [0.2, 0.1, 0.9, 0.8, 0.6, 0.2, 0.2, 0.9]

        # Initialise from the evolutionary algorithm class
        super()._initialise()

        # Initialise TLBO population only at beginning after LHS
        self.population = copy.deepcopy(self.surrogate.population)

        # Extract surrogate model from surrogate strategy
        self.obj_surrogate = self.surrogate.obj_surrogates[0]

    def _next(self):

        ## 1. Knowledge Mining (3 new points)
        optima = self.surrogate_exploitation(nr_survived=self.n_infill_exploitation)
        optimal_population = Population(self.problem, n_individuals=len(optima))
        optimal_population.assign_var(self.problem, optima)
        is_duplicate = self.check_duplications(optimal_population, self.surrogate.population,
                                               epsilon=self.duplication_tolerance)
        optimal_population = optimal_population[np.invert(is_duplicate)[0]]
        n_eval = len(optimal_population)

        # ## SATLBO_MODE_no_exploit
        # rand_idx = np.random.randint(0, len(optimal_population), 1)
        # optimal_population = optimal_population[rand_idx]
        # #######

        optimal_population = self.evaluator.do(self.problem.obj_func, self.problem, optimal_population)
        self.surrogate.population = Population.merge(self.surrogate.population, optimal_population)
        print('Exploit infill: ', len(optimal_population))

        # Update population for TLBO
        ranked_population = self.survival.do(self.problem, self.surrogate.population, self.n_lhs_split, gen=self.n_gen,
                                           max_gen=self.max_gen)
        new_teacher = ranked_population[0]

        # # # Do internal iterations of TLBO with random LHS Initialisation
        initialise = LatinHypercubeSampling(iterations=100)
        initialise.do(2*self.n_lhs_split, x_lower=self.problem.x_lower, x_upper=self.problem.x_upper)
        x_init = initialise.x
        lhs_population = Population(self.problem, self.n_lhs_split)
        lhs_population.assign_var(self.problem, x_init)
        lhs_objectives = np.zeros((len(lhs_population), 1))
        temp_vars = lhs_population.extract_var()
        for cntr in range(len(lhs_population)):
            lhs_objectives[cntr, 0] = self.obj_surrogate.predict_model(temp_vars[cntr], 0)
        lhs_population.assign_obj(lhs_objectives)

        # Extract LHS points not too close to surrogate top-ranked ones
        species_seeds = self.identify_species_seeds(lhs_population, self.n_lhs_split, seeds=ranked_population, rs=self.rs_distance)
        lhs_population = lhs_population[species_seeds]

        # Create TLBO Population
        temp_population = Population.merge(ranked_population, lhs_population)

        # Conduct Inner Generations
        for gen_idx in range(self.tlbo_max_gen):
            temp_population = self.teachingPhase(temp_population, new_teacher)
            temp_population = self.learningPhase(temp_population, g=self.g, h=self.h)
        is_duplicate = self.check_duplications(temp_population, self.surrogate.population,
                                               epsilon=self.duplication_tolerance)
        # Re-assign the population for next generation
        self.population = copy.deepcopy(temp_population)

        # Eliminate duplicates and evaluate expensively
        temp_population = temp_population[np.invert(is_duplicate)[0]]
        exploration_population = self.survival.do(self.problem, temp_population, self.n_infill_total - n_eval,
                                                  gen=self.n_gen, max_gen=self.max_gen)
        exploration_population = self.evaluator.do(self.problem.obj_func, self.problem, exploration_population)

        self.surrogate.population = Population.merge(self.surrogate.population, exploration_population)
        print('Explore infill: ', len(exploration_population))

        infill_population = Population.merge(optimal_population, exploration_population)

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

    def teachingPhase(self, temp_population, new_teacher):

        teacher_var = new_teacher.var   #.extract_var()

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
            a = np.random.uniform(0, 0.5, 1)  ## SINGLE RANDOM VAR OR PER DIMENSION ??
            b = np.random.uniform(0.5, 1, 1)  ## GENERATED ONCE, OR PER INDIVIDUAL POPULATION

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
            new_objectives[cntr, 0] = self.obj_surrogate.predict_model(new_var[cntr], 0)
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
            new_objectives[cntr, 0] = self.obj_surrogate.predict_model(new_var[cntr], 0)
        new_population.assign_obj(new_objectives)

        return new_population

    def surrogate_exploitation(self, generations=30, population_size=30,
                               nr_survived=1):

        # need to loop through each surrogate
        optima = np.zeros((self.obj_surrogate.n_kernels, self.problem.n_var))

        for cntr in range(self.obj_surrogate.n_kernels):
            lower_bound = self.obj_surrogate.l_b
            upper_bound = self.obj_surrogate.u_b

            if len(self.obj_surrogate.x) > population_size:
                temp_population = Population(self.problem, n_individuals=len(self.obj_surrogate.x))
                temp_population.assign_var(self.problem, self.obj_surrogate.x)
                temp_population.assign_obj(self.obj_surrogate.y[:].reshape(-1, 1))
                temp_population = self.survival.do(self.problem, temp_population, population_size,
                                                   gen=self.n_gen, max_gen=self.max_gen)
                x_init1 = temp_population.extract_var()
                # add LHS to increase diversity of initial generation
                initialise = LatinHypercubeSampling(iterations=500)
                seed = np.random.randint(low=0, high=10000, size=(1,))
                initialise.do(population_size, x_lower=lower_bound, x_upper=upper_bound, seed=seed)
                x_init2 = initialise.x

                x_init = np.vstack((x_init1, x_init2))
            else:
                initialise = LatinHypercubeSampling(iterations=100)
                initialise.do(2 * population_size, x_lower=lower_bound, x_upper=upper_bound)
                x_init = initialise.x

            # todo - check
            initial_population = Population(self.problem, n_individuals=len(x_init))
            initial_population.assign_var(self.problem, x_init)

            initial_objectives = np.zeros((len(x_init), 1))
            for cntr2 in range(len(x_init)):
                initial_objectives[cntr2, 0] = self.obj_surrogate.predict_model(x_init[cntr2], cntr)
            initial_population.assign_obj(initial_objectives)
            pareto_archive = copy.deepcopy(initial_population)

            for iteration_counter in range(generations):
                offspring_population = self.create_offspring(initial_population=initial_population,
                                                             pareto_archive=pareto_archive)
                offspring_variables = offspring_population.extract_var()
                # limit to the domain size
                offspring_variables = self.repair_out_of_bounds(lower_bound, upper_bound, offspring_variables)
                offspring_objectives = np.zeros((len(offspring_population), 1))
                for cntr2 in range(len(offspring_population)):
                    offspring_objectives[cntr2, 0] = self.obj_surrogate.predict_model(offspring_variables[cntr2], cntr)
                offspring_population.assign_obj(offspring_objectives)
                offspring_population = self.survival.do(self.problem, offspring_population, population_size,
                                                        gen=self.n_gen, max_gen=self.max_gen)
                initial_population = copy.deepcopy(offspring_population)

                pareto_archive = self.update_archive(pareto_archive, offspring_population,
                                                     population_size=population_size)

            offspring_population = self.survival.do(self.problem, offspring_population, nr_survived,
                                                    gen=self.n_gen, max_gen=self.max_gen)

            # print('exploit dist to origin: ', np.linalg.norm(offspring_population[0].var))

            # to account for the case where nr_survived is larger than 1 - picks the last one
            if nr_survived > 1:
                temp = offspring_population.extract_var()
                optima[cntr, :] = temp[nr_survived - 1, :]
            else:
                optima[cntr, :] = offspring_population.extract_var()
        return optima

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

    def identify_species_seeds(self, population, n_seeds, seeds=None, rs=150.0):

        # Rank population from best to worst
        ranked_population = self.survival.do(self.problem, population, len(population), gen=self.n_gen,
                                           max_gen=self.max_gen)
        ranked_vars = ranked_population.extract_var()
        len_seeds = 0

        while len_seeds < n_seeds:
            rs *= 0.9

            # If a set of seeds is already provided, conduct algorithm differently
            if seeds is not None:
                seed_vars = seeds.extract_var()
                initial_seed_vault = list(np.arange(len(seed_vars)))
                seed_vault = list([])
            else:
                initial_seed_vault = list([])
                seed_vault = list([0])

            # Determine seeds
            for cntr in range(len(ranked_population)):
                found = False

                # Break loop if desired number of seeds are found
                if len(seed_vault) >= n_seeds:
                    break

                # Check seeds provided
                for seed in initial_seed_vault:
                    distance = np.linalg.norm(ranked_vars[cntr, :] - seed_vars[seed, :])
                    if distance >= rs:
                        found = True
                    else:
                        found = False

                # Check new seeds generated
                for seed in seed_vault:
                    distance = np.linalg.norm(ranked_vars[cntr, :] - ranked_vars[seed, :])
                    if distance >= rs:
                        found = True
                    else:
                        found = False

                if found:
                    seed_vault.append(cntr)

            len_seeds = len(seed_vault)

        return seed_vault

    def _select_random_indices(self, population_size, nr_indices, current_index=None):
        index_list = list(range(population_size))
        if current_index is not None:
            index_list.pop(current_index)
        selected_indices = random.sample(index_list, nr_indices)
        return selected_indices

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
