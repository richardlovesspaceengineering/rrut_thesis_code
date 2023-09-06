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


class SATLBO(EvolutionaryAlgorithm):
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
        self.n_infill_total = 4    ## SATLBO11 --> Test reducing exploration infills to 4 to improvee CEC probs
        # print('N Infill: ', self.n_infill_total)

        # MODE Parameters
        # memory for f and CR
        self.f = [0.8, 1.0, 1.0, 0.8, 0.8, 0.9, 0.4, 0.4]
        self.cr = [0.2, 0.1, 0.9, 0.8, 0.6, 0.2, 0.2, 0.9]

        ## Kernel Switching Parameters
        self.kernel_list = list(['cubic', 'matern_52', 'matern_32', 'matern_12', 'gaussian',
                                 'tps', 'multiquadratic', 'inv_multiquadratic', 'linear'])
        self.switch_kernels = True
        self.iterations_since_switch = 0
        self.switch_after_iterations = 5
        self.last_improvement = 60
        self.drop_threshold = 0.25
        self.number_of_kernels = 3
        self.accuracy_threshold = 10

        if self.switch_kernels:
            self.iterations_since_improvement = 0
            kernel_nr = random.sample(range(len(self.kernel_list)), self.number_of_kernels)
            self.current_kernel = copy.deepcopy(kernel_nr)
            for nr in range(self.number_of_kernels):
                print(self.kernel_list[kernel_nr[nr]])

        # Initialise from the evolutionary algorithm class
        super()._initialise()

        # Problem Landscape Prediction
        # self.predict_function_nrmse()

        # Initialise TLBO population only at beginning after LHS
        self.population = copy.deepcopy(self.surrogate.population)

    def _next(self):

        ## 0. Create Surrogates in Subspaces
        try:
            obj_surrogates = self.create_surrogates()
        except:
            population_copy = copy.deepcopy(self.surrogate.population)
            is_duplicate = self.check_duplications(population_copy[:-1], population_copy[-1], epsilon=1e-2)
            self.surrogate.population = copy.deepcopy(population_copy[np.invert(is_duplicate)[0]])
            obj_surrogates = self.create_surrogates()

        ## 1. Knowledge Mining (3 new points)
        optima = self.surrogate_exploitation(obj_surrogates, nr_survived=self.n_infill_exploitation)
        optimal_population = Population(self.problem, n_individuals=len(optima))
        optimal_population.assign_var(self.problem, optima)
        is_duplicate = self.check_duplications(optimal_population, self.surrogate.population,
                                               epsilon=self.duplication_tolerance)
        optimal_population = optimal_population[np.invert(is_duplicate)[0]]
        n_eval = len(optimal_population)
        optimal_population = self.evaluator.do(self.problem.obj_func, self.problem, optimal_population)
        self.surrogate.population = Population.merge(self.surrogate.population, optimal_population)
        print('Exploit infill: ', len(optimal_population))

        # Update population for TLBO
        # self.population = self.survival.do(self.problem, self.surrogate.population, self.n_population, gen=self.n_gen,
        #                                    max_gen=self.max_gen)
        new_teacher = self.survival.do(self.problem, self.surrogate.population, 1, gen=self.n_gen, max_gen=self.max_gen)

        #####################################
        pop_max = np.max(self.population.extract_var(), axis=0)
        pop_min = np.min(self.population.extract_var(), axis=0)
        pop_max_dist = np.linalg.norm(pop_max - pop_min)
        print('Pop dist: ', pop_max_dist)
        #####################################

        ## 2. Metaheuristic Exploration (7 new points)
        temp_population = self.teachingPhase(obj_surrogates, new_teacher)
        temp_population = self.learningPhase(obj_surrogates, temp_population, g=self.g, h=self.h)
        is_duplicate = self.check_duplications(temp_population, self.surrogate.population,
                                               epsilon=self.duplication_tolerance)
        temp_population = temp_population[np.invert(is_duplicate)[0]]
        exploration_population = self.survival.do(self.problem, temp_population, self.n_infill_total - n_eval,
                                                  gen=self.n_gen, max_gen=self.max_gen)
        exploration_population = self.evaluator.do(self.problem.obj_func, self.problem, exploration_population)

        # for i in range(len(exploration_population)):
        #     print('explore dist to origin: ', np.linalg.norm(exploration_population[i].var))

        self.surrogate.population = Population.merge(self.surrogate.population, exploration_population)
        print('Explore infill: ', len(exploration_population))

        # Re-assign the population for next generation
        self.population = copy.deepcopy(temp_population)

        ## 3. Update Optimum
        old_opt = copy.deepcopy(self.opt.obj)
        if self.surrogate is not None:
            # if self.problem.n_obj == 1:
            opt = RankAndCrowdingSurvival().do(self.problem, self.surrogate.population, 1, None, None)
        else:
            opt = RankAndCrowdingSurvival().do(self.problem, self.population, 1, None, None)
        self.opt = opt[0]

        ## 4. Kernel Switching
        if self.switch_kernels:
            self.iterations_since_switch += 1
            improvement = self.opt.obj - old_opt

            if improvement == 0:
                self.iterations_since_improvement += 1
            else:
                self.iterations_since_improvement = 0
                self.last_improvement = np.max((self.last_improvement, len(self.surrogate.population)))

            if self.iterations_since_improvement >= self.switch_after_iterations:
                test_population = copy.deepcopy(Population.merge(optimal_population, exploration_population))
                accuracy = self.prediction_accuracy(test_population, obj_surrogates)
                best_kernel = np.nanargmin(accuracy)
                kernel_to_keep = self.current_kernel[best_kernel]

                kernel_list = [i for i in range(len(self.kernel_list))]
                kernel_list.remove(kernel_to_keep)

                if len(self.kernel_list) > self.number_of_kernels + 2:
                    if len(self.surrogate.population) >= self.max_f_eval * self.drop_threshold:
                        worst_kernel = np.nanargmax(accuracy)
                        if accuracy[worst_kernel] > self.accuracy_threshold * accuracy[best_kernel]:
                            kernel_to_remove = self.current_kernel[worst_kernel]
                            kernel_list.remove(kernel_to_remove)

                kernel_nr = random.sample(kernel_list, self.number_of_kernels - 1)
                kernel_nr.append(kernel_to_keep)
                kernels = []
                for k in range(len(kernel_nr)):
                    kernels.append(self.kernel_list[kernel_nr[k]])

                if len(self.kernel_list) > self.number_of_kernels + 2:
                    if len(self.surrogate.population) >= self.max_f_eval * self.drop_threshold:
                        worst_kernel = np.nanargmax(accuracy)
                        if accuracy[worst_kernel] > self.accuracy_threshold * accuracy[best_kernel]:
                            kernel_to_remove = self.current_kernel[worst_kernel]
                            print('removed' + ' ' + self.kernel_list[kernel_to_remove] + ' ' + 'kernel')
                            self.kernel_list.remove(self.kernel_list[kernel_to_remove])

                for k in range(len(kernels)):
                    kernel_nr[k] = self.kernel_list.index(kernels[k])

                self.current_kernel = copy.deepcopy(kernel_nr)
                for nr in range(self.number_of_kernels):
                    print(self.kernel_list[kernel_nr[nr]])

                # keep the most accurate kernel and swap out the other x
                self.iterations_since_improvement = 0
                self.iterations_since_switch = 0

    def teachingPhase(self, obj_surrogates, new_teacher):

        # Use Global Surrogate
        surrogate = obj_surrogates[-1]

        teacher_var = new_teacher.extract_var()

        temp_pop = copy.deepcopy(self.population)
        temp_var = temp_pop.extract_var()
        new_var = np.zeros(np.shape(temp_var))

        # Compute mean
        mean_var = np.mean(temp_var, axis=0)  ## Check axis

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
            new_objectives[cntr, 0] = surrogate.predict(new_var[cntr])
        new_population.assign_obj(new_objectives)

        return new_population

    def learningPhase(self, obj_surrogates, population, n_new_points=7, g=-2, h=6):

        # Use Global Surrogate
        surrogate = obj_surrogates[-1]

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
            new_objectives[cntr, 0] = surrogate.predict(new_var[cntr])
        new_population.assign_obj(new_objectives)

        return new_population

    def surrogate_exploitation(self, obj_surrogates, generations=100, population_size=20,
                               nr_survived=1):

        # need to loop through each surrogate
        optima = np.zeros((len(obj_surrogates), obj_surrogates[0].n_dim))

        for cntr in range(len(obj_surrogates)):
            surrogate = obj_surrogates[cntr]
            lower_bound = surrogate.l_b
            upper_bound = surrogate.u_b

            if len(surrogate.x) > population_size:
                temp_population = Population(self.problem, n_individuals=len(surrogate.x))
                temp_population.assign_var(self.problem, surrogate.x)
                temp_population.assign_obj(surrogate.y[:].reshape(-1, 1))
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
                initial_objectives[cntr2, 0] = surrogate.predict(x_init[cntr2])
            initial_population.assign_obj(initial_objectives)
            pareto_archive = copy.deepcopy(initial_population)

            for iteration_counter in range(generations):
                offspring_population = self.create_offspring(initial_population=initial_population,
                                                             pareto_archive=pareto_archive)
                offspring_variables = offspring_population.extract_var()
                # limit to the domain size
                self.repair_out_of_bounds(lower_bound, upper_bound, offspring_variables)
                offspring_objectives = np.zeros((len(offspring_population), 1))
                for cntr2 in range(len(offspring_population)):
                    offspring_objectives[cntr2, 0] = surrogate.predict(offspring_variables[cntr2])
                offspring_population.assign_obj(offspring_objectives)
                offspring_population = self.survival.do(self.problem, offspring_population, population_size,
                                                        gen=self.n_gen, max_gen=self.max_gen)
                initial_population = copy.deepcopy(offspring_population)

                # Calculate mean diversity in population
                x_diverse = initial_population.extract_var()
                print('Mean Diversity Dist: ', np.mean(distance.cdist(np.atleast_2d(x_diverse[0, :]), x_diverse[1:, :])))

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

    def create_surrogates(self):
        obj_surrogates = list()
        population = copy.deepcopy(self.surrogate.population)
        for cntr in range(len(self.T)):
            self.surrogate.obj_surrogates[0].model.kernel_type = self.kernel_list[self.current_kernel[cntr]]
            population_size = int(np.ceil(len(self.surrogate.population) / self.T[cntr]))
            training_population = self.survival.do(self.problem, population, population_size,
                                                   gen=self.n_gen, max_gen=self.max_gen)

            x = training_population.extract_var()
            y = training_population.extract_obj()
            if self.T[cntr] == 1:
                l_b = self.problem.x_lower
                u_b = self.problem.x_upper
            else:
                l_b = np.min(x, axis=0)
                u_b = np.max(x, axis=0)
            selected = self.select_individuals(l_b, u_b, population)
            x = population[selected].extract_var()
            y = population[selected].extract_obj()
            # check for duplicates
            is_duplicate = self.check_duplications(population[selected], other=None, epsilon=self.duplication_tolerance)
            if np.sum(is_duplicate) > 0:
                x_new = np.array([])
                y_new = np.array([])
                for cntr in range(len(x)):
                    if not is_duplicate[0][cntr]:
                        x_new = np.append(x_new, x[cntr, :])
                        y_new = np.append(y_new, y[cntr, :])
                y_new = np.atleast_2d(y_new.T)
                x_new = np.reshape(x_new, (-1, x.shape[1]))
            else:
                x_new = copy.deepcopy(x)
                y_new = copy.deepcopy(y)

            if isinstance(self.surrogate.obj_surrogates[0], RadialBasisFunctions):
                temp_surrogate = RadialBasisFunctions(n_dim=self.problem.n_var,
                                                      l_b=l_b,
                                                      u_b=u_b,
                                                      c=0.5,
                                                      p_type='linear',
                                                      kernel_type=self.surrogate.obj_surrogates[
                                                          0].model.kernel_type)

            temp_surrogate.add_points(x_new, y_new.flatten())
            temp_surrogate.train()

            obj_surrogates.append(temp_surrogate)
        return obj_surrogates

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
        offspring1 = self.rand_1_bin(var_array, f, cr, population_size=len(var_array))
        f, cr = self.assign_f_and_cr_from_pool()
        offspring2 = self.current_to_randbest_1_bin(var_array, personal_best_array, f, cr,
                                                    population_size=len(var_array))

        # merge everything into a very large population
        merged_population = Population.merge(initial_population, offspring1)
        merged_population = Population.merge(merged_population, offspring2)
        merged_population = Population.merge(merged_population, offspring3)
        merged_population = Population.merge(merged_population, offspring4)
        merged_population = Population.merge(merged_population, offspring5)
        merged_population = Population.merge(merged_population, offspring1)
        merged_population = Population.merge(merged_population, offspring2)

        return merged_population

    def prediction_accuracy(self, optimal_population, obj_surrogates):
        x_var = optimal_population.extract_var()
        obj_values = optimal_population.extract_obj()
        prediction_accuracy = np.zeros(len(obj_surrogates))
        for surrogate_nr in range(len(obj_surrogates)):
            surrogate = obj_surrogates[surrogate_nr]
            accuracy = 0
            if len(optimal_population) > 1:
                for optima_nr in range(len(optimal_population)):
                    x_temp = x_var[optima_nr, :]
                    opt_pred = surrogate.predict(x_temp)
                    accuracy += np.abs(obj_values[optima_nr] - opt_pred)
            else:
                x_temp = x_var
                opt_pred = surrogate.predict(x_temp)
                accuracy += np.abs(obj_values - opt_pred)

            prediction_accuracy[surrogate_nr] = accuracy
        return prediction_accuracy

    def predict_function_nrmse(self):
        # Extract Surrogate Population
        temp_population = copy.deepcopy(self.surrogate.population)
        l_b = self.problem.x_lower
        u_b = self.problem.x_upper
        x_train = temp_population.extract_var()
        y_train = temp_population.extract_obj().flatten()
        dF = np.max(y_train) - np.min(y_train)

        # Train 2 surrogate instances
        mars_surrogate = MARSRegression(n_dim=self.problem.n_var, l_b=l_b,  u_b=u_b,
                                                      max_terms=self.problem.n_var, max_degree=2)
        mars_surrogate.add_points(x_train, y_train)
        mars_surrogate.train()

        rbf_surrogate = RadialBasisFunctions(n_dim=self.problem.n_var, l_b=l_b, u_b=u_b, c=0.5,
                                                      p_type='linear', kernel_type='cubic')
        rbf_surrogate.add_points(x_train, y_train)
        rbf_surrogate.train()

        # Create evaluation points
        initialise = LatinHypercubeSampling(iterations=100)
        initialise.do(2 * len(temp_population), x_lower=l_b, x_upper=u_b)
        lhs_var = initialise.x

        # Evaluate Surrogate Predictions at LHS points
        mars_prediction = np.zeros(len(lhs_var))
        rbf_prediction = np.zeros(len(lhs_var))
        for i in range(len(lhs_var)):
            mars_prediction[i] = mars_surrogate.predict(lhs_var[i])
            rbf_prediction[i] = rbf_surrogate.predict(lhs_var[i])

        # Calculate Normalised RMSE
        RMSE = np.sqrt(np.mean((rbf_prediction - mars_prediction)**2))
        NRMSE = RMSE / dF
        print('RMSE : ', np.round(RMSE, 2), 'NRMSE : ', np.round(NRMSE, 3))


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
