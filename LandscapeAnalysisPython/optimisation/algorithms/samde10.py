import numpy as np
from scipy.spatial import distance
import copy
import random
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-talk')
np.set_printoptions(suppress=True)
# matplotlib.use('TkAgg')

# from optimisation.model.algorithm import Algorithm
from optimisation.algorithms.evolutionary_algorithm import EvolutionaryAlgorithm

from optimisation.surrogate.models.ensemble import EnsembleSurrogate
from optimisation.surrogate.models.rbf import RadialBasisFunctions
from optimisation.surrogate.models.mars import MARSRegression

from optimisation.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from optimisation.operators.survival.rank_and_crowding_survival import RankAndCrowdingSurvival
from optimisation.util.non_dominated_sorting import NonDominatedSorting
from optimisation.model.population import Population
from optimisation.model.repair import BasicBoundsRepair


class SAMDE(EvolutionaryAlgorithm):
    """
    Surrogate-Assisted Multimodal Differential Evolution
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

        # SAMDE Parameters
        self.n_infill = 3
        self.T = np.array([1, 1, 1])
        self.n_clusters = [3, 3, 3]
        self.cluster_centres = [[], [], []]

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

        self.population = copy.deepcopy(self.surrogate.population)

    def _next(self):

        ## 0. Create Surrogates in Subspaces
        try:
            obj_surrogates = self.create_surrogates()
        except:
            population_copy = copy.deepcopy(self.surrogate.population)
            is_duplicate = self.check_duplications(population_copy[:-1], population_copy[-1], epsilon=1e-3)
            self.surrogate.population = copy.deepcopy(population_copy[np.invert(is_duplicate)[0]])
            obj_surrogates = self.create_surrogates()

        ## 1. Surrogate Exploitation
        exploit_population = self.surrogate_exploitation(obj_surrogates, nr_survived=self.n_infill)  # Max 3 points

        # Duplicates within the exploit population
        is_duplicate = self.check_duplications(exploit_population, other=None, epsilon=self.duplication_tolerance)
        exploit_population = exploit_population[np.invert(is_duplicate)[0]]

        # Duplicates with the surrogate population
        is_duplicate = self.check_duplications(exploit_population, self.surrogate.population, epsilon=self.duplication_tolerance)
        exploit_population = exploit_population[np.invert(is_duplicate)[0]]

        # Evaluate with expensive function
        exploit_population = self.evaluator.do(self.problem.obj_func, self.problem, exploit_population)
        self.surrogate.population = Population.merge(self.surrogate.population, exploit_population)
        n_exploit = len(exploit_population)
        print('Exploit infill: ', n_exploit)

        ## 1.0 Surrogate-Assisted Multimodal DE Exploration
        top_ranked = self.survival.do(self.problem, self.surrogate.population, self.n_population)
        explore_population = self.local_minima_exploitation(obj_surrogates, top_ranked, n_population=self.n_population,
                                                            n_generations=20, n_survive=2)  # Max 6 points

        # Duplicates within the local minima exploration population
        is_duplicate = self.check_duplications(explore_population, other=None, epsilon=self.duplication_tolerance)
        explore_population = explore_population[np.invert(is_duplicate)[0]]

        # Duplicates with the surrogate population
        is_duplicate = self.check_duplications(explore_population, self.surrogate.population, epsilon=self.duplication_tolerance)
        explore_population = explore_population[np.invert(is_duplicate)[0]]

        # Evaluate with expensive function
        explore_population = self.evaluator.do(self.problem.obj_func, self.problem, explore_population)
        self.surrogate.population = Population.merge(self.surrogate.population, explore_population)
        n_explore = len(explore_population)
        print('Explore infill: ', n_explore)

        ## 2. Update Optimum
        optimal_population = Population.merge(exploit_population, explore_population)
        old_opt = copy.deepcopy(self.opt.obj)
        opt = RankAndCrowdingSurvival().do(self.problem, self.surrogate.population, 1, None, None)
        self.opt = opt[0]

        ## 3. Kernel Switching
        if self.switch_kernels:
            self.iterations_since_switch += 1
            improvement = self.opt.obj - old_opt

            if improvement == 0:
                self.iterations_since_improvement += 1
            else:
                self.iterations_since_improvement = 0
                self.last_improvement = np.max((self.last_improvement, len(self.surrogate.population)))

            if self.iterations_since_improvement >= self.switch_after_iterations:
                test_population = copy.deepcopy(optimal_population)
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

    def local_minima_exploitation(self, obj_surrogates, init_population, n_population=30, n_generations=5, n_survive=2):

        # Calculate each of the surrogates
        optima = None

        for cntr in range(len(obj_surrogates)):
            surrogate = obj_surrogates[cntr]

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
            for cntr1 in range(len(x_var)):
                y_var[cntr1, 0] = surrogate.predict(x_var[cntr1])
            de_population.assign_obj(y_var)
            pareto_population = copy.deepcopy(de_population)

            for iteration_counter in range(n_generations):

                # Create Offspring
                offspring_population = self.create_offspring(de_population, de_population)
                offspring_variables = offspring_population.extract_var()

                # limit to the domain size
                offspring_variables = self.repair_out_of_bounds(self.problem.x_lower, self.problem.x_upper, offspring_variables)

                # Surrogate screening
                offspring_objectives = np.zeros((len(offspring_population), 1))
                for cntr2 in range(len(offspring_population)):
                    offspring_objectives[cntr2, 0] = surrogate.predict(offspring_variables[cntr2, :])
                offspring_population.assign_obj(offspring_objectives)

                # Survival for next generation
                # Two Objective Reference Point Ranking for Modality Preservation
                min_idx = np.argmin(offspring_objectives)
                current_best = offspring_variables[min_idx]
                offspring_distances = distance.cdist(np.atleast_2d(current_best), offspring_variables)
                obj_array = np.hstack((offspring_objectives, -offspring_distances[0].reshape(len(offspring_distances[0]), 1)))
                # reference_indices = self.reference_point_selection(obj_array, n_population)
                # offspring_population = offspring_population[reference_indices]
                offspring_population = self.modal_sort_and_ranking(offspring_population, surrogate, n_population, cosine=True)
                de_population = copy.deepcopy(offspring_population)

                # Update Pareto Archive
                pareto_population = self.update_archive(pareto_population, offspring_population, population_size=n_population)

            # Final survival for infill points
            infill_population = self.modal_sort_and_ranking(de_population, surrogate, n_survive, cosine=False)
            infill_var = infill_population.extract_var()

            # Store optimal variables
            if optima is not None:
                optima = np.vstack((optima, infill_var))
            else:
                optima = infill_var

        # Assign optimum points to population
        optima_population = Population(self.problem, len(optima))
        optima_population.assign_var(self.problem, optima)

        return optima_population

    def reference_point_selection(self, obj_array, n_survived, points_per_ref=1):
        # Select how many points to be survived per reference point
        n_ref = int(n_survived / points_per_ref)
        ref_intervals = np.linspace(-1.0, 0.0, n_ref)

        # Generate n_ref Reference Points
        min_d = np.min(obj_array[:, 1], axis=0)
        reference_var = np.zeros((n_ref, len(obj_array[0, :])))
        reference_var[:, 1] = ref_intervals

        # Normalise Objectives
        max_f = np.max(obj_array[:, 0], axis=0)
        min_f = np.min(obj_array[:, 0], axis=0)
        obj_array[:, 0] = (obj_array[:, 0] - min_f) / (max_f - min_f)
        obj_array[:, 1] = -obj_array[:, 1] / min_d
        # copy_obj_array = copy.deepcopy(obj_array)

        # Find the closest points in PF to reference vectors
        survived_indices = None
        for i in reversed(range(n_ref)):
            dist = distance.cdist(np.atleast_2d(reference_var[i]), obj_array)[0]

            # Set a large distance if points already selected
            if survived_indices is not None:
                dist[survived_indices] = 1e9
            indices = np.argpartition(dist, points_per_ref)[:points_per_ref]

            # Store array of indices
            if survived_indices is not None:
                survived_indices = np.hstack((survived_indices, indices))
            else:
                survived_indices = indices

        # # Plot Check
        # fig, ax = plt.subplots(1, 1, figsize=(9, 7))
        # ax.scatter(obj_array[:, 1], obj_array[:, 0], color='black', marker='o', s=60, label='Offspring Population')
        # ax.scatter(obj_array[survived_indices, 1], obj_array[survived_indices, 0], color='cyan', marker='+', s=60, label='Reference Point Ranking')
        # ax.scatter(reference_var[:, 1], reference_var[:, 0], color='red', marker='o', s= 100, label='Reference Points')
        # ax.set_xlabel('Distance from Current Best (Negative)')
        # ax.set_ylabel('Surrogate Prediction')
        # ax.legend(loc='upper right')
        # plt.show()

        # Return Unique Survived Indices
        return np.unique(survived_indices)

    def modal_sort_and_ranking(self, population, surrogate, n_survived, stop_at_cluster=10, cosine=True):
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
                flag = self.HVT(current_best, x_q, surrogate, samples=samples)
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

    def rank_multiple_objectives(self, objectives_array):

        # Conduct ranking for each constraint
        rank_for_cons = np.zeros(cons_values.shape)
        for cntr in range(problem.n_con):
            cons_to_be_ranked = cons_values[:, cntr]
            fronts_to_be_ranked = NonDominatedSorting().do(cons_to_be_ranked.reshape((len(pop), 1)),
                                                           n_stop_if_ranked=len(pop))
            rank_for_cons[:, cntr] = self.rank_front_only(fronts_to_be_ranked, (len(pop)))
        rank_constraints = np.sum(rank_for_cons, axis=1)

        # Sum the normalised scores
        rank_objectives = np.sum(temp_obj, axis=1)

        # Simple Ranking
        objs_to_be_ranked = rank_objectives
        fronts_to_be_ranked = NonDominatedSorting().do(objs_to_be_ranked.reshape((len(temp_pop), 1)),
                                                       n_stop_if_ranked=len(temp_pop))
        rank_for_objs = self.rank_front_only(fronts_to_be_ranked, len(temp_pop))
        survivors = rank_for_objs.argsort()[:n_survive]

        # Survived population
        dummy_pop = Population(self.problem, len(survivors))
        dummy_pop.assign_obj(temp_obj[survivors])
        dummy_pop.assign_var(self.problem, temp_var[survivors, :])

        return dummy_pop

    @staticmethod
    def rank_front_only(fronts, n_survive):

        cntr_rank = 1
        rank = np.zeros(n_survive)
        for k, front in enumerate(fronts):

            # Save rank and crowding to the individuals
            for j, i in enumerate(front):
                rank[i] = cntr_rank

            cntr_rank += len(front)

        return rank

    def HVT(self, x_p, x_q, surrogate, samples=[0.2, 0.5, 0.75]):
        # Handle the case when x_p and x_q are the same
        if np.all(x_p == x_q):
            return 0

        # Function values at points p and q
        y_p = surrogate.predict(x_p)
        y_q = surrogate.predict(x_q)
        y_max = np.max((y_p, y_q))

        x_init = np.zeros((len(samples), self.problem.n_var))
        modality = np.zeros(len(samples))
        for i in range(len(samples)):
            # Calculate minimum fitness interior points between p and q
            x_init[i, :] = x_p + (x_q - x_p) * samples[i]

            # Determine modality using interior points
            dy = y_max - surrogate.predict(x_init[i, :])
            # print(dy)
            if dy < 0.0:
                modality[i] = 1

        # Return 0 if modality corresponds, or 1 if not of the same modality
        if np.sum(modality) == 0.0:
            return 0
        else:
            return 1

    def surrogate_exploitation(self, obj_surrogates, generations=50, population_size=20,
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

        # Assign optimum points to population
        optima_population = Population(self.problem, len(optima))
        optima_population.assign_var(self.problem, optima)

        return optima_population

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
