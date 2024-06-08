import numpy as np
from scipy.spatial import distance
from sklearn.cluster import KMeans
import random
from scipy.stats import cauchy, norm
import copy
import warnings
import random

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
from optimisation.operators.survival.multiple_constraints_ranking_survival import MultipleConstraintsRankingSurvival

from optimisation.util.non_dominated_sorting import NonDominatedSorting
from optimisation.util.rank_fronts import rank_fronts

from optimisation.surrogate.models.ensemble import EnsembleSurrogate
from optimisation.surrogate.models.rbf import RadialBasisFunctions
from optimisation.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling


class MODE(EvolutionaryAlgorithm):
    """
    multi-offspring DE - should only be used with surrogate as this creates a lot of offspring
    """

    def __init__(self,
                 n_population=100,
                 sampling=LatinHypercubeSampling(),
                 selection=RandomSelection(),
                 survival=None,
                 crossover=None,
                 mutation=None,
                 **kwargs):

        # archives
        self.pareto_archive = None
        self.adaptive_archive = None
        self.adaptive_archive_size = int(1.4 * n_population)

        # memory for f and CR
        self.f = [0.8, 1.0, 1.0, 0.8, 0.8, 0.9, 0.4, 0.4]
        self.cr = [0.2, 0.1, 0.9, 0.8, 0.6, 0.2, 0.2, 0.9]

        # survival
        if survival is not None:
            self.survival = survival
        else:
            survival = RankAndCrowdingSurvival()
            self.survival = RankAndCrowdingSurvival()

        # set the list of sub-population sizes (see SATLBO from Dong2021b)
        self.surrogate_size = [1, 3, 5]
        # self.surrogate_size = [1, 5]
        # self.surrogate_size = [1, 3]
        self.surrogate_size_local = 5

        self.local_and_global = False
        self.surrogate_sizes = [1, 3]

        # settings for the internal optimisation runs
        self.generations = 30
        self.population_size = 20

        # settings for the final optimisation run on the real objective function
        self.final_generations = 0
        self.population_size_final_generations = 10
        # cutoff that decides if LHS initialisation is used
        self.random_cutoff = 0.25
        self.use_infill = False
        self.infill_criterion = 'mu'
        self.use_infill_after_iterations = 30
        self.last_improvement = 60
        self.infill_chance = 0.10
        self.infill_range = 0.1  # domain size for exploration
        self.infill_range_list = [0.01, 0.025, 0.05, 0.075]
        self.duplication_tolerance = 1e-4
        self.use_both = True
        self.exploration_iterations = 2
        self.start_exploration = 3 # gets multiplied by number of dimension
        self.LHS_chance = 0
        self.LHS_range = 0.025

        self.use_extra_DoE = False
        self.extra_DoE_size = 1 # gets multiplied by n_var
        self.extra_DoE_time = 0.25 # fraction of total function calls
        self.extra_DoE_domain = 0.05 # domain size

        self.switch_after_iterations = 5

        self.kernel_list = list(['cubic', 'matern_52', 'matern_32', 'matern_12', 'gaussian',
                                 'tps', 'multiquadratic', 'inv_multiquadratic', 'linear'])
        # self.kernel_list = list(['cubic', 'matern_52', 'gaussian'])
        # self.kernel_list = list(['cubic', 'matern_52', 'matern_32', 'matern_12', 'gaussian', 'linear'])
        self.switch_kernels = True
        self.iterations_since_switch = 0
        self.force_after_iterations = None

        # drop bad kernels from kernel list after the optimisation has progressed past the treshold
        # see this as a fraction of the total number of real function calls
        self.drop_threshold = 0.25

        self.rbf_c = 0.5
        self.c_list = [0.5]

        self.use_global = True
        self.switch_tail = True
        self.linear_tail = True
        self.constant_tail = False
        self.no_tail = False

        self.number_of_kernels = 3
        self.accuracy_threshold = 10
        self.predict_all = False

        # TODO - when had to run extra loop switch between global and local

        super().__init__(n_population=n_population,
                         sampling=sampling,
                         selection=selection,
                         crossover=crossover,
                         mutation=mutation,
                         survival=survival,
                         n_offspring=n_population,
                         **kwargs)

    def _initialise(self):

        self.max_f_eval = self.max_gen + self.surrogate.n_training_pts
        print(self.max_f_eval)

        # Set up the archives
        self.pareto_archive = Population(self.problem, 0)
        self.adaptive_archive = Population(self.problem, 0)

        self.force_after_iterations = int(self.problem.n_var / self.number_of_kernels)

        # multiply by number of variables
        self.start_exploration *= self.problem.n_var/self.max_f_eval

        ## VARIABLE FOR REAL F EVAL POPULATION
        self.n_pop_old = self.n_population

        ## HARDCODE NUMBER OF INFILL CRITERIA
        self.duplicate_count = 0
        self.e_flag = True
        self.e_tol = 1.5 / self.problem.n_var
        self.infeasible_flag = False

        # Initialise from the evolutionary algorithm class
        super()._initialise()

        self.training_data = None
        # update pareto archive as non-dominated solutions from pareto archive and population
        self.update_pareto_archive()

        if self.surrogate.sampling_strategy == 'mu':
            self.use_infill = False
        else:
            self.use_infill = True

        if self.local_and_global:
            self.surrogate_size = [self.surrogate_sizes[0]]
            self.iterations_since_improvement = 0

        if self.switch_kernels:
            self.iterations_since_improvement = 0
            # self.use_global = True

            # kernel_nr = np.random.randint(low=0, high=len(self.kernel_list), size=(self.number_of_kernels,))
            kernel_nr = random.sample(range(len(self.kernel_list)), self.number_of_kernels)
            self.current_kernel = copy.deepcopy(kernel_nr)
            for nr in range(self.number_of_kernels):
                print(self.kernel_list[kernel_nr[nr]])

            # self.surrogate.obj_surrogates[0].model.kernel_type = self.kernel_list[kernel_nr]

    def _next(self):

        if self.surrogate is None:
            print('you should not use this algorithm if you are not using surrogates')
            raise ValueError("you should not use MODE if you are not using surrogates")

        if not isinstance(self.surrogate.obj_surrogates[0], (EnsembleSurrogate, RadialBasisFunctions)):
            raise ValueError("has only been set up for surrogates for now. You can extend this easily")

        if len(self.surrogate.obj_surrogates) > 1:
            raise ValueError("for now only set up for single objective")

        randint = np.random.randint(len(self.c_list))
        self.rbf_c = self.c_list[randint]

        randint = np.random.randint(len(self.infill_range_list))
        self.infill_range = self.infill_range_list[randint]

        try:
            obj_surrogates = self.create_surrogates()
        except:
            population_copy = copy.deepcopy(self.surrogate.population)
            is_duplicate = self.check_duplications(population_copy[:-1], population_copy[-1], epsilon=1e-2)
            self.surrogate.population = copy.deepcopy(population_copy[np.invert(is_duplicate)[0]])
            obj_surrogates = self.create_surrogates()

        ## Check for any feasible points
        if self.surrogate.cons_surrogates != []:
            raise ValueError("for now only set up for unconstrained")
            # todo duplicate the loop for the objective surrogates here

        # # choose infill or not
        # rand_nr = np.random.random()
        # # rand_cutoff = 1 - (len(self.surrogate.population)/self.max_f_eval)*self.infill_chance
        # rand_cutoff = (1-self.infill_chance)*((len(self.surrogate.population)/self.max_f_eval))+self.infill_chance
        # if rand_nr > rand_cutoff:
        #     self.use_infill = True
        #     self.surrogate.sampling_strategy = self.infill_criterion
        #     print('using infill')
        # else:
        #     self.use_infill = False
        #     self.surrogate.sampling_strategy = 'mu'
        #     print('using mu')

        if self.use_both:
            self.use_infill = False
            print('exploitation')

        # find the optimal solution for each surrogate using MODE
        if not self.use_infill:
            try:
                rand_nr = np.random.random()
                LHS_cutoff = len(self.surrogate.population)/self.max_f_eval * self.LHS_chance
                if len(self.surrogate.population)/self.max_f_eval > self.start_exploration:
                    if rand_nr < LHS_cutoff:
                        optima = self.surrogate_exploitation(obj_surrogates,
                                                             generations=self.generations,
                                                             population_size=self.population_size,
                                                             nr_survived=1,
                                                             use_lhs=True)

                    else:
                        optima = self.surrogate_exploitation(obj_surrogates,
                                                             generations=self.generations,
                                                             population_size=self.population_size,
                                                             nr_survived=1)
                else:
                    optima = self.surrogate_exploitation(obj_surrogates,
                                                         generations=self.generations,
                                                         population_size=self.population_size,
                                                         nr_survived=1)

                # remove duplicates inside optima
                if len(optima) > 0:
                    optimal_population = Population(self.problem, n_individuals=len(optima))
                    optimal_population.assign_var(self.problem, optima)
                    if len(optima) > 1:
                        is_duplicate = self.check_duplications(optimal_population, other=None,
                                                               epsilon=self.duplication_tolerance)
                        optimal_population = optimal_population[np.invert(is_duplicate)[0]]
                    # remove duplicates
                    is_duplicate = self.check_duplications(optimal_population, self.surrogate.population,
                                                           epsilon=self.duplication_tolerance)
                    optimal_population = optimal_population[np.invert(is_duplicate)[0]]
                else:
                    is_duplicate = [True]
            except np.linalg.LinAlgError:
                print('linalg error')
                is_duplicate = self.check_duplications(self.surrogate.population, self.surrogate.population)

            if sum(is_duplicate[0]) == len(is_duplicate[0]):
                # all are false so need to redo
                print('had to run extra loop')
                population = copy.deepcopy(self.surrogate.population)
                population_size = int(np.ceil(len(self.surrogate.population) / self.surrogate_size_local))
                training_population = self.survival.do(self.problem, population, population_size,
                                                       gen=self.n_gen, max_gen=self.max_gen)

                x = training_population.extract_var()
                # selected = self.select_individuals(l_b, u_b, population)
                # x = population[selected].extract_var()
                # get the mean of the optima

                temp_optimum = np.mean(x, axis=0)
                # range
                low_bound = np.min(x, axis=0)
                up_bound = np.max(x, axis=0)
                new_optima = copy.deepcopy(optima)
                for cntr in range(len(optima)):
                    rand_nrs = -0.5 + np.random.random(30)
                    new_optima[cntr, :] = temp_optimum + rand_nrs * (up_bound - low_bound)
                try:
                    if len(new_optima) > 0:
                        if len(new_optima) > 1:
                            # remove duplicates inside optima
                            optimal_population = Population(self.problem, n_individuals=len(new_optima))
                            optimal_population.assign_var(self.problem, new_optima)
                            is_duplicate = self.check_duplications(optimal_population, other=None,
                                                                   epsilon=self.duplication_tolerance)
                            optimal_population = optimal_population[np.invert(is_duplicate)[0]]
                        # remove duplicates
                        if len(optimal_population) > 0:
                            is_duplicate = self.check_duplications(optimal_population, self.surrogate.population,
                                                                   epsilon=self.duplication_tolerance)
                            optimal_population = optimal_population[np.invert(is_duplicate)[0]]

                except np.linalg.LinAlgError:
                    print('linalg error')
                    optimal_population = Population(self.problem, n_individuals=0)
                finally:
                    self.iterations_since_improvement += self.switch_after_iterations

            # need to add the optima to the surrogates and to self.surrogate.population
            optimal_population = self.evaluator.do(self.problem.obj_func, self.problem, optimal_population)

            old_surrogate_pop = copy.deepcopy(self.surrogate.population)

            self.surrogate.population = Population.merge(self.surrogate.population, optimal_population)
            # self.surrogate.population = copy.deepcopy(self.surrogate.population)
            temp_values = optimal_population.extract_obj()
            print(temp_values)

            # if self.predict_all:
            #     self.current_kernel = range(len(self.kernel_list))
            #     obj_surrogates = self.create_surrogates()
            if len(optimal_population) > 0:
                accuracy = self.prediction_accuracy(optimal_population, obj_surrogates)

            try:
                obj_surrogates = self.create_surrogates()
            except:
                self.surrogate.population = old_surrogate_pop
                obj_surrogates = self.create_surrogates()
        # self.surrogate.obj_surrogates[0] = copy.deepcopy(obj_surrogates[0])

        if (len(self.surrogate.population)/self.max_f_eval) > self.start_exploration:
            if np.random.random() < self.infill_chance:
                if self.use_both:
                    self.use_infill = True
                    print('exploration')
                    print('infill range = {}'.format(self.infill_range))

        # exploration using the infill criterion
        if self.use_infill:
            try:
                if np.random.random() < 0.5:
                    print('explore LHS')
                    optima = self.surrogate_exploration(obj_surrogates,
                                                        generations=self.exploration_iterations,
                                                        population_size=self.population_size,
                                                        use_lhs=True)
                else:
                    # print('exploit')
                    # optima = self.surrogate_exploitation(obj_surrogates,
                    #                                     generations=self.exploration_iterations,
                    #                                     population_size=self.population_size,
                    #                                     nr_survived=1,
                    #                                     use_lhs=True)
                    print('explore no LHS')
                    optima = self.surrogate_exploration(obj_surrogates,
                                                        generations=self.exploration_iterations,
                                                        population_size=self.population_size,
                                                        use_lhs=False)
                # remove duplicates inside optima
                optimal_population = Population(self.problem, n_individuals=len(optima))
                optimal_population.assign_var(self.problem, optima)
                is_duplicate = self.check_duplications(optimal_population, other=None,
                                                       epsilon=self.duplication_tolerance)
                optimal_population = optimal_population[np.invert(is_duplicate)[0]]
                # remove duplicates
                is_duplicate = self.check_duplications(optimal_population, self.surrogate.population,
                                                       epsilon=self.duplication_tolerance)
                optimal_population = optimal_population[np.invert(is_duplicate)[0]]
            except np.linalg.LinAlgError:
                print('linalg error')
                is_duplicate = self.check_duplications(self.surrogate.population, self.surrogate.population)

            if sum(is_duplicate[0]) == len(is_duplicate[0]):
                # all are false so need to redo
                print('had to run extra loop')
                self.switch_local_and_global()
                # surrogate_size_original = copy.deepcopy(self.surrogate_size[0])
                # self.surrogate_size[0] = copy.deepcopy(self.surrogate_size_local)
                try:
                    optima = self.surrogate_exploration(obj_surrogates,
                                                        generations=self.exploration_iterations,
                                                        population_size=self.population_size,
                                                        nr_survived=3,
                                                        use_lhs=True)
                    # self.surrogate_size[0] = copy.deepcopy(surrogate_size_original)

                    # remove duplicates inside optima
                    optimal_population = Population(self.problem, n_individuals=len(optima))
                    optimal_population.assign_var(self.problem, optima)
                    is_duplicate = self.check_duplications(optimal_population, other=None,
                                                           epsilon=self.duplication_tolerance)
                    optimal_population = optimal_population[np.invert(is_duplicate)[0]]
                    # remove duplicates
                    is_duplicate = self.check_duplications(optimal_population, self.surrogate.population,
                                                           epsilon=self.duplication_tolerance)
                    optimal_population = optimal_population[np.invert(is_duplicate)[0]]
                except np.linalg.LinAlgError:
                    print('linalg error')
                    optimal_population = Population(self.problem, 0)
                finally:
                    self.switch_local_and_global()

            # need to add the optima to the surrogates and to self.surrogate.population
            optimal_population = self.evaluator.do(self.problem.obj_func, self.problem, optimal_population)

            self.surrogate.population = Population.merge(self.surrogate.population, optimal_population)
            # self.surrogate.population = copy.deepcopy(self.surrogate.population)
            temp_values = optimal_population.extract_obj()
            print(temp_values)

            accuracy = self.prediction_accuracy(optimal_population, obj_surrogates)
            # is_duplicate = self.check_duplications(self.surrogate.population, optimal_population)
            # optimal_population = copy.deepcopy(self.surrogate.population)
            # optimal_population = optimal_population[np.invert(is_duplicate)[0]]
            # self.surrogate.population = optimal_population
            obj_surrogates = self.create_surrogates()
            self.surrogate.obj_surrogates[0] = copy.deepcopy(obj_surrogates[0])

        ## Update optimum
        old_opt = copy.deepcopy(self.opt.obj)

        if self.surrogate is not None:
            # if self.problem.n_obj == 1:
            opt = RankAndCrowdingSurvival().do(self.problem, self.surrogate.population, 1, None, None)
        else:
            opt = RankAndCrowdingSurvival().do(self.problem, self.population, 1, None, None)
        self.opt = opt[0]

        # if len(self.surrogate.population) >= self.max_f_eval * self.drop_threshold:
        #     self.random_cutoff = 0.25

        if self.local_and_global:
            improvement = self.opt.obj - old_opt

            if improvement == 0:
                self.iterations_since_improvement += 1
            else:
                self.iterations_since_improvement = 0

            if self.iterations_since_improvement > self.switch_after_iterations:
                self.switch_local_and_global()
                self.iterations_since_improvement = 0

        if self.switch_kernels:
            self.iterations_since_switch += 1
            improvement = self.opt.obj - old_opt

            if improvement == 0:
                self.iterations_since_improvement += 1
            else:
                self.iterations_since_improvement = 0
                self.last_improvement = np.max((self.last_improvement, len(self.surrogate.population)))

            if self.iterations_since_improvement >= self.switch_after_iterations:

                if self.predict_all:
                    print('trying to find best kernel')
                    self.current_kernel = range(len(self.kernel_list))
                    obj_surrogates = self.create_surrogates()
                    if len(optimal_population) > 0:
                        accuracy = self.prediction_accuracy(optimal_population, obj_surrogates)
                    else:
                        # use random point
                        rand_var = np.random.random(self.problem.n_var)
                        rand_var -= 0.5
                        rand_var *= 0.025
                        # rand_var = np.atleast_2d(rand_var)
                        optimum_location = self.opt.var
                        rand_location = optimum_location + rand_var *(self.problem.x_upper - self.problem.x_lower)
                        rand_location = np.max((rand_location, self.problem.x_lower), axis=0)
                        rand_location = np.min((rand_location, self.problem.x_upper), axis=0)
                        optimal_population = Population(self.problem, 1)
                        rand_location = np.atleast_2d(rand_location)
                        optimal_population.assign_var(self.problem, rand_location)
                        optimal_population = self.evaluator.do(self.problem.obj_func, self.problem, optimal_population)

                        accuracy = self.prediction_accuracy(optimal_population, obj_surrogates)

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

                # force a kernel that has been kept for a long time to go out
                # force remove a kernel with bad prediction capability

                # keep the most accurate kernel and swap out the other x
                # self.surrogate.obj_surrogates[0].model.kernel_type = self.kernel_list[kernel_nr]
                self.iterations_since_improvement = 0
                self.iterations_since_switch = 0
                if self.switch_tail:
                    random_number = np.random.rand()
                    if random_number <= 0.5:
                        self.linear_tail = True
                        self.constant_tail = False
                        self.no_tail = False
                    elif random_number <= 0.75:
                        self.linear_tail = False
                        self.constant_tail = True
                        self.no_tail = False
                    else:
                        self.linear_tail = False
                        self.constant_tail = False
                        self.no_tail = True

                # if np.random.random() < 0.5:
                #     self.switch_local_and_global()

            if self.iterations_since_switch > self.force_after_iterations:
                if self.predict_all:
                    print('trying to find best kernel')
                    self.current_kernel = range(len(self.kernel_list))
                    obj_surrogates = self.create_surrogates()
                    if len(optimal_population) > 0:
                        accuracy = self.prediction_accuracy(optimal_population, obj_surrogates)
                    else:
                        # use random point
                        rand_var = np.random.random(self.problem.n_var)
                        rand_var -= 0.5
                        rand_var *= 0.025
                        # rand_var = np.atleast_2d(rand_var)
                        optimum_location = self.opt.var
                        rand_location = optimum_location + rand_var *(self.problem.x_upper - self.problem.x_lower)
                        rand_location = np.max((rand_location, self.problem.x_lower), axis=0)
                        rand_location = np.min((rand_location, self.problem.x_upper), axis=0)
                        optimal_population = Population(self.problem, 1)
                        rand_location = np.atleast_2d(rand_location)
                        optimal_population.assign_var(self.problem, rand_location)
                        optimal_population = self.evaluator.do(self.problem.obj_func, self.problem, optimal_population)

                        accuracy = self.prediction_accuracy(optimal_population, obj_surrogates)

                best_kernel = np.nanargmin(accuracy)
                kernel_to_keep = self.current_kernel[best_kernel]

                kernel_list = [i for i in range(len(self.kernel_list))]
                kernel_list.remove(kernel_to_keep)

                # make sure that the kernel that will be removed is not selected
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

                self.iterations_since_improvement = 0
                self.iterations_since_switch = 0
                if self.switch_tail:
                    random_number = np.random.rand()
                    if random_number <= 0.5:
                        self.linear_tail = True
                        self.constant_tail = False
                        self.no_tail = False
                    elif random_number <= 0.75:
                        self.linear_tail = False
                        self.constant_tail = True
                        self.no_tail = False
                    else:
                        self.linear_tail = False
                        self.constant_tail = False
                        self.no_tail = True

        ## Reset multiple infill criteria count
        self.duplicate_count = 0

        # if len(self.surrogate.population) - self.last_improvement  > self.use_infill_after_iterations:
        #     print('adding infill criterion')
        #     self.use_infill = not self.use_infill
        #     self.surrogate.sampling_strategy = self.infill_criterion
        #     self.last_improvement = len(self.surrogate.population)

        if self.use_extra_DoE:
            if len(self.surrogate.population) >= self.max_f_eval*self.extra_DoE_time:
                self.use_extra_DoE = not self.use_extra_DoE # to make sure it only gets used once
                self.extra_DoE_size *= self.problem.n_var
                print(len(self.surrogate.population))
                print('adding extra DoE')
                # calculate the new domain
                optimum_location = self.opt.var
                var_range = self.problem.x_upper - self.problem.x_lower
                lower_bound = optimum_location - self.extra_DoE_domain * var_range
                upper_bound = optimum_location + self.extra_DoE_domain * var_range

                lower_bound = np.max((lower_bound, self.problem.x_lower), axis=0)
                upper_bound = np.min((upper_bound, self.problem.x_upper), axis=0)

                # create points
                initialise = LatinHypercubeSampling(iterations=100)
                initialise.do(self.extra_DoE_size, x_lower=lower_bound, x_upper=upper_bound)
                x_init = initialise.x

                # evaluate points
                new_population = Population(self.problem, n_individuals=self.extra_DoE_size)
                new_population.assign_var(self.problem, x_init)
                is_duplicate = self.check_duplications(new_population, self.surrogate.population,
                                                       epsilon=self.duplication_tolerance)
                new_population = new_population[np.invert(is_duplicate)[0]]
                new_population = self.evaluator.do(self.problem.obj_func, self.problem, new_population)
                self.surrogate.population = Population.merge(self.surrogate.population, new_population)
                print(len(self.surrogate.population))

        if len(self.surrogate.population) >= self.max_f_eval:
            # add final iteration using the real function evaluation
            population = copy.deepcopy(self.surrogate.population)
            starting_population = self.survival.do(self.problem, population, self.population_size_final_generations,
                                                   gen=self.n_gen, max_gen=self.max_gen)
            pareto_archive = copy.deepcopy(starting_population)
            for counter in range(self.final_generations):
                var_array = starting_population.extract_var()
                personal_best_array = pareto_archive.extract_var()
                # f and cr taken from Yang2017
                offspring = self.best_1_bin(var_array, personal_best_array, f=0.9, cr=0.2,
                                            population_size=len(var_array))

                offspring_variables = offspring.extract_var()
                # limit to the domain size
                self.repair_out_of_bounds(self.problem.x_lower, self.problem.x_upper, offspring_variables)
                offspring.assign_var(self.problem, offspring_variables)
                # for i in range(len(offspring)):
                #     offspring[i].scale_var(self.problem)

                offspring = self.evaluator.do(self.problem.obj_func, self.problem, offspring)
                offspring = Population.merge(starting_population, offspring)
                offspring = self.survival.do(self.problem, offspring, self.population_size_final_generations,
                                             gen=self.n_gen, max_gen=self.max_gen)
                starting_population = copy.deepcopy(offspring)
                pareto_archive = self.update_archive(pareto_archive, offspring,
                                                     population_size=self.population_size_final_generations)
                opt = self.survival.do(self.problem, starting_population, 1, None, None)
                self.opt = opt[0]
                print('Optimum objective value:', self.opt.obj)
            self.finished = True

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

    def switch_local_and_global(self):
        if self.use_global:
            self.surrogate_size[0] = self.surrogate_sizes[1]
            print('switching to local')
        else:
            self.surrogate_size[0] = self.surrogate_sizes[0]
            print('switching to global')
        self.use_global = not self.use_global

    def create_surrogates(self):
        obj_surrogates = list()
        population = copy.deepcopy(self.surrogate.population)

        ## CLUSTER
        cluster_frac = np.array([1, 3, 5])
        rand_int = np.random.randint(0, len(cluster_frac), 1)

        min_pop = np.max((self.problem.n_var, len(population) / 5))
        pops = np.floor(np.linspace(min_pop, len(population), 3))  # 3 surrogate sizes

        for cntr in range(len(self.current_kernel)):
            population_size = pops[cntr]
            self.surrogate.obj_surrogates[0].model.kernel_type = self.kernel_list[self.current_kernel[cntr]]

            ## CLUSTER
            # population_size = int(np.ceil(len(self.surrogate.population) / self.surrogate_size[cntr]))
            # if population_size < self.problem.n_var:
            #     population_size = self.problem.n_var
            n_clusters = np.ceil(population_size / cluster_frac[rand_int])
            if n_clusters < self.problem.n_var:
                n_clusters = population_size

            print(f"Cluster size: ", n_clusters, ' of pop size: ', population_size)
            training_population = self.survival.do(self.problem, population, population_size,
                                                   gen=self.n_gen, max_gen=self.max_gen)

            x_var = training_population.extract_var()
            obj_val = training_population.extract_obj()
            kmeans = KMeans(n_clusters=int(n_clusters), max_iter=500)
            kmeans.fit(x_var)
            cluster_centres = kmeans.cluster_centers_
            labels = kmeans.labels_

            training_cluster_centres = np.zeros((len(cluster_centres), self.problem.n_var))
            training_cluster_obj = np.zeros((len(cluster_centres), self.problem.n_obj))
            for i in range(len(cluster_centres)):
                dist = distance.cdist(np.atleast_2d(cluster_centres[i]), x_var[labels == i, :])[0]
                cluster_ind = np.argmin(dist)
                training_cluster_centres[i] = x_var[labels == i, :][cluster_ind, :]
                temp_ind = training_cluster_centres[i] == x_var
                training_cluster_obj[i] = obj_val[np.where(temp_ind == True)[0][0]]

            x = training_cluster_centres
            y = training_cluster_obj

            if len(x) == len(x_var):
                l_b = self.problem.x_lower
                u_b = self.problem.x_upper
            else:
                l_b = np.min(x, axis=0)
                u_b = np.max(x, axis=0)
            # selected = self.select_individuals(l_b, u_b, population)
            # x = population[selected].extract_var()
            # y = population[selected].extract_obj()
            # check for duplicates
            # is_duplicate = self.check_duplications(population[selected], other=None, epsilon=self.duplication_tolerance)
            # if np.sum(is_duplicate) > 0:
            #     x_new = np.array([])
            #     y_new = np.array([])
            #     for cntr in range(len(x)):
            #         if not is_duplicate[0][cntr]:
            #             x_new = np.append(x_new, x[cntr, :])
            #             y_new = np.append(y_new, y[cntr, :])
            #     y_new = np.atleast_2d(y_new.T)
            #     x_new = np.reshape(x_new, (-1, x.shape[1]))
            # else:
            x_new = copy.deepcopy(x)
            y_new = copy.deepcopy(y)

            if isinstance(self.surrogate.obj_surrogates[0], EnsembleSurrogate):
                temp_surrogate = EnsembleSurrogate(n_dim=self.problem.n_var,
                                                   l_b=l_b,
                                                   u_b=u_b)
            elif isinstance(self.surrogate.obj_surrogates[0], RadialBasisFunctions):
                if self.surrogate.obj_surrogates[0].model.kernel_type.lower() == 'linear':
                    temp_surrogate = RadialBasisFunctions(n_dim=self.problem.n_var,
                                                          l_b=l_b,
                                                          u_b=u_b,
                                                          c=self.rbf_c,
                                                          p_type=None,
                                                          kernel_type=self.surrogate.obj_surrogates[
                                                              0].model.kernel_type)
                else:
                    if self.linear_tail:
                        temp_surrogate = RadialBasisFunctions(n_dim=self.problem.n_var,
                                                              l_b=l_b,
                                                              u_b=u_b,
                                                              c=self.rbf_c,
                                                              p_type='linear',
                                                              kernel_type=self.surrogate.obj_surrogates[
                                                                  0].model.kernel_type)
                    elif self.no_tail:
                        temp_surrogate = RadialBasisFunctions(n_dim=self.problem.n_var,
                                                              l_b=l_b,
                                                              u_b=u_b,
                                                              c=self.rbf_c,
                                                              p_type=None,
                                                              kernel_type=self.surrogate.obj_surrogates[
                                                                  0].model.kernel_type)
                    elif self.constant_tail:
                        temp_surrogate = RadialBasisFunctions(n_dim=self.problem.n_var,
                                                              l_b=l_b,
                                                              u_b=u_b,
                                                              c=self.rbf_c,
                                                              p_type='constant',
                                                              kernel_type=self.surrogate.obj_surrogates[
                                                                  0].model.kernel_type)

            temp_surrogate.add_points(x_new, y_new[:, 0])
            temp_surrogate.train()

            obj_surrogates.append(temp_surrogate)
        return obj_surrogates

    def check_duplications(self, pop, other, epsilon=1e-3):

        dist = self.calc_dist(pop, other)
        dist[np.isnan(dist)] = np.inf

        is_duplicate = [np.any(dist < epsilon, axis=1)]

        return is_duplicate

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

    def create_merged_population(self, offspring, trial_array, population_size=None):
        if population_size is None:
            population_size = self.n_population
        merged_population = Population.merge(self.population, offspring)
        merged_population = self.survival.do(self.problem, merged_population, population_size,
                                             gen=self.n_gen, max_gen=self.max_gen)
        # find which offspring survived
        merged_population_var_array = merged_population.extract_var()
        survived = np.zeros(population_size, dtype=bool)
        for idx in range(population_size):
            test_array = trial_array[idx, :]
            survived[idx] = (merged_population_var_array == test_array).all(1).any()
        self.population = merged_population
        return survived

    def update_archive(self, population, offspring, population_size=None):
        if population_size is None:
            population_size = self.n_population
        merged_population = Population.merge(population, offspring)
        merged_population = self.survival.do(self.problem, merged_population, population_size,
                                             gen=self.n_gen, max_gen=self.max_gen)

        return merged_population

    def surrogate_exploitation(self, obj_surrogates, generations=100, population_size=20,
                               nr_survived=1, use_lhs=False, exploitation=True):
        # need to loop through each surrogate
        optima = np.zeros((len(obj_surrogates), len(obj_surrogates[0].l_b)))

        if use_lhs:
            # calculate the new domain
            optimum_location = self.opt.var
            var_range = self.problem.x_upper - self.problem.x_lower
            lower_bound = optimum_location - self.LHS_range * var_range
            upper_bound = optimum_location + self.LHS_range * var_range

            lower_bound = np.max((lower_bound, self.problem.x_lower), axis=0)
            upper_bound = np.min((upper_bound, self.problem.x_upper), axis=0)
        else:
            lower_bound = self.problem.x_lower
            upper_bound = self.problem.x_upper

        for cntr in range(len(obj_surrogates)):
            surrogate = obj_surrogates[cntr]
            if np.random.random(1) < self.random_cutoff:
                # get the space for LHS from the top solutions identified so far
                # population = copy.deepcopy(self.surrogate.population)
                # training_population = self.survival.do(self.problem, population,
                #                                        int(population_size/self.surrogate_size_local),
                #                                        gen=self.n_gen, max_gen=self.max_gen)
                #
                # x = training_population.extract_var()
                # l_b = np.min(x, axis=0)
                # u_b = np.max(x, axis=0)
                initialise = LatinHypercubeSampling(iterations=100)
                # initialise.do(2*population_size, x_lower=l_b, x_upper=u_b)
                initialise.do(2 * population_size, x_lower=lower_bound, x_upper=upper_bound)
                x_init = initialise.x
            elif use_lhs:
                initialise = LatinHypercubeSampling(iterations=100)
                initialise.do(2 * population_size, x_lower=lower_bound, x_upper=upper_bound)
                x_init = initialise.x
            else:
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
                                                             pareto_archive=pareto_archive,
                                                             exploitation=exploitation)
                offspring_variables = offspring_population.extract_var()
                # limit to the domain size
                self.repair_out_of_bounds(lower_bound, upper_bound, offspring_variables)
                offspring_objectives = np.zeros((len(offspring_population), 1))
                for cntr2 in range(len(offspring_population)):
                    offspring_objectives[cntr2, 0] = surrogate.predict(offspring_variables[cntr2])
                offspring_population.assign_obj(offspring_objectives)
                # TODO make survival ToR with distance as replacement for constraints
                offspring_population = self.survival.do(self.problem, offspring_population, population_size,
                                                        gen=self.n_gen, max_gen=self.max_gen)
                initial_population = copy.deepcopy(offspring_population)

                pareto_archive = self.update_archive(pareto_archive, offspring_population,
                                                     population_size=population_size)

            offspring_population = self.survival.do(self.problem, offspring_population, nr_survived,
                                                    gen=self.n_gen, max_gen=self.max_gen)

            # to account for the case where nr_survived is larger than 1 - picks the last one

            if nr_survived > 1:
                temp = offspring_population.extract_var()
                optima[cntr, :] = temp[nr_survived - 1, :]
            else:
                optima[cntr, :] = offspring_population.extract_var()

        return optima

    def repair_out_of_bounds(self, lower_bound, upper_bound, var_array, **kwargs):

        # Upper and lower bounds masks
        upper_mask = var_array > upper_bound
        lower_mask = var_array < lower_bound

        # Repair variables lying outside bounds
        var_array[upper_mask] = np.tile(upper_bound, (len(var_array), 1))[upper_mask]
        var_array[lower_mask] = np.tile(lower_bound, (len(var_array), 1))[lower_mask]

        return var_array

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

    def create_offspring(self, initial_population, pareto_archive, exploitation):
        var_array = initial_population.extract_var()
        personal_best_array = pareto_archive.extract_var()
        if exploitation:
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

            # merge everything into a very large population
            merged_population = Population.merge(initial_population, offspring1)
            merged_population = Population.merge(merged_population, offspring2)
            merged_population = Population.merge(merged_population, offspring3)
            merged_population = Population.merge(merged_population, offspring4)
            merged_population = Population.merge(merged_population, offspring5)

        else:
            f, cr = self.assign_f_and_cr_from_pool()
            offspring1 = self.rand_1_bin(var_array, f, cr, population_size=len(var_array))
            f, cr = self.assign_f_and_cr_from_pool()
            offspring2 = self.current_to_randbest_1_bin(var_array, personal_best_array, f, cr,
                                                        population_size=len(var_array))
            # merge everything into a very large population
            merged_population = Population.merge(initial_population, offspring1)
            merged_population = Population.merge(merged_population, offspring2)

        return merged_population

    def create_multiple_offspring(self):
        ## MAKE SURE ONLY REAL F EVALS ARE USED
        # extract variables and create union of external archive and population
        if len(self.surrogate.population) < self.n_pop_old:
            var_array = self.population.extract_var()
            self.n_population = len(var_array)
        # Check if surrogate pop is greater than desired population
        elif len(self.surrogate.population) >= self.n_pop_old:
            best_pop = self.survival.do(self.problem, self.surrogate.population, self.n_pop_old,
                                        gen=self.n_gen, max_gen=self.max_gen)
            var_array = best_pop.extract_var()
            self.n_population = len(var_array)
        else:
            var_array = self.population.extract_var()
        # var_array = self.population.extract_var()

        personal_best_array = self.pareto_archive.extract_var()
        # create the various offspring
        f, cr = self.assign_f_and_cr_from_pool()
        offspring1 = self.rand_1_bin(var_array, f, cr)
        f, cr = self.assign_f_and_cr_from_pool()
        offspring2 = self.current_to_randbest_1_bin(var_array, personal_best_array, f, cr)
        f, cr = self.assign_f_and_cr_from_pool()
        offspring3 = self.best_1_bin(var_array, personal_best_array, f, cr)
        f1, cr = self.assign_f_and_cr_from_pool()
        f2, _ = self.assign_f_and_cr_from_pool()
        offspring4 = self.best_2_bin(var_array, personal_best_array, f1, f2, cr)
        f, cr = self.assign_f_and_cr_from_pool()
        offspring5 = self.current_to_rand_1_bin(var_array, f, cr)
        f, cr = self.assign_f_and_cr_from_pool()
        offspring6 = self.modified_rand_to_best_1_bin(var_array, personal_best_array, f, cr)
        f, cr = self.assign_f_and_cr_from_pool()
        offspring7 = self.current_to_best_1_bin(var_array, personal_best_array, f, cr)

        # merge everything into a very large population
        merged_population = Population.merge(self.population, offspring1)
        merged_population = Population.merge(merged_population, offspring2)
        merged_population = Population.merge(merged_population, offspring3)
        merged_population = Population.merge(merged_population, offspring4)
        merged_population = Population.merge(merged_population, offspring5)
        merged_population = Population.merge(merged_population, offspring6)
        merged_population = Population.merge(merged_population, offspring7)

        return merged_population

    def surrogate_exploration(self, obj_surrogates, generations=100, population_size=20, nr_survived=1,
                              use_lhs=False, exploitation=False):
        # need to loop through each surrogate
        optima = np.zeros((len(obj_surrogates), len(obj_surrogates[0].l_b)))

        # calculate the new domain
        optimum_location = self.opt.var
        var_range = self.problem.x_upper - self.problem.x_lower
        lower_bound = optimum_location - self.infill_range * var_range
        upper_bound = optimum_location + self.infill_range * var_range

        lower_bound = np.max((lower_bound, self.problem.x_lower), axis=0)
        upper_bound = np.min((upper_bound, self.problem.x_upper), axis=0)

        for cntr in range(len(obj_surrogates)):
            surrogate = obj_surrogates[cntr]
            if np.random.random(1) <= 0.5:
                initialise = LatinHypercubeSampling(iterations=100)
                initialise.do(5*population_size, x_lower=lower_bound, x_upper=upper_bound)
                x_init = initialise.x
            elif use_lhs:
                initialise = LatinHypercubeSampling(iterations=100)
                initialise.do(5*population_size, x_lower=lower_bound, x_upper=upper_bound)
                x_init = initialise.x
            else:
                if len(surrogate.x) > population_size:
                    temp_population = Population(self.problem, n_individuals=len(surrogate.x))
                    temp_population.assign_var(self.problem, surrogate.x)
                    temp_population.assign_obj(surrogate.y[:].reshape(-1, 1))
                    temp_population = self.survival.do(self.problem, temp_population, population_size,
                                                       gen=self.n_gen, max_gen=self.max_gen)
                    x_init = temp_population.extract_var()
                    self.repair_out_of_bounds(lower_bound, upper_bound, x_init)
                else:
                    initialise = LatinHypercubeSampling(iterations=100)
                    initialise.do(4*population_size, x_lower=lower_bound, x_upper=upper_bound)
                    x_init = initialise.x

            initial_population = Population(self.problem, n_individuals=population_size)
            initial_population.assign_var(self.problem, x_init)

            initial_objectives = np.zeros((population_size, 1))
            for cntr2 in range(population_size):
                initial_objectives[cntr2, 0] = self.predict_infill(surrogate, x_init[cntr2])
            initial_population.assign_obj(initial_objectives)
            pareto_archive = copy.deepcopy(initial_population)

            for iteration_counter in range(generations):
                offspring_population = self.create_offspring(initial_population=initial_population,
                                                             pareto_archive=pareto_archive,
                                                             exploitation=exploitation)
                offspring_variables = offspring_population.extract_var()
                # limit to the domain size
                self.repair_out_of_bounds(lower_bound, upper_bound, offspring_variables)
                offspring_objectives = np.zeros((len(offspring_population), 1))
                for cntr2 in range(len(offspring_population)):
                    offspring_objectives[cntr2, 0] = self.predict_infill(surrogate, offspring_variables[cntr2])
                offspring_population.assign_obj(offspring_objectives)
                offspring_population = self.survival.do(self.problem, offspring_population, population_size,
                                                        gen=self.n_gen, max_gen=self.max_gen)
                initial_population = copy.deepcopy(offspring_population)

                pareto_archive = self.update_archive(pareto_archive, offspring_population,
                                                     population_size=population_size)

            offspring_population = self.survival.do(self.problem, offspring_population, nr_survived,
                                                    gen=self.n_gen, max_gen=self.max_gen)

            # to account for the case where nr_survived is larger than 1 - picks the last one

            if nr_survived > 1:
                temp = offspring_population.extract_var()
                optima[cntr, :] = temp[nr_survived - 1, :]
            else:
                optima[cntr, :] = offspring_population.extract_var()

        return optima

    def predict_infill(self, surrogate, x):
        if self.surrogate.sampling_strategy == 'ei':
            obj = surrogate.predict_ei(x, ksi=0.01)
        elif self.surrogate.sampling_strategy == 'ei_idw':
            obj = surrogate.predict_ei_idw(x, ksi=0.01)
        elif self.surrogate.sampling_strategy == 'idw':
            obj = surrogate.predict_idw(x, delta=2)
        elif self.surrogate.sampling_strategy == 'lcb':
            obj = surrogate.predict_lcb(x, alpha=0.5)
        elif self.surrogate.sampling_strategy == 'lcb_idw':
            obj = surrogate.predict_lcb_idw(x, alpha=0.5)
        elif self.surrogate.sampling_strategy == 'lcb_r1':
            weight_rand = np.random.uniform(0, 3, 1)
            obj = surrogate.predict_lcb(x, alpha=weight_rand)
        elif self.surrogate.sampling_strategy == 'lcb_r2':
            weight_rand = np.random.uniform(0, 1.96, 1)
            obj = surrogate.predict_lcb(x, alpha=weight_rand)
        elif self.surrogate.sampling_strategy == 'e_lcb':
            obj = surrogate.predict_lcb(x, alpha=self.surrogate.e_alpha)
        elif self.surrogate.sampling_strategy == 'mwee':
            obj = surrogate.predict_wee(x, weight=self.surrogate.weight)
        elif self.surrogate.sampling_strategy == 'wb2s':
            obj = surrogate.predict_wb2s(x, scale=self.surrogate.scale)
        elif self.surrogate.sampling_strategy == 'mu':
            obj = surrogate.predict_mu(x)
        elif self.surrogate.sampling_strategy == 'wei':
            obj = surrogate.predict_wee(x, weight=self.surrogate.wei_weight)
        elif self.surrogate.sampling_strategy == 'e_PF':
            # Check for random number below exploitation threshold
            if np.random.uniform(low=0, high=1, size=1) > self.e_tol:
                self.e_flag = True
            else:
                self.e_flag = False
            obj = surrogate.predict_mu(x)
        elif self.surrogate.sampling_strategy == 'portfolio':
            rand_num = np.random.uniform(low=0, high=1, size=1)
            if rand_num <= 0.2:
                # MWEE
                obj = surrogate.predict_wee(x, weight=self.surrogate.weight)
            elif rand_num > 0.2 and rand_num <= 0.4:
                # EI
                obj = surrogate.predict_ei(x, ksi=0.01)
            elif rand_num > 0.4 and rand_num <= 0.6:
                # IDW
                obj = surrogate.predict_idw(x, delta=2)
            elif rand_num > 0.6 and rand_num <= 0.8:
                # WEI
                obj = surrogate.predict_wei(x, weight=self.surrogate.wei_weight)
            elif rand_num > 0.8 and rand_num <= 1.0:
                # LCB
                obj = surrogate.predict_lcb(x, alpha=0.5)
        return obj

    def evaluate_offspring_selection(self, offspring_selection, var_test):
        # TODO need to set all the defaults for the various infill functions!
        ## TODO: PUT IN FUNCTION (pass back offspring_selection)
        if self.surrogate.sampling_strategy == 'ei':
            for infill_ctr in range(len(offspring_selection)):
                for idx in range(len(offspring_selection[infill_ctr])):
                    x = var_test[infill_ctr][idx]
                    for obj_cntr in range(len(self.surrogate.obj_surrogates)):
                        offspring_selection[infill_ctr][idx].obj[obj_cntr] = self.surrogate.obj_surrogates[
                            obj_cntr].predict_ei(x, ksi=0.01)
        elif self.surrogate.sampling_strategy == 'idw':
            for infill_ctr in range(len(offspring_selection)):
                for idx in range(len(offspring_selection[infill_ctr])):
                    x = var_test[infill_ctr][idx]
                    for obj_cntr in range(len(self.surrogate.obj_surrogates)):
                        offspring_selection[infill_ctr][idx].obj[obj_cntr] = self.surrogate.obj_surrogates[
                            obj_cntr].predict_idw(x, delta=2)
        elif self.surrogate.sampling_strategy == 'lcb':
            for infill_ctr in range(len(offspring_selection)):
                for idx in range(len(offspring_selection[infill_ctr])):
                    x = var_test[infill_ctr][idx]
                    for obj_cntr in range(len(self.surrogate.obj_surrogates)):
                        offspring_selection[infill_ctr][idx].obj[obj_cntr] = self.surrogate.obj_surrogates[
                            obj_cntr].predict_lcb(x, alpha=0.5)  # alpha=self.surrogate.alpha)
        elif self.surrogate.sampling_strategy == 'e_lcb':
            for infill_ctr in range(len(offspring_selection)):
                for idx in range(len(offspring_selection[infill_ctr])):
                    x = var_test[infill_ctr][idx]
                    for obj_cntr in range(len(self.surrogate.obj_surrogates)):
                        offspring_selection[infill_ctr][idx].obj[obj_cntr] = self.surrogate.obj_surrogates[
                            obj_cntr].predict_lcb(x, alpha=self.surrogate.e_alpha)
        elif self.surrogate.sampling_strategy == 'mwee':
            for infill_ctr in range(len(offspring_selection)):
                for idx in range(len(offspring_selection[infill_ctr])):
                    x = var_test[infill_ctr][idx]
                    for obj_cntr in range(len(self.surrogate.obj_surrogates)):
                        offspring_selection[infill_ctr][idx].obj[obj_cntr] = self.surrogate.obj_surrogates[
                            obj_cntr].predict_wee(x, weight=self.surrogate.weight)
        elif self.surrogate.sampling_strategy == 'wb2s':
            for infill_ctr in range(len(offspring_selection)):
                for idx in range(len(offspring_selection[infill_ctr])):
                    x = var_test[infill_ctr][idx]
                    for obj_cntr in range(len(self.surrogate.obj_surrogates)):
                        offspring_selection[infill_ctr][idx].obj[obj_cntr] = self.surrogate.obj_surrogates[
                            obj_cntr].predict_wb2s(x, scale=self.surrogate.scale[obj_cntr])
        elif self.surrogate.sampling_strategy == 'mu':
            for infill_ctr in range(len(offspring_selection)):
                for idx in range(len(offspring_selection[infill_ctr])):
                    x = var_test[infill_ctr][idx]
                    for obj_cntr in range(len(self.surrogate.obj_surrogates)):
                        offspring_selection[infill_ctr][idx].obj[obj_cntr] = self.surrogate.obj_surrogates[
                            obj_cntr].predict_mu(x)
        elif self.surrogate.sampling_strategy == 'wei':
            for infill_ctr in range(len(offspring_selection)):
                for idx in range(len(offspring_selection[infill_ctr])):
                    x = var_test[infill_ctr][idx]
                    for obj_cntr in range(len(self.surrogate.obj_surrogates)):
                        offspring_selection[infill_ctr][idx].obj[obj_cntr] = self.surrogate.obj_surrogates[
                            obj_cntr].predict_wee(x, weight=self.surrogate.wei_weight)
        elif self.surrogate.sampling_strategy == 'e_PF':
            # Check for random number below exploitation threshold
            if np.random.uniform(low=0, high=1, size=1) > self.e_tol:
                self.e_flag = True
            else:
                self.e_flag = False
            for infill_ctr in range(len(offspring_selection)):
                for idx in range(len(offspring_selection[infill_ctr])):
                    x = var_test[infill_ctr][idx]
                    for obj_cntr in range(len(self.surrogate.obj_surrogates)):
                        # Evaluate surrogate mean predictions
                        offspring_selection[infill_ctr][idx].obj[obj_cntr] = self.surrogate.obj_surrogates[
                            obj_cntr].predict_mu(x)

        elif self.surrogate.sampling_strategy == 'portfolio':
            rand_num = np.random.uniform(low=0, high=1, size=1)
            for infill_ctr in range(len(offspring_selection)):
                for idx in range(len(offspring_selection[infill_ctr])):
                    x = var_test[infill_ctr][idx]
                    for obj_cntr in range(len(self.surrogate.obj_surrogates)):
                        if rand_num <= 0.2:
                            # MWEE
                            offspring_selection[infill_ctr][idx].obj[obj_cntr] = self.surrogate.obj_surrogates[
                                obj_cntr].predict_wee(x, weight=self.surrogate.weight)
                        elif rand_num > 0.2 and rand_num <= 0.4:
                            # EI
                            offspring_selection[infill_ctr][idx].obj[obj_cntr] = self.surrogate.obj_surrogates[
                                obj_cntr].predict_ei(x, ksi=0.01)
                        elif rand_num > 0.4 and rand_num <= 0.6:
                            # IDW
                            offspring_selection[infill_ctr][idx].obj[obj_cntr] = self.surrogate.obj_surrogates[
                                obj_cntr].predict_idw(x, delta=2)
                        elif rand_num > 0.6 and rand_num <= 0.8:
                            # WEI
                            offspring_selection[infill_ctr][idx].obj[obj_cntr] = self.surrogate.obj_surrogates[
                                obj_cntr].predict_wei(x, weight=self.surrogate.wei_weight)
                        elif rand_num > 0.8 and rand_num <= 1.0:
                            # LCB
                            offspring_selection[infill_ctr][idx].obj[obj_cntr] = self.surrogate.obj_surrogates[
                                obj_cntr].predict_lcb(x, alpha=0.5)
        elif self.surrogate.sampling_strategy == 'portfolio3':
            for infill_ctr in range(len(offspring_selection)):
                for idx in range(len(offspring_selection[infill_ctr])):
                    x = var_test[infill_ctr][idx]
                    for obj_cntr in range(len(self.surrogate.obj_surrogates)):
                        if infill_ctr == 0:
                            # MWEE
                            offspring_selection[infill_ctr][idx].obj[obj_cntr] = self.surrogate.obj_surrogates[
                                obj_cntr].predict_wee(x, weight=self.surrogate.weight)
                        elif infill_ctr == 1:
                            # WEI
                            offspring_selection[infill_ctr][idx].obj[obj_cntr] = self.surrogate.obj_surrogates[
                                obj_cntr].predict_wei(x, weight=self.surrogate.wei_weight)
                        elif infill_ctr == 2:
                            # LCB
                            offspring_selection[infill_ctr][idx].obj[obj_cntr] = self.surrogate.obj_surrogates[
                                obj_cntr].predict_lcb(x, alpha=0.5)
        else:
            warnings.warn('requested infill strategy not implemented yet so objectives will be used')
        ## WB2S: if self.obj.scale[i] = above^^

        return offspring_selection

    def evaluate_offspring_selection_constraints(self, offspring_selection, var_test):
        if self.surrogate.sampling_strategy == 'portfolio':
            rand_num = np.random.uniform(low=0, high=1, size=1)
            for infill_ctr in range(len(offspring_selection)):
                for idx in range(len(offspring_selection[infill_ctr])):
                    x = var_test[infill_ctr][idx]
                    for obj_cntr in range(len(self.surrogate.obj_surrogates)):
                        if rand_num <= 0.2:
                            # MWEE
                            offspring_selection[infill_ctr][idx].obj[obj_cntr] = self.surrogate.obj_surrogates[
                                obj_cntr].predict_wee(x, weight=self.surrogate.weight)
                        elif rand_num > 0.2 and rand_num <= 0.4:
                            # EI
                            offspring_selection[infill_ctr][idx].obj[obj_cntr] = self.surrogate.obj_surrogates[
                                obj_cntr].predict_ei(x, ksi=0.01)
                        elif rand_num > 0.4 and rand_num <= 0.6:
                            # IDW
                            offspring_selection[infill_ctr][idx].obj[obj_cntr] = self.surrogate.obj_surrogates[
                                obj_cntr].predict_idw(x, delta=2)
                        elif rand_num > 0.6 and rand_num <= 0.8:
                            # WEI
                            offspring_selection[infill_ctr][idx].obj[obj_cntr] = self.surrogate.obj_surrogates[
                                obj_cntr].predict_wei(x, weight=self.surrogate.wei_weight)
                        elif rand_num > 0.8 and rand_num <= 1.0:
                            # LCB
                            offspring_selection[infill_ctr][idx].obj[obj_cntr] = self.surrogate.obj_surrogates[
                                obj_cntr].predict_lcb(x, alpha=0.5)

                        # Multiply Infill by POF
                        if self.surrogate.constraint_strategy == 'pof':
                            for cons_cntr in range(len(self.surrogate.cons_surrogates)):
                                offspring_selection[infill_ctr][idx].obj[obj_cntr] = self.surrogate.cons_surrogates[
                                    cons_cntr].predict_pof(x, offspring_selection[infill_ctr][idx].obj[obj_cntr], k=1.0)
                                offspring_selection[infill_ctr][idx].cons[cons_cntr] = None

                        if self.surrogate.constraint_strategy == 'pof_isc':
                            for cons_cntr in range(len(self.surrogate.cons_surrogates)):
                                offspring_selection[infill_ctr][idx].obj[obj_cntr] = self.surrogate.cons_surrogates[
                                    cons_cntr].predict_pof_isc(x, offspring_selection[infill_ctr][idx].obj[obj_cntr],
                                                               k=1.0)
                        if self.surrogate.constraint_strategy == 'pof_if':
                            for cons_cntr in range(len(self.surrogate.cons_surrogates)):
                                offspring_selection[infill_ctr][idx].obj[obj_cntr] = self.surrogate.cons_surrogates[
                                    cons_cntr].predict_pof_if(x, offspring_selection[infill_ctr][idx].obj[obj_cntr],
                                                              self.opt, k=1.0)
                        if self.surrogate.constraint_strategy == 'e_pof':
                            rand_num = np.random.uniform(low=0, high=1, size=1)
                            for cons_cntr in range(len(self.surrogate.cons_surrogates)):
                                if rand_num <= (1 / 3):
                                    # POF
                                    offspring_selection[infill_ctr][idx].obj[obj_cntr] = self.surrogate.cons_surrogates[
                                        cons_cntr].predict_pof(x, offspring_selection[infill_ctr][idx].obj[obj_cntr],
                                                               k=1.0)
                                elif rand_num > (1 / 3) and rand_num <= (2 / 3):
                                    # POF*ISC
                                    offspring_selection[infill_ctr][idx].obj[obj_cntr] = self.surrogate.cons_surrogates[
                                        cons_cntr].predict_pof_isc(x,
                                                                   offspring_selection[infill_ctr][idx].obj[obj_cntr],
                                                                   k=1.0)
                                elif rand_num > (2 / 3) and rand_num <= 1:
                                    # POF*IF
                                    offspring_selection[infill_ctr][idx].obj[obj_cntr] = self.surrogate.cons_surrogates[
                                        cons_cntr].predict_pof_if(x, offspring_selection[infill_ctr][idx].obj[obj_cntr],
                                                                  self.opt, k=1.0)
                        # When no infeasible points are found
                        # if self.infeasible_flag:
                        #     offspring_selection[infill_ctr][idx].obj[obj_cntr] = 1

                    rand_num_1 = np.random.uniform(low=0, high=1, size=1)
                    for cons_cntr in range(len(self.surrogate.cons_surrogates)):
                        # consLCB
                        if self.surrogate.constraint_strategy == 'conslcb':
                            offspring_selection[infill_ctr][idx].cons[cons_cntr] = self.surrogate.cons_surrogates[
                                cons_cntr].predict_conslcb(x, alpha=1.96)
                        # Expected Violation (EV)
                        elif self.surrogate.constraint_strategy == 'ev':
                            offspring_selection[infill_ctr][idx].cons[cons_cntr] = self.surrogate.cons_surrogates[
                                cons_cntr].predict_ev(x, Pev=0.001, weight=0.5)
                        # Weighted Expected Violation (WEV)
                        elif self.surrogate.constraint_strategy == 'wev':
                            offspring_selection[infill_ctr][idx].cons[cons_cntr] = self.surrogate.cons_surrogates[
                                cons_cntr].predict_ev(x, Pev=0.001, weight=self.surrogate.wei_weight)
                        # cons MWEE
                        elif self.surrogate.constraint_strategy == 'consmwee':
                            offspring_selection[infill_ctr][idx].cons[cons_cntr] = self.surrogate.cons_surrogates[
                                cons_cntr].predict_conswee(x, weight=self.surrogate.weight)
                        # cons IDW
                        elif self.surrogate.constraint_strategy == 'considw':
                            offspring_selection[infill_ctr][idx].cons[cons_cntr] = self.surrogate.cons_surrogates[
                                cons_cntr].predict_idw(x, delta=2)
                        elif self.surrogate.constraint_strategy == 'consportfolio' or self.surrogate.constraint_strategy == 'consportfolio_ks':
                            if rand_num_1 <= 0.2:
                                # consMWEE
                                offspring_selection[infill_ctr][idx].cons[cons_cntr] = self.surrogate.cons_surrogates[
                                    cons_cntr].predict_conswee(x, weight=self.surrogate.weight)
                            elif rand_num_1 > 0.2 and rand_num <= 0.4:
                                # consEI (EV)
                                offspring_selection[infill_ctr][idx].cons[cons_cntr] = self.surrogate.cons_surrogates[
                                    cons_cntr].predict_ev(x, Pev=0.001, weight=0.5)
                            elif rand_num_1 > 0.4 and rand_num <= 0.6:
                                # consIDW
                                offspring_selection[infill_ctr][idx].cons[cons_cntr] = self.surrogate.cons_surrogates[
                                    cons_cntr].predict_idw(x, delta=2)
                            elif rand_num_1 > 0.6 and rand_num <= 0.8:
                                # consWEI (WEV)
                                offspring_selection[infill_ctr][idx].cons[cons_cntr] = self.surrogate.cons_surrogates[
                                    cons_cntr].predict_ev(x, Pev=0.001, weight=self.surrogate.wei_weight)
                            elif rand_num_1 > 0.8 and rand_num <= 1.0:
                                # # consLCB
                                # offspring_selection[infill_ctr][idx].cons[cons_cntr] = self.surrogate.cons_surrogates[
                                #     cons_cntr].predict_conslcb(x, alpha=1.96)
                                # Individual PoF
                                offspring_selection[infill_ctr][idx].cons[cons_cntr] = self.surrogate.cons_surrogates[
                                    cons_cntr].predict_pof(x, -1, k=1.1)
                        elif self.surrogate.constraint_strategy == 'sumpof':
                            # Individual PoF
                            offspring_selection[infill_ctr][idx].cons[cons_cntr] = self.surrogate.cons_surrogates[
                                cons_cntr].predict_pof(x, -1, k=1.1)
        else:
            warnings.warn('requested infill strategy not implemented yet so objectives will be used')
        return offspring_selection

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

    def update_pareto_archive(self):
        # Merge the pareto archive with the current population
        updated_pareto = Population.merge(self.population, self.pareto_archive)
        if len(updated_pareto) > self.n_population:
            index_list = list(range(len(updated_pareto)))
            selected_indices = random.sample(index_list, self.n_population)
            updated_pareto = updated_pareto[selected_indices]
        self.pareto_archive = updated_pareto

    def update_pareto_archive_exploitation(self, population, archive, population_size):
        # Merge the pareto archive with the current population
        updated_pareto = Population.merge(population, archive)
        if len(updated_pareto) > population_size:
            index_list = list(range(len(updated_pareto)))
            selected_indices = random.sample(index_list, population_size)
            updated_pareto = updated_pareto[selected_indices]
        return updated_pareto

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

