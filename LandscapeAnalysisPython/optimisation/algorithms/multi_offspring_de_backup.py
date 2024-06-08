import numpy as np
from scipy.spatial import distance
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


class MODE(EvolutionaryAlgorithm):
    """
    multi-offspring DE - should only be used with surrogate as this creates a lot of offspring
    """

    def __init__(self,
                 n_population=100,
                 sampling=LatinHypercubeSampling(),
                 selection=RandomSelection(),
                 # survival=RankAndCrowdingSurvival(),
                 # survival=MultipleConstraintsRankingSurvival(),
                 survival=None,
                 crossover=None,
                 mutation=None,
                 var_selection='best',
                 var_n=1,
                 var_mutation='bin',
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
        self.f = [0.8, 1.0, 1.0, 0.8, 0.8, 0.9, 0.4, 0.4]
        self.cr = [0.2, 0.1, 0.9, 0.8, 0.6, 0.2, 0.2, 0.9]
        self.memory_length = 5
        self.memory_index = None

        # survival
        if survival is not None:
            self.survival = survival
        else:
            survival = RankAndCrowdingSurvival()
            self.survival = RankAndCrowdingSurvival()

        # flag to use spiral movement operator of WOA
        self.use_wo = True
        # flag to use improved spiral movement operator from TLBO
        self.use_spiral = True

        # flag to select F and CR from pool for each offspring strategy
        self.f_and_cr_per_strategy = False
        self.diversity_by_distance = False
        self.diversity_from_initialDOE = False
        self.diversity_by_clustering = False
        self.random_cutoff = 0.0
        self.number_of_groups = 5
        self.use_improvement = False

        # DE crossover
        # if crossover is None:
        #     crossover = DifferentialEvolutionCrossover(weight=f, dither=dither, jitter=jitter)
        #
        # # DE mutation
        # if self.var_mutation == 'exp':
        #     mutation = ExponentialCrossover(cr)
        # elif self.var_mutation == 'bin':
        #     mutation = BiasedCrossover(cr)

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

        ## VARIABLE FOR REAL F EVAL POPULATION
        self.n_pop_old = self.n_population

        ## HARDCODE NUMBER OF INFILL CRITERIA
        self.duplicate_count = 0
        self.e_flag = True
        self.e_tol = 1.5 / self.problem.n_var
        self.infeasible_flag = False

        ## Setup dummy population to initially optimise constraint infill criteria
        self.dummy_problem = copy.copy(self.problem)
        self.dummy_problem.n_obj = self.problem.n_con
        self.dummy_problem.n_con = 0
        # self.dummy_pop = Population(self.dummy_problem, 0)

        # Initialise from the evolutionary algorithm class
        super()._initialise()

        # update pareto archive as non-dominated solutions from pareto archive and population
        self.update_pareto_archive()

        # set memory index to 0
        self.memory_index = 0
        # set the f and cr memory
        self.training_data = None

        # self.f_memory = self.f * np.ones(self.memory_length)
        # self.cr_memory = self.cr * np.ones(self.memory_length)

    def _next(self):

        # Conduct surrogate model refinement (adaptive sampling)
        if self.surrogate is not None:
            if self.training_data is not None:
                self.surrogate.run(self.problem, self.training_data)
            #     TODO need to add the training data to the surrogate population

            else:
                self.surrogate.run(self.problem)
        #     should add training data in the call to surrogate.run !

        ## RESET TRAINING DATA EACH ITERATION
        self.training_data = None

        ## Check for any feasible points
        if self.surrogate.cons_surrogates != []:
            constraint_sum = self.surrogate.population.extract_cons_sum()
            if (constraint_sum <= 0).any():
                self.infeasible_flag = False
            else:
                self.infeasible_flag = True
        # print(f"Infeasible: {self.infeasible_flag}")

        # create offspring
        offspring = self.create_multiple_offspring()

        # evaluate offspring
        if self.surrogate is not None:
            offspring = self.evaluator.do(self.surrogate.obj_func, self.problem, offspring)
        else:
            offspring = self.evaluator.do(self.problem.obj_func, self.problem, offspring)

        # here is where the infill criterion needs to get called
        # copy offspring first and make it a population
        offspring_selection = Population(self.problem, 0)
        offspring_selection = [Population.merge(offspring_selection, offspring) for _ in range(self.n_infill_criteria)]

        # extract variables from the population
        var_test = [offspring_selection[idx].extract_var() for idx in range(len(offspring_selection))]

        # Call the infill function evaluator for the offspring_selection
        if self.surrogate.cons_surrogates == []:
            offspring_selection = self.evaluate_offspring_selection(offspring_selection, var_test)
        else:
            offspring_selection = self.evaluate_offspring_selection_constraints(offspring_selection, var_test)

        # selection to generate the next off-spring
        offspring = self.survival.do(self.problem, offspring, self.n_population,
                                     gen=self.n_gen, max_gen=self.max_gen)

        trial_array = self.population.extract_var()

        # selection for the infill points
        if self.surrogate is not None:
            for infill_ctr in range(len(offspring_selection)):
                # remove duplicates
                is_duplicate = self.check_duplications(offspring_selection[infill_ctr], self.surrogate.population)
                offspring_selection[infill_ctr] = offspring_selection[infill_ctr][np.invert(is_duplicate)[0]]

                ## CHANGES FOR e-PF infill algorithm
                if self.e_flag:
                    offspring_selection[infill_ctr] = self.survival.do(self.problem, offspring_selection[infill_ctr],
                                                                       self.surrogate.n_infill,
                                                                       gen=self.n_gen, max_gen=self.max_gen)
                else:
                    ## TODO: Implement random selection from non-dominated front of the surrogate predictions
                    offspring_selection[infill_ctr] = self.survival.do(self.problem, offspring_selection[infill_ctr],
                                                                       self.n_population,
                                                                       gen=self.n_gen, max_gen=self.max_gen)
                    pf_mask = np.zeros((len(offspring_selection[infill_ctr])), dtype=bool)
                    rand_pf = np.random.randint(low=0, high=len(offspring_selection[infill_ctr]))
                    pf_mask[rand_pf] = True
                    offspring_selection[infill_ctr] = offspring_selection[infill_ctr][pf_mask]

                # merge population and find which offspring survived
                survived = self.create_merged_population(offspring_selection[infill_ctr], trial_array)
                # offspring_array = offspring_selection[survived].extract_var()
                # check if they have not yet been tested - easier to use remove_duplicates from population?
                # use dist_calc method from duplicate.py

                # remove duplicates
                # is_duplicate = self.check_duplications(offspring_selection[infill_ctr], self.surrogate.population)
                # offspring_selection[infill_ctr] = offspring_selection[infill_ctr][np.invert(is_duplicate)[0]]

                # evaluate the selected infill points with the real function
                offspring_selection[infill_ctr] = self.evaluator.do(self.problem.obj_func, self.problem,
                                                                    offspring_selection[infill_ctr])

                ## COUNT MISSING INFILL POINTS IN SELECTION
                if offspring_selection[infill_ctr].extract_var() == []:
                    self.duplicate_count += 1
                else:
                    if len(self.surrogate.cons_surrogates) == 0:
                        if self.training_data is not None:
                            self.training_data = (
                                np.vstack((self.training_data[0], offspring_selection[infill_ctr].extract_var())),
                                np.vstack((self.training_data[1], offspring_selection[infill_ctr].extract_obj()))
                            )
                        else:
                            self.training_data = (offspring_selection[infill_ctr].extract_var(),
                                                  np.atleast_2d(offspring_selection[infill_ctr].extract_obj()))

                    # elif len(self.surrogate.cons_surrogates) == 1:
                    # TODO - add the extraction of the cons_sum or cons_viol surrogate here
                    #    pass
                    else:
                        if self.training_data is not None:
                            self.training_data = (
                                np.vstack((self.training_data[0], offspring_selection[infill_ctr].extract_var())),
                                np.vstack(
                                    (self.training_data[1], np.hstack((offspring_selection[infill_ctr].extract_obj(),
                                                                       offspring_selection[
                                                                           infill_ctr].extract_cons()))))
                            )
                        else:
                            self.training_data = (offspring_selection[infill_ctr].extract_var(),
                                                  np.hstack((offspring_selection[infill_ctr].extract_obj(),
                                                             offspring_selection[infill_ctr].extract_cons())))

                    # Add the training data to the population
                    self.surrogate.population = Population.merge(self.surrogate.population,
                                                                 offspring_selection[infill_ctr])

        self.surrogate.population = copy.deepcopy(self.surrogate.population)

        ## CALL TERMINATION IF NO INFILL POINTS WERE MANAGED ACROSS THE NUMBER OF INFILL CRITERIA
        if self.duplicate_count >= self.n_infill_criteria:
            self.finished = True

        survived = self.create_merged_population(offspring, trial_array)

        # # updated pareto archive and adaptive archive
        self.update_pareto_archive()
        self.update_adaptive_archive(offspring[survived])
        #
        # # update the memory for f and cr
        # self.update_f_and_cr_memory(cr_array, f_array, survived)

        old_opt = self.opt

        ## Update optimum
        if self.surrogate is not None:
            # if self.problem.n_obj == 1:
            opt = RankAndCrowdingSurvival().do(self.problem, self.surrogate.population, 1, None, None)
        else:
            opt = RankAndCrowdingSurvival().do(self.problem, self.population, 1, None, None)
        self.opt = opt[0]

        # TODO - needs adjusting when switching to multi-objective

        if self.use_improvement:
            improvement = self.opt.obj[0] - old_opt.obj[0]
            if improvement != 0:
                pass
            elif np.random.random() < 1 / 3:
                self.diversity_by_distance = not self.diversity_by_distance
            elif np.random.random() < 2 / 3:
                self.diversity_by_clustering = not self.diversity_by_clustering
            else:
                self.diversity_from_initialDOE = not self.diversity_from_initialDOE

        ## Reset multiple infill criteria count
        self.duplicate_count = 0

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

    def create_multiple_offspring(self):
        ## MAKE SURE ONLY REAL F EVALS ARE USED
        # extract variables and create union of external archive and population
        # TODO: make a new function for this and instead of relying on survival use a two-ranking style approach
        # equal likelihood to be ranked as currently (so whatever survival method is used) and ranked based
        # on objective and distance from optimum. Should add more diversity back in when it has
        # sampled too close to optimum too ofter
        if self.diversity_by_distance:
            if len(self.surrogate.population) < self.n_pop_old:
                var_array = self.population.extract_var()
                self.n_population = len(var_array)
            # Check if surrogate pop is greater than desired population
            elif len(self.surrogate.population) >= self.n_pop_old:
                if np.random.random() < self.random_cutoff:
                    best_pop = self.survival.do(self.problem, self.surrogate.population, self.n_pop_old,
                                                gen=self.n_gen, max_gen=self.max_gen)
                else:
                    # generate random indices
                    # indices1 = random.sample(range(np.int64(np.floor(len(self.surrogate.population)/2))),
                    #                          np.int64(np.floor(self.n_pop_old/2)))
                    # select best member from population
                    best_individual = self.survival.do(self.problem, self.surrogate.population, 1,
                                                       gen=self.n_gen, max_gen=self.max_gen)
                    # calculate distances
                    distance = self.calc_dist(self.surrogate.population, best_individual)
                    # sort based on distance
                    sorted_population = np.argsort(distance, axis=0)

                    obj_array = self.surrogate.population.extract_obj()

                    if self.problem.n_con == 0:
                        fronts_obj = NonDominatedSorting().do(obj_array,
                                                              n_stop_if_ranked=len(self.surrogate.population))
                    else:
                        cons_array = self.surrogate.population.extract_cons()
                        fronts_obj = NonDominatedSorting().do(obj_array, cons_val=cons_array,
                                                              n_stop_if_ranked=len(self.surrogate.population))
                    fronts_distance = NonDominatedSorting().do(distance,
                                                               n_stop_if_ranked=len(self.surrogate.population))
                    rank_1 = rank_fronts(fronts_obj, obj_array, self.surrogate.population, np.zeros(0, dtype=np.int))
                    rank_2 = rank_fronts(fronts_distance, distance, self.surrogate.population,
                                         np.zeros(0, dtype=np.int))
                    rank_composite = rank_1 + rank_2
                    survivors = np.argsort(rank_composite)
                    if len(survivors) > self.n_pop_old:
                        survivors = survivors[:self.n_pop_old]

                    best_pop = self.surrogate.population[survivors]
                var_array = best_pop.extract_var()
                self.n_population = len(var_array)

            else:
                var_array = self.population.extract_var()
        elif self.diversity_from_initialDOE:
            if len(self.surrogate.population) < self.n_pop_old:
                var_array = self.population.extract_var()
                self.n_population = len(var_array)
            # Check if surrogate pop is greater than desired population
            elif len(self.surrogate.population) >= 4 * self.n_pop_old:
                if np.random.random() < self.random_cutoff:
                    best_pop = self.survival.do(self.problem, self.surrogate.population, self.n_pop_old,
                                                gen=self.n_gen, max_gen=self.max_gen)
                else:
                    # generate random indices
                    # indices1 = random.sample(range(np.int64(np.floor(len(self.surrogate.population)/2))),
                    #                          np.int64(np.floor(self.n_pop_old/2)))
                    # TODO run with the version below to force more early sampled points as parents (extra diversity)
                    indices1 = random.sample(range(np.int64(np.floor(2 * self.n_pop_old))),
                                             np.int64(np.floor(self.n_pop_old / 2)))
                    if self.n_pop_old % 2 == 0:
                        indices2 = random.sample(range(np.int64(np.floor(len(self.surrogate.population) / 2))),
                                                 np.int64(np.floor(self.n_pop_old / 2)))
                    else:
                        indices2 = random.sample(range(np.int64(np.floor(len(self.surrogate.population) / 2))),
                                                 np.int64(np.floor(self.n_pop_old / 2) + 1))

                    indices2 = [i + np.int64(np.floor(len(self.surrogate.population) / 2)) for i in indices2]
                    indices = indices1 + indices2
                    best_pop = self.surrogate.population[indices]
                var_array = best_pop.extract_var()
                self.n_population = len(var_array)
            elif len(self.surrogate.population) >= 2 * self.n_pop_old:
                if np.random.random() < self.random_cutoff:
                    best_pop = self.survival.do(self.problem, self.surrogate.population, self.n_pop_old,
                                                gen=self.n_gen, max_gen=self.max_gen)
                else:
                    indices1 = random.sample(range(np.int64(np.floor(self.n_pop_old))),
                                             np.int64(np.floor(self.n_pop_old / 2)))
                    if self.n_pop_old % 2 == 0:
                        indices2 = random.sample(range(np.int64(np.floor(len(self.surrogate.population) / 2))),
                                                 np.int64(np.floor(self.n_pop_old / 2)))
                    else:
                        indices2 = random.sample(range(np.int64(np.floor(len(self.surrogate.population) / 2))),
                                                 np.int64(np.floor(self.n_pop_old / 2) + 1))

                    indices2 = [i + np.int64(np.floor(len(self.surrogate.population) / 2)) for i in indices2]
                    indices = indices1 + indices2
                    best_pop = self.surrogate.population[indices]
                var_array = best_pop.extract_var()
                self.n_population = len(var_array)
            elif len(self.surrogate.population) >= self.n_pop_old:
                if np.random.random() < self.random_cutoff:
                    best_pop = self.survival.do(self.problem, self.surrogate.population, self.n_pop_old,
                                                gen=self.n_gen, max_gen=self.max_gen)
                else:
                    # generate random indices
                    indices = random.sample(range(len(self.surrogate.population)), self.n_pop_old)
                    best_pop = self.surrogate.population[indices]
                var_array = best_pop.extract_var()
                self.n_population = len(var_array)
            else:
                var_array = self.population.extract_var()
        elif self.diversity_by_clustering:
            # TODO group solutions based on ranking and then pick top x solutions from each group
            # use the rank by distance principle to get two ranks but use obj rank to split in groups
            # and then use best x solutions from each group or ones with larger distance?
            # maybe both with equal probability so 1/3rd each
            if len(self.surrogate.population) < self.n_pop_old:
                var_array = self.population.extract_var()
                self.n_population = len(var_array)
            # Check if surrogate pop is greater than desired population
            elif len(self.surrogate.population) >= self.n_pop_old:
                if np.random.random() < self.random_cutoff:
                    best_pop = self.survival.do(self.problem, self.surrogate.population, self.n_pop_old,
                                                gen=self.n_gen, max_gen=self.max_gen)
                else:
                    # select best member from population
                    best_individual = self.survival.do(self.problem, self.surrogate.population, 1,
                                                       gen=self.n_gen, max_gen=self.max_gen)
                    # calculate distances
                    distance = self.calc_dist(self.surrogate.population, best_individual)
                    # sort based on distance
                    sorted_population = np.argsort(distance, axis=0)

                    obj_array = self.surrogate.population.extract_obj()

                    if self.problem.n_con == 0:
                        fronts_obj = NonDominatedSorting().do(obj_array,
                                                              n_stop_if_ranked=len(self.surrogate.population))
                    else:
                        cons_array = self.surrogate.population.extract_cons()
                        fronts_obj = NonDominatedSorting().do(obj_array, cons_val=cons_array,
                                                              n_stop_if_ranked=len(self.surrogate.population))
                    fronts_distance = NonDominatedSorting().do(distance,
                                                               n_stop_if_ranked=len(self.surrogate.population))
                    rank_1 = rank_fronts(fronts_obj, obj_array, self.surrogate.population, np.zeros(0, dtype=np.int))
                    rank_2 = rank_fronts(fronts_distance, distance, self.surrogate.population,
                                         np.zeros(0, dtype=np.int))
                    # split in groups
                    # find the indices to retain
                    # need to retain x indices from each group
                    number_to_retain = np.int(np.floor(self.n_pop_old / self.number_of_groups))
                    group_size = np.int(np.floor(len(rank_1) / self.number_of_groups))

                    indices_to_retain = []

                    for cntr in range(self.number_of_groups):
                        for cntr2 in range(number_to_retain):
                            index = cntr * group_size + cntr2
                            indices_to_retain.append(index)

                    # now find the ones to retain
                    rank_copy = copy.deepcopy(rank_1)
                    rank_copy2 = copy.deepcopy(rank_2)
                    for cntr in range(len(rank_1)):
                        if np.int64(rank_1[cntr]) in indices_to_retain:
                            rank_copy[cntr] = rank_1[cntr]
                            rank_copy2[cntr] = rank_2[cntr]
                        else:
                            rank_copy[cntr] = np.max(rank_1) + rank_1[cntr]
                            rank_copy2[cntr] = np.max(rank_2) + rank_2[cntr]
                    if np.random.random() < self.random_cutoff:
                        survivors = np.argsort(rank_copy)
                    else:
                        survivors = np.argsort(rank_copy2)
                    if len(survivors) > self.n_pop_old:
                        survivors = survivors[:self.n_pop_old]

                    best_pop = self.surrogate.population[survivors]
                var_array = best_pop.extract_var()
                self.n_population = len(var_array)


        else:
            best_pop = self.survival.do(self.problem, self.surrogate.population, self.n_pop_old,
                                        gen=self.n_gen, max_gen=self.max_gen)
            var_array = best_pop.extract_var()
            self.n_population = len(var_array)

        personal_best_array = self.pareto_archive.extract_var()
        # create the various offspring
        if self.f_and_cr_per_strategy:
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
        else:
            f, cr = self.assign_f_and_cr_from_pool()
            offspring1 = self.rand_1_bin(var_array, f, cr)
            offspring2 = self.current_to_randbest_1_bin(var_array, personal_best_array, f, cr)
            offspring3 = self.best_1_bin(var_array, personal_best_array, f, cr)
            f2, _ = self.assign_f_and_cr_from_pool()
            offspring4 = self.best_2_bin(var_array, personal_best_array, f, f2, cr)
            offspring5 = self.current_to_rand_1_bin(var_array, f, cr)
            offspring6 = self.modified_rand_to_best_1_bin(var_array, personal_best_array, f, cr)
            offspring7 = self.current_to_best_1_bin(var_array, personal_best_array, f, cr)

        # evaluate everything on the surrogate
        # offspring1 = self.evaluator.do(self.surrogate.obj_func, self.problem, offspring1)
        # offspring2 = self.evaluator.do(self.surrogate.obj_func, self.problem, offspring2)
        # offspring3 = self.evaluator.do(self.surrogate.obj_func, self.problem, offspring3)
        # offspring4 = self.evaluator.do(self.surrogate.obj_func, self.problem, offspring4)
        # offspring5 = self.evaluator.do(self.surrogate.obj_func, self.problem, offspring5)
        # offspring6 = self.evaluator.do(self.surrogate.obj_func, self.problem, offspring6)
        # offspring7 = self.evaluator.do(self.surrogate.obj_func, self.problem, offspring7)

        # merge everything into a very large population
        merged_population = Population.merge(self.population, offspring1)
        merged_population = Population.merge(merged_population, offspring2)
        merged_population = Population.merge(merged_population, offspring3)
        merged_population = Population.merge(merged_population, offspring4)
        merged_population = Population.merge(merged_population, offspring5)
        merged_population = Population.merge(merged_population, offspring6)
        merged_population = Population.merge(merged_population, offspring7)

        return merged_population

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

    def rand_1_bin(self, var_array, f, cr):
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

        offspring = Population(self.problem, self.n_population)
        offspring.assign_var(self.problem, trial_array)
        offspring = BasicBoundsRepair().do(self.problem, offspring)

        return offspring

    def best_1_bin(self, var_array, personal_best_array, f, cr):
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

        offspring = Population(self.problem, self.n_population)
        offspring.assign_var(self.problem, trial_array)
        offspring = BasicBoundsRepair().do(self.problem, offspring)

        return offspring

    def best_2_bin(self, var_array, personal_best_array, f1, f2, cr):
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

        offspring = Population(self.problem, self.n_population)
        offspring.assign_var(self.problem, trial_array)
        offspring = BasicBoundsRepair().do(self.problem, offspring)

        return offspring

    def current_to_randbest_1_bin(self, var_array, personal_best_array, f, cr):
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

        offspring = Population(self.problem, self.n_population)
        offspring.assign_var(self.problem, trial_array)
        offspring = BasicBoundsRepair().do(self.problem, offspring)

        return offspring

    def current_to_best_1_bin(self, var_array, personal_best_array, f, cr):
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

        offspring = Population(self.problem, self.n_population)
        offspring.assign_var(self.problem, trial_array)
        offspring = BasicBoundsRepair().do(self.problem, offspring)

        return offspring

    def modified_rand_to_best_1_bin(self, var_array, personal_best_array, f, cr):
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

        offspring = Population(self.problem, self.n_population)
        offspring.assign_var(self.problem, trial_array)
        offspring = BasicBoundsRepair().do(self.problem, offspring)

        return offspring

    def current_to_rand_1_bin(self, var_array, f, cr):
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

        offspring = Population(self.problem, self.n_population)
        offspring.assign_var(self.problem, trial_array)
        offspring = BasicBoundsRepair().do(self.problem, offspring)

        return offspring

    def assign_f_and_cr_from_pool(self):
        ind = random.randint(0, len(self.f) - 1)
        f = self.f[ind]
        cr = self.cr[ind]
        return f, cr

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

