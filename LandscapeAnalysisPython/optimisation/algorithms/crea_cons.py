import copy
import random
import time
from termcolor import colored
import sys

import numpy as np
from scipy.spatial import distance
from scipy.stats import norm
from scipy.stats import rankdata

from optimisation.setup import Setup
from optimisation.model.population import Population
from optimisation.algorithms.sa_evolutionary_algorithm import SAEvolutionaryAlgorithm

from optimisation.operators.mutation.de_mutation import DifferentialEvolutionMutation
from optimisation.operators.sampling.lhs_loader import LHSLoader
from optimisation.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from optimisation.operators.selection.random_selection import RandomSelection
from optimisation.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from optimisation.operators.mutation.polynomial_mutation import PolynomialMutation
from optimisation.operators.survival.theta_survival import ThetaSurvival
from optimisation.operators.survival.generalised_mcr_ref_vector_survival import gMCRSurvival
from optimisation.operators.survival.rank_and_crowding_survival import RankAndCrowdingSurvival
from optimisation.operators.survival.two_ranking_survival import TwoRankingSurvival
from optimisation.operators.survival.population_based_epsilon_survival import PopulationBasedEpsilonSurvival

from optimisation.model.duplicate import DefaultDuplicateElimination
from optimisation.model.repair import BasicBoundsRepair, BounceBackBoundsRepair
from optimisation.model.evaluator import Evaluator
from optimisation.metrics.indicator import Indicator
from optimisation.output.generation_extractor import GenerationExtractor
from optimisation.model.normalisation import Normalisation

from optimisation.util.dominator import calculate_domination_matrix

from optimisation.surrogate.models.rbf import RadialBasisFunctions
from optimisation.surrogate.rbf_tuner import RBFTuner

# TODO: Move static methods from theta survival class to optimsation/utils/
from optimisation.util.hyperplane_normalisation import HyperplaneNormalisation
from optimisation.util.non_dominated_sorting import NonDominatedSorting

import matplotlib.pyplot as plt
plt.style.use('seaborn-talk')
np.set_printoptions(suppress=True)
# matplotlib.use('TkAgg')
line_colors = ['green', 'blue', 'red', 'orange', 'cyan', 'lawngreen', 'm', 'orangered', 'sienna', 'gold', 'violet',
               'indigo', 'cornflowerblue']


class CREA_CONS(SAEvolutionaryAlgorithm):
    """
    Classification and Regression Evolutionary Algorithm - Constrained
    """
    def __init__(self,
                 ref_dirs=None,
                 n_population=109,
                 surrogate=None,
                 sampling=LatinHypercubeSampling(criterion='maximin', iterations=10000),  # LHSLoader()
                 selection=RandomSelection(),
                 crossover=SimulatedBinaryCrossover(eta=20, prob=1.0),
                 mutation=PolynomialMutation(eta=20, prob=None),
                 eliminate_duplicates=DefaultDuplicateElimination(epsilon=1e-4),
                 **kwargs):

        self.ref_dirs = ref_dirs
        # self.surrogate_strategy = surrogate
        self.indicator = Indicator(metric='igd')
        self.repair = BasicBoundsRepair()

        # Have to define here given the need to pass ref_dirs
        if 'survival' in kwargs:
            survival = kwargs['survival']
            del kwargs['survival']
        else:
            survival = gMCRSurvival(filter_infeasible=False, use_generalised_mcr=True)
            # survival = TwoRankingSurvival(filter_infeasible=False)
            # survival = PopulationBasedEpsilonSurvival(filter_infeasible=False)

        # Number of internal offspring iterations
        if 'n_offspring_iter' in kwargs:
            self.w_max_0 = kwargs['n_offspring_iter']
        else:
            self.w_max_0 = 10

        # Output path
        if 'output_path' in kwargs:
            self.output_path = kwargs['output_path']

        super().__init__(n_population=n_population,
                         sampling=sampling,
                         selection=selection,
                         crossover=crossover,
                         mutation=mutation,
                         survival=survival,
                         surrogate=surrogate,
                         eliminate_duplicates=eliminate_duplicates,
                         **kwargs)

    def _initialise(self):
        # Initialise from the evolutionary algorithm class
        super()._initialise()

        # Generation parameters
        self.max_f_eval = self.max_gen
        print('max_eval: ', self.max_f_eval)
        self.duplication_tolerance = 1e-4

        # Reference Vectors
        self.norm = HyperplaneNormalisation(self.ref_dirs.shape[1])
        self.ideal = np.full(self.problem.n_obj, np.inf)
        self.nadir = np.full(self.problem.n_obj, -np.inf)
        self.normaliser = Normalisation()

        # Generational parameters
        self.last_cluster_index = -1
        self.last_minimised = -1
        self.cluster_range = np.arange(len(self.ref_dirs))
        self.n_init = self.evaluator.n_eval
        self.max_infill = self.max_gen - self.n_init
        self.n_infill = 1
        self.n_offspring = 800
        self.w_max = self.w_max_0                                      # Initial number of internal offspring iterations

        # RBF regression surrogate hyperparameter tuner
        self.rbf_tuning_freq = int(self.max_gen / self.problem.n_var)
        self.rbf_tuning_frac = 0.20
        self.rbf_tuner = RBFTuner(n_dim=self.problem.n_var, lb=self.problem.x_lower, ub=self.problem.x_upper,
                                  c=0.5, p_type='linear', kernel_type='gaussian',
                                  width_range=(0.1, 10), train_test_split=self.rbf_tuning_frac,
                                  max_evals=50, verbose=False)

        # Minimisation Exploitation of RBF surrogates
        self.minimise_frac = int(self.max_gen / self.problem.n_var)

        # Offspring generation flag
        self.offspring_flag = 'sbx_poly'  # 'sbx_poly', 'mode'

        # Initialise Archives of non-dominated solutions
        self.opt = self.survive_best_front(self.population, return_fronts=False, filter_feasible=True)
        self.representatives = self.opt
        self.offspring_archive = None

        # Evaluator for predicted objective values
        self.surrogate_evaluator = Evaluator()
        self.surrogate_strategy = self.surrogate

        # Data extractor
        self.filename = f"{self.problem.name.lower()}_{self.save_name}_maxgen_{round(self.max_f_eval)}_sampling_" \
                        f"{self.surrogate_strategy.n_training_pts}_seed_{self.surrogate_strategy.sampling_seed}"
        self.data_extractor = GenerationExtractor(filename=self.filename, base_path=self.output_path)
        self.performance_indicator_and_data_storage(self.population)

        # Calculate initial epsilon values
        self.gamma_min = 0.05
        self.gamma_max = 2.0
        self.gamma = self.gamma_min
        self.gamma_grad = (self.gamma_max - self.gamma_min) / (self.max_f_eval - self.n_init)
        self.gamma_intercept = self.gamma_min - self.gamma_grad * self.n_init
        self.feasible_frac = self.feasible_fraction(self.population)

    def _next(self):
        # # Reduce the dynamic constraint boundary
        # self.gamma = self.gamma_grad * self.evaluator.n_eval + self.gamma_intercept
        self.gamma = ((self.evaluator.n_eval - self.n_init) / self.max_infill) ** 6 + 0.1

        # Update Current feasible fraction in population
        self.feasible_frac = self.feasible_fraction(self.population)
        print(f"gamma: {self.gamma:.2f}, feas_frac: {self.feasible_frac:.3f}")

        # calculate the ideal and nadir points of the entire population
        self.update_norm_bounds(self.population)

        # Conduct mating using the current population
        self.offspring = self.generate_offspring(self.representatives, n_gen=self.w_max, n_offspring=self.n_offspring,
                                                 n_survive=int(self.n_offspring / 4))
        # self.offspring = self.generate_offspring(self.representatives, n_gen=20, n_offspring=300,
        #                                          n_survive=200)

        # Conduct two-stage pre-selection
        infill_population = self.roulette_wheel_infill_selection(self.offspring, n_survive=self.n_infill)

        # Find most extreme-feasible points in generated offspring
        if ((self.evaluator.n_eval + 1) % self.minimise_frac) == 0:
            extra_infill = self.select_most_feasible_diverse(self.offspring_archive)
            infill_population = Population.merge(infill_population, extra_infill)
            print(colored('----------------- Selecting most decision-diverse, feasible individual ------------------', 'green'))

        # Evaluate infill point expensively
        infill_population = self.evaluator.do(self.problem.obj_func, self.problem, infill_population)

        # Merge the offspring with the current population
        old_population = copy.deepcopy(self.population)
        self.population = Population.merge(self.population, infill_population)

        # Update optimum (non-dominated solutions)
        # self.opt, fronts = self.survive_best_front(self.population, return_fronts=True, filter_feasible=False)
        self.opt = self.survive_best_front(self.population, return_fronts=False, filter_feasible=True)
        self.representatives = self.opt

        # Hyperparameter tuning of RBF widths (for each objective surrogate)
        # if (self.n_gen % self.rbf_tuning_freq) == 0:
        #     self.regressor_tuning(fronts)

        # Calculate Performance Indicator and store generational data
        self.performance_indicator_and_data_storage(infill_population)

        # Update regressors with new infill point
        self.update_regressor(infill_population)

        # print(np.round(self.feasible_frac, 3), infill_population.extract_obj(), infill_population.extract_cons())

        # TODO: DEBUG --------------------------------------------------------------------
        # if ((self.evaluator.n_eval-1) % 50) == 0: #  or self.evaluator.n_eval == 11:
        if self.evaluator.n_eval > (self.max_f_eval-1):
            self.plot_pareto(ref_vec=self.ref_dirs, scaling=1.5, labels=['Survived', 'Old', 'New'],
                             obj_array1=self.problem.pareto_set.extract_obj(),
                             obj_array2=old_population.extract_obj(),
                             obj_array3=infill_population.extract_obj())
        # TODO: DEBUG --------------------------------------------------------------------
        
            # # TODO: VISUALISE 2D PROBLEM POF LANDSCAPES -------------------------------------------
            # n_grid = 50
            # x1 = np.linspace(self.problem.x_lower[0], self.problem.x_upper[0], n_grid)
            # x2 = np.linspace(self.problem.x_lower[1], self.problem.x_upper[1], n_grid)
            # x11, x22 = np.meshgrid(x1, x2)
            # x_mesh = np.array((x11, x22)).T
            #
            # idw_arr = np.zeros((n_grid, n_grid))
            # cons_arr = np.zeros((n_grid, n_grid, self.problem.n_con))
            # for i in range(n_grid):
            #     for j in range(n_grid):
            #         idw_arr[i, j] = predict_idw_dist((x_mesh[i, j, :] - self.problem.x_lower) / (self.problem.x_upper - self.problem.x_lower),
            #                                          (self.population.extract_var() - self.problem.x_lower) / (self.problem.x_upper - self.problem.x_lower))
            #
            #         for k, model in enumerate(self.surrogate.cons_surrogates):
            #             cons_arr[i, j, k] = model.predict(x_mesh[i, j, :])
            #
            # rank_sum_pof = predict_rank_sum_pof(cons_arr, idw_arr, self.problem.n_con, gamma=self.gamma)
            # rank_sum_conslcb = predict_conslcb(cons_arr, self.problem.n_con)
            #
            # # cons_arr[cons_arr <= 0.0] = 0.0
            # # cv = np.sum(cons_arr, axis=2)
            # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
            # ax_surro = ax.contourf(x11, x22, rank_sum_pof.T, 60, cmap='Blues')
            # ax.contour(x11, x22, rank_sum_pof.T, [0.999], colors='lime', linestyles='-.')
            #
            # # ax_surro = ax.contourf(x11, x22, rank_sum_conslcb.T, 60, cmap='Blues')
            # # ax.contour(x11, x22, rank_sum_conslcb.T, [0.001], colors='lime', linestyles='-.')
            #
            # # for i in range(self.problem.n_con):
            # #     ax.contour(x11, x22, cons_arr[:, :, i].T, [0.0], colors='red', linestyles='-.')
            #
            # x_vars = self.population.extract_var()
            # ax.scatter(x_vars[:, 0], x_vars[:, 1], color='w', edgecolor='k', s=25)
            #
            # # ax.axis('off')
            # plt.tight_layout()
            # plt.show()
            # # TODO: VISUALISE 2D PROBLEM POF LANDSCAPES -------------------------------------------

    def generate_offspring(self, representatives, n_offspring, n_survive, n_gen):
        # survived = copy.deepcopy(representatives)
        # survived = copy.deepcopy(self.population)

        # Create LHS for initialisation
        self.sampling.iterations = 1000
        self.sampling.do(n_survive, self.problem.x_lower, self.problem.x_upper, np.random.randint(0, 1e6, 1))
        survived = Population(self.problem, n_survive)
        survived.assign_var(self.problem, self.sampling.x)
        survived = self.surrogate_evaluator.do(self.surrogate.obj_func, self.problem, survived)
        survived = Population.merge(survived, self.opt)

        # Feasible-only archive
        self.offspring_archive = Population(self.problem, 0)

        for gen in range(n_gen - 1):
            # Generate offspring for specified size
            offspring = self.mating.do(self.problem, survived, n_offspring)  # GA-SBX
            offspring = self.repair.do(self.problem, offspring)

            # Evaluate predicted objective values with surrogates
            offspring = self.surrogate_evaluator.do(self.surrogate.obj_func, self.problem, offspring)

            # # Merge offspring with parent individuals
            # offspring = Population.merge(offspring, survived)

            # Filter predicted feasible-only individuals within the tau threshold
            survived = self.conv_div_feas_survival(offspring, tau=0.9)

            # Store all-feasible only individuals in archive
            self.offspring_archive = Population.merge(self.offspring_archive, survived)

            # Select offspring to survive to the next generation
            survived = RankAndCrowdingSurvival().do(self.problem, survived, n_survive=n_survive)
            # front = NonDominatedSorting().do(np.atleast_2d(survived.extract_obj()), only_non_dominated_front=True)
            # survived = survived[front]

        # Select non-dominated from feasible individuals
        front = NonDominatedSorting().do(np.atleast_2d(self.offspring_archive.extract_obj()), only_non_dominated_front=True)
        survived = self.offspring_archive[front]
        # print(len(survived1), len(survived), len(self.offspring_archive))

        # # # TODO: VISUALISE 2D PROBLEM POF LANDSCAPES -------------------------------------------
        # n_grid = 50
        # x1 = np.linspace(self.problem.x_lower[0], self.problem.x_upper[0], n_grid)
        # x2 = np.linspace(self.problem.x_lower[1], self.problem.x_upper[1], n_grid)
        # x11, x22 = np.meshgrid(x1, x2)
        # x_mesh = np.array((x11, x22)).T
        #
        # x_vars = survived.extract_var()
        # nondom_vars = survived1.extract_var()
        # x_lhs = self.population.extract_var()
        # other_vars = self.offspring_archive.extract_var()
        #
        # idw_arr = np.zeros((n_grid, n_grid))
        # cons_arr = np.zeros((n_grid, n_grid, self.problem.n_con))
        # for i in range(n_grid):
        #     for j in range(n_grid):
        #         idw_arr[i, j] = predict_idw_dist(
        #             (x_mesh[i, j, :] - self.problem.x_lower) / (self.problem.x_upper - self.problem.x_lower),
        #             (x_lhs - self.problem.x_lower) / (self.problem.x_upper - self.problem.x_lower))
        #
        #         for k, model in enumerate(self.surrogate.cons_surrogates):
        #             cons_arr[i, j, k] = model.predict(x_mesh[i, j, :])
        #
        # # print(np.min(idw_arr), np.max(idw_arr))
        # rank_sum_pof = predict_rank_sum_pof(cons_arr, idw_arr, self.problem.n_con, gamma=self.gamma, delta=1e-4)
        # # cons_arr[cons_arr <= 0.0] = 0.0
        # # cv = np.sum(cons_arr, axis=2)
        # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        # ax_surro = ax.contourf(x11, x22, rank_sum_pof.T, 60, cmap='Blues')
        # ax.contour(x11, x22, rank_sum_pof.T, [0.9], colors='lime', linestyles='-.')
        # # for i in range(self.problem.n_con):
        # #     ax.contour(x11, x22, cons_arr[:, :, i].T, [0.0], colors='red', linestyles='-.')
        #
        # ax.scatter(other_vars[:, 0], other_vars[:, 1], color='k', edgecolor='k', s=40)
        # ax.scatter(x_vars[:, 0], x_vars[:, 1], color='r', edgecolor='w', s=40)
        # ax.scatter(nondom_vars[:, 0], nondom_vars[:, 1], color='w', s=40)
        #
        # # ax.axis('off')
        # plt.tight_layout()
        # plt.show()
        # # # TODO: VISUALISE 2D PROBLEM POF LANDSCAPES -------------------------------------------

        return survived

    # def generate_offspring(self, representatives, n_offspring=300, n_gen=10):
    #     # survived = copy.deepcopy(representatives)
    #     survived = copy.deepcopy(self.population)
    #
    #     # Feasibility mechanism
    #     flag = True
    #     if self.feasible_frac > 0.0:
    #         if np.random.uniform(0.0, 1.0, 1) <= 0.3:
    #             flag = False
    #
    #     for gen in range(n_gen - 1):
    #         # Generate offspring for specified size
    #         offspring = self.mating.do(self.problem, survived, n_offspring)  # SBX
    #         offspring = self.repair.do(self.problem, offspring)
    #
    #         # Evaluate predicted objective values with surrogates
    #         offspring = self.surrogate_evaluator.do(self.surrogate.obj_func, self.problem, offspring)
    #
    #         # Merge offspring with parent individuals
    #         offspring = Population.merge(offspring, survived)
    #
    #         # Select offspring to survive to the next generation
    #         # Generalised-MCR survival
    #         # survived = self.survival.do(self.problem, offspring, self.n_population,
    #         #                             gen=self.n_gen, max_gen=self.max_gen)
    #
    #         # Non-dominated sorting of convergence, diversity and feasibility
    #         survived = self.conv_div_feas_survival(offspring, flag=flag)
    #
    #     survived1 = survived
    #     # Select non-dominated from feasible individuals
    #     fronts = NonDominatedSorting().do(survived.extract_obj(), return_rank=False)
    #     survived = survived[fronts[0]]
    #     survived2 = survived
    #
    #     # if ((self.evaluator.n_eval-1) % 50) == 0 or self.evaluator.n_eval == 10:
    #     #     # TODO: VISUALISE 2D PROBLEM POF LANDSCAPES -------------------------------------------
    #     #     n_grid = 50
    #     #     x1 = np.linspace(self.problem.x_lower[0], self.problem.x_upper[0], n_grid)
    #     #     x2 = np.linspace(self.problem.x_lower[1], self.problem.x_upper[1], n_grid)
    #     #     x11, x22 = np.meshgrid(x1, x2)
    #     #     x_mesh = np.array((x11, x22)).T
    #     #
    #     #     x_vars = survived1.extract_var()
    #     #     x_lhs = self.population.extract_var()
    #     #
    #     #     idw_arr = np.zeros((n_grid, n_grid))
    #     #     cons_arr = np.zeros((n_grid, n_grid, self.problem.n_con))
    #     #     for i in range(n_grid):
    #     #         for j in range(n_grid):
    #     #             idw_arr[i, j] = predict_idw_dist(
    #     #                 (x_mesh[i, j, :] - self.problem.x_lower) / (self.problem.x_upper - self.problem.x_lower),
    #     #                 (x_lhs - self.problem.x_lower) / (
    #     #                             self.problem.x_upper - self.problem.x_lower))
    #     #
    #     #             for k, model in enumerate(self.surrogate.cons_surrogates):
    #     #                 cons_arr[i, j, k] = model.predict(x_mesh[i, j, :])
    #     #
    #     #     rank_sum_pof = predict_rank_sum_pof(cons_arr, idw_arr, self.problem.n_con, gamma=self.gamma)
    #     #     # cons_arr[cons_arr <= 0.0] = 0.0
    #     #     # cv = np.sum(cons_arr, axis=2)
    #     #     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    #     #     ax_surro = ax.contourf(x11, x22, rank_sum_pof.T, 60, cmap='Blues')
    #     #     ax.contour(x11, x22, rank_sum_pof.T, [0.999], colors='lime', linestyles='-.')
    #     #     # for i in range(self.problem.n_con):
    #     #     #     ax.contour(x11, x22, cons_arr[:, :, i].T, [0.0], colors='red', linestyles='-.')
    #     #
    #     #     other_vars = offspring.extract_var()
    #     #     ax.scatter(other_vars[:, 0], other_vars[:, 1], color='k', edgecolor='k', s=40)
    #     #     ax.scatter(x_vars[:, 0], x_vars[:, 1], color='r', edgecolor='w', s=30)
    #     #
    #     #     nondom_vars = survived2.extract_var()
    #     #     ax.scatter(nondom_vars[:, 0], nondom_vars[:, 1], color='m', s=20)
    #     #
    #     #     # ax.axis('off')
    #     #     plt.tight_layout()
    #     #     plt.show()
    #     #     # TODO: VISUALISE 2D PROBLEM POF LANDSCAPES -------------------------------------------
    #
    #     # Create offspring of final n_offspring size
    #     # survived = self.mating.do(self.problem, survived, n_offspring)  # SBX
    #     # survived = self.repair.do(self.problem, survived)
    #
    #     # # # TODO: DEBUG --------------------------------------------------------------------
    #     # survived = self.surrogate_evaluator.do(self.surrogate.obj_func, self.problem, survived)
    #     # self.plot_pareto(ref_vec=self.ref_dirs, scaling=1, labels=['Survived', 'Reps', 'Pop'],
    #     #                  obj_array3=self.population.extract_obj(),
    #     #                  obj_array2=self.problem.pareto_set.extract_obj(),
    #     #                  obj_array1=survived.extract_obj())
    #     # # # TODO: DEBUG ---------------------------------------------------------------------
    #
    #     # if (self.evaluator.n_eval % 50) == 0:
    #     # if self.evaluator.n_eval > 199:
    #     #     # TODO: VISUALISE 2D PROBLEM POF LANDSCAPES -------------------------------------------
    #     #     n_grid = 50
    #     #     x1 = np.linspace(self.problem.x_lower[0], self.problem.x_upper[0], n_grid)
    #     #     x2 = np.linspace(self.problem.x_lower[1], self.problem.x_upper[1], n_grid)
    #     #     x11, x22 = np.meshgrid(x1, x2)
    #     #     x_mesh = np.array((x11, x22)).T
    #     #
    #     #     x_vars = survived.extract_var()
    #     #
    #     #     idw_arr = np.zeros((n_grid, n_grid))
    #     #     cons_arr = np.zeros((n_grid, n_grid, self.problem.n_con))
    #     #     for i in range(n_grid):
    #     #         for j in range(n_grid):
    #     #             idw_arr[i, j] = predict_idw_dist(
    #     #                 (x_mesh[i, j, :] - self.problem.x_lower) / (self.problem.x_upper - self.problem.x_lower),
    #     #                 (x_vars - self.problem.x_lower) / (
    #     #                             self.problem.x_upper - self.problem.x_lower))
    #     #
    #     #             for k, model in enumerate(self.surrogate.cons_surrogates):
    #     #                 cons_arr[i, j, k] = model.predict(x_mesh[i, j, :])
    #     #
    #     #     rank_sum_pof = predict_rank_sum_pof(cons_arr, idw_arr, self.problem.n_con, gamma=self.gamma)
    #     #     # cons_arr[cons_arr <= 0.0] = 0.0
    #     #     # cv = np.sum(cons_arr, axis=2)
    #     #     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    #     #     ax_surro = ax.contourf(x11, x22, rank_sum_pof.T, 60, cmap='Blues')
    #     #     ax.contour(x11, x22, rank_sum_pof.T, [0.999], colors='lime', linestyles='-.')
    #     #     for i in range(self.problem.n_con):
    #     #         ax.contour(x11, x22, cons_arr[:, :, i].T, [0.0], colors='red', linestyles='-.')
    #     #
    #     #     ax.scatter(x_vars[:, 0], x_vars[:, 1], color='w', edgecolor='k', s=25)
    #     #     # ax.axis('off')
    #     #     plt.tight_layout()
    #     #     plt.show()
    #     #     # TODO: VISUALISE 2D PROBLEM POF LANDSCAPES -------------------------------------------
    #
    #     return survived

    def conv_div_feas_survival(self, population, tau=0.9):
        # Probability of feasibility
        pof = self.predict_rank_sum_pof(population, gamma=self.gamma, delta=1e-4)[:, None]

        # Constrained-lower confidence bounds (without uncertainty)
        # conslcb = self.predict_conslcb(population)[:, None]

        # Decision-space distance between offspring
        # min_dist = np.amin(self.calc_dist(population), axis=1)[:, None]

        # EDN Convergence
        # dom_ranks = self.predict_edn_ranks(population)[:, None]

        # Angle-Diversity in objectives calculation
        # min_angle = self.predict_angle_diversity(population)[:, None]

        # Objective values
        # if flag:
        #     # Diversity-feasibility focused
        #     new_obj_arr = -np.hstack((min_angle, pof))
        # else:
        #     # Convergence-diversity-feasibility focused
        #     new_obj_arr = -np.hstack((dom_ranks, min_angle, pof))
        # new_obj_arr = -np.hstack((min_dist, pof))

        # Find non-dominated front
        # fronts = NonDominatedSorting().do(new_obj_arr, return_rank=False)
        # survived = population[fronts[0]]

        # Select feasible cut-off above the specified feasibility threshold
        selected = (pof.flatten() > tau)
        survived = population[selected]

        # TODO: roulette-wheel selection of feasible fraction
        # rand_int = random.choices(population=list(range(len(population))), weights=pof.flatten(), k=max(int(tau*np.count_nonzero(selected)), 1))
        # # rand_int = random.choices(population=list(range(len(population))), weights=pof.flatten(), k=200)
        # survived = population[rand_int]

        # fig = plt.figure()
        # # ax = plt.axes(projection='2d')
        # fig, ax = plt.subplots(ncols=1, nrows=1)
        # # ax.scatter3D(new_obj_arr[:, 0], new_obj_arr[:, 1], new_obj_arr[:, 2], c='k', alpha=0.2)
        # # ax.scatter3D(new_obj_arr[fronts[0], 0], new_obj_arr[fronts[0], 1], new_obj_arr[fronts[0], 2], c='r', alpha=1.0)
        # ax.scatter(new_obj_arr[:, 0], new_obj_arr[:, 1], c='k', alpha=0.2)
        # ax.scatter(new_obj_arr[fronts[0], 0], new_obj_arr[fronts[0], 1], c='r', alpha=1.0)
        # # ax.set_xlabel('Dom Ranks')
        # ax.set_xlabel('-min_dist')
        # ax.set_ylabel('-PoF')
        # # ax.legend()
        # plt.show()

        return survived

    def roulette_wheel_infill_selection(self, offspring, n_survive=1):
        # Determine which offspring are too close to existing population
        is_duplicate = self.check_duplications(offspring, self.population,
                                               epsilon=self.duplication_tolerance)
        offspring = offspring[np.invert(is_duplicate)[0]]
        offspring = self.surrogate_evaluator.do(self.surrogate.obj_func, self.problem, offspring)

        # Angle-Diversity in objectives calculation
        min_angle = self.predict_angle_diversity(offspring)
        min_angle = (min_angle - np.min(min_angle) / np.max(min_angle) - np.min(min_angle))
        min_angle = min_angle / np.sum(min_angle)

        # TODO: Decision-space distance between offspring
        # min_dist = np.amin(self.calc_dist(offspring, other=self.population), axis=1)
        # selected = np.array([np.argmax(min_dist)])

        # # EDN Convergence
        # dom_ranks = self.predict_edn_ranks(offspring)
        # dom_ranks = (dom_ranks - np.min(dom_ranks) / np.max(dom_ranks) - np.min(dom_ranks))

        # Probability of Feasibility calculation
        # pof = self.predict_pof_rbf(offspring)
        # pof = self.predict_rank_sum_pof(offspring, gamma=self.gamma)

        # Convert problem as unconstrained and set the roulette-wheel selection probabilities
        # selection_probabilities = min_angle * pof
        # selection_probabilities = 0.7 * min_angle + 0.3 * dom_ranks
        # selection_probabilities = selection_probabilities / np.sum(selection_probabilities)

        # Conduct roulette-wheel selection given the selection probabilities
        rand_int = random.choices(population=list(range(len(offspring))), weights=min_angle, k=n_survive)
        # rand_int = np.random.randint(0, len(offspring), 1)
        infill_population = copy.deepcopy(offspring[rand_int])

        # Direct selection
        # selected = np.array([np.argmax(min_angle)])
        # infill_population = copy.deepcopy(offspring[selected])

        # # TODO: DEBUG --------------------------------------------------------------------
        # self.plot_pareto(ref_vec=self.ref_dirs, scaling=5.0, labels=['Front', 'Offs', 'infill'],
        #                  obj_array1=self.problem.pareto_set.extract_obj(),
        #                  obj_array2=offspring.extract_obj(),
        #                  # obj_array3=offspring[np.array([np.argmax(selection_probabilities)])].extract_obj())
        #                  obj_array3=infill_population.extract_obj())
        # # TODO: DEBUG --------------------------------------------------------------------

        return infill_population

    def select_most_feasible_diverse(self, offspring):
        # Calculate minimum distance between offspring and existing population
        min_dist = np.amin(self.calc_dist(offspring, other=self.population), axis=1)

        # Select most decision-space diverse, feasible offspring as extra infill point
        selected = np.array([np.argmax(min_dist)])
        infill = offspring[selected]

        return infill

    def regressor_tuning(self, fronts):
        print(colored('----------------- Hyper-tuning Regressor Surrogate Widths ------------------', 'green'))
        # Conduct optimisation of each objective surrogate
        opt_models, opt_params = self.rbf_tuner.do(problem=self.problem,
                                                   population=self.population,
                                                   n_obj=len(self.surrogate.obj_surrogates),
                                                   fronts=fronts)
        # Set newly optimised RBF models
        for cntr, model in enumerate(opt_models):
            self.surrogate.obj_surrogates[cntr].model = model
        print(colored('----------------------------------------------------------------------------', 'green'))

    def performance_indicator_and_data_storage(self, population):
        # Calculate performance indicator
        igd = self.indicator.do(self.problem, self.opt, self.evaluator.n_eval, return_value=True)
        print('IGD: ', np.round(igd, 4))

        # Store fronts and population
        self.data_extractor.add_generation(population, self.n_gen)
        self.data_extractor.add_front(self.opt, self.n_gen, indicator_values=np.array(igd))

    def update_regressor(self, infill):
        # Ensure atleast one infill point is available
        if len(infill) == 0:
            print(colored('Did not find an infill point!', 'yellow'))
            return

        # Extract update training data
        new_vars = np.atleast_2d(infill.extract_var())
        new_obj = np.atleast_2d(infill.extract_obj())
        new_cons = np.atleast_2d(infill.extract_cons())

        try:
            # Update training data and train objective models
            for obj_cntr in range(self.problem.n_obj):
                # Objective surrogates
                self.surrogate.obj_surrogates[obj_cntr].add_points(new_vars, new_obj[:, obj_cntr])
                self.surrogate.obj_surrogates[obj_cntr].train()
        except np.linalg.LinAlgError:
            print(colored('Objective regressor update failed!', 'red'))

        try:
            # Update training data and train constraint models
            for cons_cntr in range(self.problem.n_con):
                # Constraint surrogates
                self.surrogate.cons_surrogates[cons_cntr].add_points(new_vars, new_cons[:, cons_cntr])
                self.surrogate.cons_surrogates[cons_cntr].train()
        except np.linalg.LinAlgError:
            print(colored('Constraint regressor update failed!', 'red'))

    def update_norm_bounds(self, population):
        # Extract objectives
        obj_array = population.extract_obj()

        # Find lower and upper bounds
        f_min = np.min(obj_array, axis=0)
        f_max = np.max(obj_array, axis=0)

        # Update the ideal and nadir points
        self.ideal = np.minimum(f_min, self.ideal)
        self.nadir = f_max
        
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

    def predict_pof_rbf(self, offspring, scale=1.0):
        pop_x_vars = self.population.extract_var()
        off_x_vars = offspring.extract_var()
        cons_arr = offspring.extract_cons()

        # Normalise inputs to [0, 1]
        pop_x_vars = (pop_x_vars - self.problem.x_lower) / (self.problem.x_upper - self.problem.x_lower)
        off_x_vars = (off_x_vars - self.problem.x_lower) / (self.problem.x_upper - self.problem.x_lower)

        # IDW uncertainty measure
        sigma = self.predict_idw_dist(off_x_vars, pop_x_vars)

        # Scaling for constraints
        max_abs_con_vals = np.amax(np.abs(cons_arr), axis=0)
        max_abs_con_vals[max_abs_con_vals == 0.0] = 1.0

        pof = np.ones(len(offspring))
        dummy_ones = np.ones(len(offspring))
        for i in range(self.problem.n_con):
            # Calculate violation term
            violation = -cons_arr[:, i] / (sigma * max_abs_con_vals[i] * scale)

            # Calculate PoF for individual constraint
            individual_pof = np.minimum(2 * norm.cdf(violation), dummy_ones)

            # Aggregate constraints via product
            pof *= individual_pof

        return pof

    def predict_rank_sum_pof(self, offspring, delta=1e-5, gamma=1.0):
        pop_x_vars = self.population.extract_var()
        off_x_vars = offspring.extract_var()
        cons_arr = copy.deepcopy(offspring.extract_cons())

        # Normalise inputs to [0, 1]
        pop_x_vars = (pop_x_vars - self.problem.x_lower) / (self.problem.x_upper - self.problem.x_lower)
        off_x_vars = (off_x_vars - self.problem.x_lower) / (self.problem.x_upper - self.problem.x_lower)

        # IDW uncertainty measure
        sigma = np.zeros(len(off_x_vars))
        for i in range(len(off_x_vars)):
            sigma[i] = self.predict_idw_dist(off_x_vars[i], pop_x_vars)

        # Transform feasible constraint values to 0.0
        # cons_arr = cons_arr - self.eps  # TODO: Eps-feasibility
        cons_arr[cons_arr <= 0.0] = 0.0

        # Calculate ranks for individual constraints
        ranks = np.zeros(cons_arr.shape)
        for cntr in range(self.problem.n_con):
            ranks[:, cntr] = rankdata(cons_arr[:, cntr].flatten(), method='dense') - 1

        # Aggregate and normalise ranks
        rank_sum = np.sum(ranks, axis=1)
        normalised_rank_sum = (rank_sum - np.min(rank_sum)) / (np.max(rank_sum) - np.min(rank_sum))

        # Calculate the rank-sum probability of feasibility
        beta = gamma / (sigma + delta)
        pof = np.exp(-beta * normalised_rank_sum ** 2)

        return pof

    def predict_conslcb(self, population):
        # Extract constraints from population
        cons_arr = copy.deepcopy(population.extract_cons())

        # Absolute value of constraints
        cons_arr = np.abs(cons_arr)

        # Calculate ranks for individual constraints
        ranks = np.zeros(cons_arr.shape)
        for cntr in range(self.problem.n_con):
            ranks[:, cntr] = rankdata(cons_arr[:, cntr].flatten(), method='dense') - 1

        # Aggregate and normalise ranks
        rank_sum = np.sum(ranks, axis=1)
        normalised_rank_sum = (rank_sum - np.min(rank_sum)) / (np.max(rank_sum) - np.min(rank_sum))

        return normalised_rank_sum

    def predict_edn_ranks(self, population):
        dom_mat = calculate_domination_matrix(population.extract_obj(), domination_type='pareto')
        dom_mat[dom_mat != 1] = 0
        dom_ranks = np.sum(dom_mat, axis=1)
        dom_ranks = dom_ranks / np.max(dom_ranks)

        return dom_ranks

    def predict_angle_diversity(self, population):
        obj_arr = np.atleast_2d(population.extract_obj())
        real_obj_arr = np.atleast_2d(self.population.extract_obj())

        # Find lower and upper bounds
        f_min = np.min(obj_arr, axis=0)
        f_max = np.max(obj_arr, axis=0)

        # Normalise objective
        obj_arr = (obj_arr - f_min) / (f_max - f_min)
        real_obj_arr = (real_obj_arr - f_min) / (f_max - f_min)

        # Scale by vector distances
        dist_to_ideal = np.linalg.norm(obj_arr, axis=1) # TODO: crashes in 20d mw7 ~173 evals
        dist_to_ideal[dist_to_ideal < 1e-64] = 1e-64
        obj_arr = obj_arr / dist_to_ideal[:, None]

        dist_to_ideal = np.linalg.norm(real_obj_arr, axis=1)
        dist_to_ideal[dist_to_ideal < 1e-64] = 1e-64
        real_obj_arr = real_obj_arr / dist_to_ideal[:, None]

        # Calculate Angle between offspring and population
        acute_angle = np.arccos(obj_arr @ real_obj_arr.T)
        min_angle = np.min(acute_angle, axis=1)

        return min_angle

    @staticmethod
    def predict_idw_dist(x, x_lhs):
        # Compute IDW distance
        x = np.atleast_2d(x)
        dist = distance.cdist(x, x_lhs)
        if np.min(dist) == 0:
            z_function = 0
        else:
            weight_i = 1 / dist ** 2
            weight_sum = np.sum(weight_i)
            z_function = np.arctan(1 / weight_sum)

        return z_function

    # @staticmethod
    def survive_best_front(self, population, return_fronts=False, filter_feasible=True):
        population = copy.deepcopy(population)
        fronts = None
        survived = None

        # Filter by constraint violation
        feasible_frac = 0.0
        if filter_feasible:
            # Select feasible population only
            cons_array = population.extract_cons_sum()
            feasible_mask = cons_array == 0.0
            feasible_frac = np.count_nonzero(feasible_mask) / len(population)
            if feasible_frac > 0.0:
                # Conduct non-dominated sorting and select first front only from feasible individuals
                fronts = NonDominatedSorting().do(np.atleast_2d(population[feasible_mask].extract_obj()), return_rank=False)
                survived = population[fronts[0]]

        if not filter_feasible or feasible_frac == 0.0:
            # Conduct survival on population for next generation
            survived = self.survival.do(self.problem, population, self.n_population,
                                        gen=self.n_gen, max_gen=self.max_gen)

        if return_fronts:
            return survived, fronts
        return survived

    @staticmethod
    def feasible_fraction(population):
        cons_array = population.extract_cons_sum()
        feasible_mask = cons_array == 0.0
        feasible_frac = np.count_nonzero(feasible_mask) / len(population)

        return feasible_frac

    @staticmethod
    def reduce_boundary(eps0, n_gen, max_gen, cp, delta=1e-8):
        A = eps0 + delta
        base_num = np.log((eps0 + delta) / delta)
        B = max_gen / np.power(base_num, 1 / cp) + 1e-15
        eps = A * np.exp(-(n_gen / B) ** cp) - delta
        eps[eps < 0.0] = 0.0

        return eps

    @staticmethod
    def get_max_cons_viol(population):
        cons_arr = copy.deepcopy(population.extract_cons())
        cons_arr[cons_arr <= 0.0] = 0.0
        max_cons_viol = np.amax(cons_arr, axis=0)
        max_cons_viol[max_cons_viol == 0] = 1.0

        return max_cons_viol

    @staticmethod
    def plot_pareto(ref_vec=None, obj_array1=None, obj_array2=None, obj_array3=None, fronts=None, scaling=1.5, labels=None):

        n_obj = len(ref_vec[0])

        if labels is None:
            labels = ['theta_pop', 'pareto_pop', 'infill']

        # 2D Plot
        if n_obj == 2:
            fig, ax = plt.subplots(1, 1, figsize=(9, 7))
            fig.supxlabel('Obj 1', fontsize=14)
            fig.supylabel('Obj 2', fontsize=14)

            # Plot reference vectors
            if ref_vec is not None:
                origin = np.zeros(len(ref_vec))
                x_vec = scaling * np.vstack((origin, ref_vec[:, 0])).T
                y_vec = scaling * np.vstack((origin, ref_vec[:, 1])).T
                for i in range(len(x_vec)):
                    if i == 0:
                        ax.plot(x_vec[i], y_vec[i], color='black', linewidth=0.5, label='reference vectors')
                    else:
                        ax.plot(x_vec[i], y_vec[i], color='black', linewidth=0.5)

            if obj_array1 is not None:
                obj_array1 = np.atleast_2d(obj_array1)
                if fronts is not None:
                    for i, frnt in enumerate(fronts):
                        ax.scatter(obj_array1[frnt, 0], obj_array1[frnt, 1], color=line_colors[i], s=75, label=f"rank {i}")
                else:
                    ax.scatter(obj_array1[:, 0], obj_array1[:, 1], color=line_colors[6], s=150, label=labels[0])

            try:
                if obj_array2 is not None:
                    obj_array2 = np.atleast_2d(obj_array2)
                    if fronts is not None:
                        for i, frnt in enumerate(fronts):
                            ax.scatter(obj_array2[frnt, 0], obj_array2[frnt, 1], color=line_colors[i], s=75, label=f"rank {i}")
                    else:
                        ax.scatter(obj_array2[:, 0], obj_array2[:, 1], color=line_colors[1], s=50, label=labels[1])
            except:
                pass

            if obj_array3 is not None:
                obj_array3 = np.atleast_2d(obj_array3)
                if fronts is not None:
                    for i, frnt in enumerate(fronts):
                        ax.scatter(obj_array3[frnt, 0], obj_array3[frnt, 1], color=line_colors[i], s=75, label=f"rank {i}")
                else:
                    ax.scatter(obj_array3[:, 0], obj_array3[:, 1], color=line_colors[2], s=25, label=labels[2])

        # 3D Plot
        elif n_obj == 3:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.set_xlabel('Obj 1')
            ax.set_ylabel('Obj 2')
            ax.set_zlabel('Obj 3')

            # Plot reference vectors
            if ref_vec is not None:
                origin = np.zeros(len(ref_vec))
                x_vec = scaling * np.vstack((origin, ref_vec[:, 0])).T
                y_vec = scaling * np.vstack((origin, ref_vec[:, 1])).T
                z_vec = scaling * np.vstack((origin, ref_vec[:, 2])).T
                for i in range(len(x_vec)):
                    if i == 0:
                        ax.plot(x_vec[i], y_vec[i], z_vec[i], color='black', linewidth=0.5, label='reference vectors')
                    else:
                        ax.plot(x_vec[i], y_vec[i], z_vec[i], color='black', linewidth=0.5)

            if obj_array1 is not None:
                obj_array1 = np.atleast_2d(obj_array1)
                if fronts is not None:
                    for i, frnt in enumerate(fronts):
                        ax.scatter3D(obj_array1[frnt, 0], obj_array1[frnt, 1], obj_array1[frnt, 2], color=line_colors[i], s=50, label=f"rank {i}")
                else:
                    ax.scatter3D(obj_array1[:, 0], obj_array1[:, 1], obj_array1[:, 2], color=line_colors[0], s=50,
                                 label=labels[0])

            if obj_array2 is not None:
                obj_array2 = np.atleast_2d(obj_array2)
                if fronts is not None:
                    for i, frnt in enumerate(fronts):
                        ax.scatter3D(obj_array2[frnt, 0], obj_array2[frnt, 1], obj_array2[frnt, 2], color=line_colors[i], s=50, label=f"rank {i}")
                else:
                    ax.scatter3D(obj_array2[:, 0], obj_array2[:, 1], obj_array2[:, 2], color=line_colors[1], s=25,
                                 label=labels[1])

            if obj_array3 is not None:
                obj_array3 = np.atleast_2d(obj_array3)
                if fronts is not None:
                    for i, frnt in enumerate(fronts):
                        ax.scatter3D(obj_array3[frnt, 0], obj_array3[frnt, 1], obj_array3[frnt, 2], color=line_colors[i], s=50, label=f"rank {i}")
                else:
                    ax.scatter3D(obj_array3[:, 0], obj_array3[:, 1], obj_array3[:, 2], color=line_colors[2], s=15,
                                 label=labels[2])

        plt.legend(loc='best', frameon=False)
        plt.show()
        # plt.show(block=False)
        # plt.pause(3)
        # plt.close()
        # plt.savefig('/home/juan/PycharmProjects/optimisation_framework/multi_obj/results/zdt1_mmode_gen_' + str(len(self.surrogate.population)) + '.png')


def predict_idw_dist(x, x_lhs):
    # Compute IDW distance
    x = np.atleast_2d(x)
    dist = distance.cdist(x, x_lhs)
    if np.min(dist) == 0:
        z_function = 0
    else:
        weight_i = 1 / dist ** 2
        weight_sum = np.sum(weight_i)
        z_function = np.arctan(1 / weight_sum)

    return z_function


def predict_rank_sum_pof(cons_arr, idw_arr, n_con, delta=1e-5, gamma=0.01):
    # Transform feasible constraint values to 0.0
    cons_arr = copy.deepcopy(cons_arr)
    cons_arr[cons_arr <= 0.0] = 0.0

    # Calculate ranks for individual constraints
    n, m, k = cons_arr.shape
    ranks = np.zeros(cons_arr.shape)
    for i in range(n_con):
        individual_rank = rankdata(cons_arr[:, :, i].flatten(), method='dense') - 1
        ranks[:, :, i] = individual_rank.reshape((n, m))

    # Aggregate and normalise ranks
    rank_sum = np.sum(ranks, axis=2)
    normalised_rank_sum = rank_sum / np.max(rank_sum)

    # Calculate the rank-sum probability of feasibility
    beta = gamma / (idw_arr + delta)
    pof = np.exp(-beta * normalised_rank_sum ** 2)

    return pof


def predict_conslcb(cons_arr, n_con):
    # Absolute value of constraints
    cons_arr = np.abs(cons_arr)

    # Calculate ranks for individual constraints
    n, m, k = cons_arr.shape
    ranks = np.zeros(cons_arr.shape)
    for i in range(n_con):
        individual_rank = rankdata(cons_arr[:, :, i].flatten(), method='dense') - 1
        ranks[:, :, i] = individual_rank.reshape((n, m))

    # Aggregate and normalise ranks
    rank_sum = np.sum(ranks, axis=2)
    normalised_rank_sum = (rank_sum - np.min(rank_sum)) / (np.max(rank_sum) - np.min(rank_sum))

    return normalised_rank_sum