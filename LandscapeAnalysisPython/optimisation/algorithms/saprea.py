import copy
import random
import numpy as np

from scipy.spatial import distance
from termcolor import colored

from optimisation.algorithms.sa_evolutionary_algorithm import SAEvolutionaryAlgorithm

from optimisation.model.population import Population
from optimisation.model.duplicate import DefaultDuplicateElimination
from optimisation.model.repair import BasicBoundsRepair, BounceBackBoundsRepair
from optimisation.model.evaluator import Evaluator

from optimisation.operators.sampling.lhs_loader import LHSLoader
from optimisation.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from optimisation.operators.mutation.polynomial_mutation import PolynomialMutation
from optimisation.operators.mutation.de_mutation import DifferentialEvolutionMutation
from optimisation.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from optimisation.operators.selection.random_selection import RandomSelection
from optimisation.operators.survival.self_adaptive_feasible_ratio_epsilon_survival import SelfAdaptiveFeasibleRatioEpsilonSurvival
from optimisation.operators.survival.rank_and_crowding_survival import RankAndCrowdingSurvival
from optimisation.operators.survival.two_ranking_survival import TwoRankingSurvival
from optimisation.operators.survival.population_based_epsilon_survival import PopulationBasedEpsilonSurvival

from optimisation.surrogate.bilog_rbf_tuner import RBFTuner

from optimisation.metrics.indicator import Indicator
from optimisation.output.generation_extractor import GenerationExtractor
from optimisation.util.non_dominated_sorting import NonDominatedSorting
from optimisation.util.calculate_hypervolume import calculate_hypervolume_pygmo
from optimisation.util.misc import bilog_transform

import matplotlib.pyplot as plt
plt.style.use('seaborn-talk')
np.set_printoptions(suppress=True)
line_colors = ['green', 'blue', 'red', 'orange', 'cyan', 'lawngreen', 'm', 'orangered', 'sienna', 'gold', 'violet', 'indigo', 'cornflowerblue']


class SAPREA(SAEvolutionaryAlgorithm):
    """
    Baseline unconstrained SAEA
    """

    def __init__(self,
                 ref_dirs=None,
                 n_population=20,
                 surrogate=None,
                 sampling=LHSLoader(),
                 selection=RandomSelection(),
                 crossover=SimulatedBinaryCrossover(eta=20, prob=1.0),
                 mutation=PolynomialMutation(eta=20, prob=None),
                 eliminate_duplicates=DefaultDuplicateElimination(epsilon=1e-4),
                 **kwargs):

        self.ref_dirs = ref_dirs
        self.indicator = Indicator(metric='igd')
        self.repair = BasicBoundsRepair()

        # Have to define here given the need to pass ref_dirs
        if 'survival' in kwargs:
            survival = kwargs['survival']
            del kwargs['survival']
        else:
            # survival = SelfAdaptiveFeasibleRatioEpsilonSurvival(filter_infeasible=False)
            survival = RankAndCrowdingSurvival(filter_infeasible=True)

        # Number of internal offspring iterations
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

        # SAEA Parameters
        self.max_f_eval = self.max_gen
        print('max_eval: ', self.max_f_eval)
        self.duplication_tolerance = 1e-4
        self.last_selected = -1
        self.n_init = self.evaluator.n_eval
        self.max_infill = self.max_gen - self.n_init
        self.n_infill = 1
        self.n_offspring = 300

        # Number of internal offspring iterations
        self.w_max = [5, 10, 15, 20]
        self.w = 20
        self.pos = -1

        # Evaluator for surrogate predictions
        self.surrogate_evaluator = Evaluator()

        # Survival mechanisms
        self.lowest_cv = None
        self.nondom = NonDominatedSorting()
        self.rank_and_crowd = RankAndCrowdingSurvival(filter_infeasible=False)
        self.opt, self.reps = self.survival_mechanism(self.population)

        # RBF regression surrogate hyperparameter tuner
        self.rbf_tuning_frac = 0.20
        self.rbf_tuner = RBFTuner(n_dim=self.problem.n_var, lb=self.problem.x_lower, ub=self.problem.x_upper,
                                  c=0.5, p_type='linear', kernel_type='gaussian',
                                  width_range=(0.1, 10), train_test_split=self.rbf_tuning_frac,
                                  max_evals=50, verbose=False)
        self.hopt_frac = int(self.max_gen / 25)
        self.transform_list = [0 for _ in range(self.problem.n_obj)]
        self.last_minimised = -1

        # Data extractor
        self.filename = f"{self.problem.name.lower()}_{self.save_name}_maxgen_{round(self.max_f_eval)}_sampling_" \
                        f"{self.surrogate.n_training_pts}_seed_{self.surrogate.sampling_seed}"
        self.data_extractor = GenerationExtractor(filename=self.filename, base_path=self.output_path)
        self.indicator = Indicator(metric='igd')
        self.igd = np.inf
        self.performance_indicator_and_data_storage(self.population)

        # Differential Evolution
        self.de_mutation = DifferentialEvolutionMutation(problem=self.problem, method='diversity')
        self.repair = BasicBoundsRepair()
        self.infilled_pop = Population(self.problem)
        self.offspring_archive = Population(self.problem)

        # TODO: Evaluate Global surrogate model accuracy
        lhs = np.genfromtxt('/home/juan/PycharmProjects/optimisation_framework/multi_obj/10000.lhs')
        self.large_lhs = (self.problem.x_upper - self.problem.x_lower) * lhs + self.problem.x_lower

        # TODO: Evaluate Local surrogate model accuracy (pareto front)
        # self.large_lhs = np.genfromtxt('/home/juan/PycharmProjects/optimisation_framework/multi_obj/cases/MODAct_files/ct1_vars.txt')

        self.test_pop = Population(self.problem, len(self.large_lhs))
        self.test_pop.assign_var(self.problem, self.large_lhs)
        self.test_pop = self.surrogate_evaluator.do(self.problem.obj_func, self.problem, self.test_pop)

        self.surro_pop = self.surrogate_evaluator.do(self.surrogate.obj_func, self.problem, copy.deepcopy(self.test_pop))
        obj_rmse = (1 / len(self.large_lhs)) * np.sum(np.power(self.test_pop.extract_obj() - self.surro_pop.extract_obj(), 2), axis=0)

        self.rmse_evals = np.array([self.evaluator.n_eval])
        self.rmse_obj = obj_rmse
        print(f"OBJ RMSE: {obj_rmse}")

    def _next(self):
        # Internal Offspring Generation
        self.offspring = self.generate_offspring(self.reps, n_offspring=self.n_offspring)

        # # TODO: DEBUG PLOT
        if (self.evaluator.n_eval-1) % 25 == 0:
            # self.plot_subplots(self.population, self.offspring, self.problem.pareto_set)
            self.surro_pop = self.surrogate_evaluator.do(self.surrogate.obj_func, self.problem, copy.deepcopy(self.test_pop))
            obj_rmse = (1 / len(self.large_lhs)) * np.sum(np.power(self.test_pop.extract_obj() - self.surro_pop.extract_obj(), 2), axis=0)
            print(f"OBJ RMSE: {obj_rmse}")

            self.rmse_evals = np.hstack((self.rmse_evals, self.evaluator.n_eval))
            self.rmse_obj = np.vstack((self.rmse_obj, obj_rmse))

        # Probabilistic Infill Point Selection
        infill = self.probabilistic_infill_selection(self.offspring, n_survive=self.n_infill)

        # Find the predicted surrogate extremes for the selected objective
        if (self.n_gen % self.hopt_frac) == 0:
            # Minimisation of surrogate predictions
            print(colored('----------------- Minimising Regressor Surrogate Predictions ------------------', 'green'))
            extra_infill = self.minimise_regressors(self.offspring, infill=infill)
            infill = Population.merge(infill, extra_infill)

        if len(infill) == 0:
            return

        # Expensive Evaluation of Infill Point
        infill = self.evaluator.do(self.problem.obj_func, self.problem, infill)

        # Update Archives
        self.population = Population.merge(self.population, infill)
        self.opt, self.reps = self.survival_mechanism(self.population)
        self.performance_indicator_and_data_storage(infill)
        print(f"N opt: {len(self.opt)}, N reps: {len(self.reps)}")

        # # Hyperparameter tuning of RBF widths (for each objective AND constraint surrogate)
        # if (self.n_gen % self.hopt_frac) == 0:
        #     train_pop, test_pop = self.split_population_rbf_tuning()
        #     print(colored('----------------- Hyper-tuning Regressor Surrogate Widths ------------------', 'green'))
        #     # Conduct rbf optimisation of each objective surrogate
        #     opt_models, opt_params = self.rbf_tuner.do_obj(problem=self.problem,
        #                                                    population=copy.deepcopy(self.population),
        #                                                    train_population=train_pop,
        #                                                    test_population=test_pop)
        #     # Set newly optimised RBF models
        #     for cntr, model in enumerate(opt_models):
        #         self.surrogate.obj_surrogates[cntr].model = model
        #         self.transform_list[cntr] = opt_params[cntr][1]
        #     print(self.transform_list)
        #     print(colored('----------------------------------------------------------------------------', 'green'))

        # Surrogate Management
        self.update_regressors(infill)

        # Debugging Output
        print(colored(f"IGD: {self.igd:.3f} Obj: {infill.extract_obj()}", 'yellow'))

        # if self.evaluator.n_eval > (self.max_f_eval - 1):
        #     self.plot_pareto(ref_vec=self.ref_dirs, scaling=1.5, labels=['Pareto', 'Pop', 'Opt'],
        #                      obj_array1=self.problem.pareto_set.extract_obj(),
        #                      obj_array2=self.population.extract_obj(),
        #                      obj_array3=self.opt.extract_obj())

        if self.evaluator.n_eval > (self.max_f_eval - 1):
            self.plot_pareto(ref_vec=self.ref_dirs, scaling=1.5, labels=['Pareto', 'Pop', 'Opt'],
                             obj_array1=self.problem.pareto_set.extract_obj(),
                             obj_array2=self.population.extract_obj(),
                             obj_array3=self.opt.extract_obj())

            fig, ax = plt.subplots(1, 1, figsize=(9, 7))
            plt.xlabel('Number of Function Evaluations', fontsize=14)
            plt.ylabel('Relative RMSE error', fontsize=14)
            for i in range(self.problem.n_obj):
                ax.plot(self.rmse_evals, self.rmse_obj[:, i] / self.rmse_obj[0, i], '-o', label=f'Obj {i+1}')
            plt.legend()
            # plt.show()
            plt.savefig('/home/juan/Desktop/biobj_54_global_rmse_history_igd_9000.png')
            plt.show()

    def generate_offspring(self, representatives, n_offspring):
        # Select number of internal offspring generations
        self.pos += 1
        if self.pos == len(self.w_max):
            self.pos = 0
        self.w = self.w_max[self.pos]

        # Select representative solutions to generate offspring from
        survived = copy.deepcopy(representatives)

        # Conduct offspring generations
        for gen in range(self.w):
            # Generate GA and DE offspring
            offspring = self.generate_ga_de_offspring(survived, n_offspring=n_offspring, evaluate=True)
            if gen < (self.w - 1):
                offspring = Population.merge(offspring, survived)

            # Select most promising individuals to survive
            survived = self.rank_and_crowd.do(self.problem, offspring, n_survive=int(n_offspring / 5))

        # self.plot_pareto(ref_vec=self.ref_dirs, scaling=1.5, labels=['Pareto', 'Pop', 'Opt'],
        #                  obj_array1=self.problem.pareto_set.extract_obj(),
        #                  obj_array2=self.opt.extract_obj(),
        #                  obj_array3=survived.extract_obj())

        return survived

    def generate_ga_de_offspring(self, survived, n_offspring, evaluate=True):
        n_to_gen = min(len(survived),  int(n_offspring / 4))

        # Generate DE offspring (MODE)
        if n_to_gen > 4:
            offspring1 = self.de_mutation.do(survived, self.reps, n_to_gen)
            offspring1 = self.repair.do(self.problem, offspring1)
        else:
            offspring1 = Population(self.problem, 0)
            n_to_gen = 0

        # Generate GA-SBX offpsring
        offspring = self.mating.do(self.problem, survived, n_offspring=n_offspring - n_to_gen)
        offspring = self.repair.do(self.problem, offspring)

        # Merge all offspring together
        offspring = Population.merge(offspring, offspring1)

        # Evaluate offspring with surrogates
        if evaluate:
            offspring = self.surrogate_evaluator.do(self.surrogate.obj_func, self.problem, offspring)
        # offspring = Population.merge(offspring, survived)

        return offspring

    def probabilistic_infill_selection(self, offspring, n_survive=1):
        # Determine which offspring are too close to existing population
        is_duplicate = self.check_duplications(offspring, self.population, epsilon=self.duplication_tolerance)
        offspring = offspring[np.invert(is_duplicate)]
        if len(offspring) == 0:
            return Population(self.problem)

        # Calculate angle diversity between offspring and the current best front
        comparison_population = Population.merge(copy.deepcopy(self.reps), copy.deepcopy(self.infilled_pop[-4:]))
        min_angle = self.predict_angle_diversity(offspring, comparison_population)

        # Calculate Hypervolume Improvement of each offspring
        hvi = self.calculate_hypervolume_improvement(offspring, self.reps)
        hvi = (hvi - np.min(hvi)) / (np.max(hvi) - np.min(hvi))
        probabilities = 2 * hvi + min_angle

        # Conduct probabilistic roulette-wheel infill selection
        rand_int = random.choices(population=list(range(len(offspring))), weights=probabilities, k=n_survive)
        # rand_int = np.array([np.argmax(probabilities)])
        infill_population = copy.deepcopy(offspring[rand_int])

        return infill_population

    def calculate_hypervolume_improvement(self, population, opt):
        # Extract objectives
        opt_obj = np.atleast_2d(opt.extract_obj())
        pop_obj = np.atleast_2d(population.extract_obj())

        # Determine reference point
        merged_obj_arr = np.vstack((opt_obj, pop_obj))
        nadir = np.max(merged_obj_arr, axis=0)
        nadir += 0.2 * np.abs(nadir)

        # Calculate HV of pareto front
        hv_pareto = calculate_hypervolume_pygmo(opt_obj, nadir)

        # Calculate hypervolume for each individual in population
        n_pop = len(population)
        hv_population = np.zeros(n_pop)
        for i in range(n_pop):
            # Merged optimum front with current individual
            merged_obj = np.vstack((copy.deepcopy(opt_obj), pop_obj[i]))

            # Calculate merged population HV
            hv_population[i] = calculate_hypervolume_pygmo(merged_obj, nadir)

        # Calculate hypervolume improvement
        hvi_arr = hv_population - hv_pareto

        return hvi_arr

    @staticmethod
    def predict_angle_diversity(population, comparison_population):
        obj_arr = np.atleast_2d(population.extract_obj())
        real_obj_arr = np.atleast_2d(comparison_population.extract_obj())

        # Find lower and upper bounds
        f_min = np.min(real_obj_arr, axis=0)
        f_max = np.max(real_obj_arr, axis=0)

        # Normalise objective
        obj_arr = (obj_arr - f_min) / (f_max - f_min)
        real_obj_arr = (real_obj_arr - f_min) / (f_max - f_min)

        # Scale by vector distances
        dist_to_ideal = np.linalg.norm(obj_arr, axis=1)
        dist_to_ideal[dist_to_ideal < 1e-64] = 1e-64
        obj_arr = obj_arr / dist_to_ideal[:, None]

        dist_to_ideal = np.linalg.norm(real_obj_arr, axis=1)
        dist_to_ideal[dist_to_ideal < 1e-64] = 1e-64
        real_obj_arr = real_obj_arr / dist_to_ideal[:, None]

        # Calculate Angle between offspring and population
        acute_angle = np.arccos(np.clip(obj_arr @ real_obj_arr.T, -1.0, 1.0))
        min_angle = np.min(acute_angle, axis=1)

        return min_angle

    def minimise_regressors(self, population, infill):
        # Evaluate with surrogate models
        population = self.surrogate_evaluator.do(self.surrogate.obj_func, self.problem, population)
        obj_arr = np.atleast_2d(population.extract_obj())

        # Update objective counter
        self.last_minimised += 1
        if self.last_minimised == self.problem.n_obj:
            self.last_minimised = 0

        # For the specified objective, find extremes of the population
        print(colored('last minimised obj: ', 'blue'), self.last_minimised)
        selected = np.argmin(obj_arr[:, self.last_minimised])
        new_infill = population[np.array([selected])]

        # Determine which offspring are too close to existing population
        is_duplicate = self.check_duplications(new_infill, self.population, epsilon=self.duplication_tolerance)
        new_infill = new_infill[np.invert(is_duplicate)]

        # Remove duplicate with previously selected infill point
        is_duplicate = self.check_duplications(new_infill, infill, epsilon=self.duplication_tolerance)
        new_infill = new_infill[np.invert(is_duplicate)]

        return new_infill

    def update_regressors(self, infill):
        # Ensure atleast one infill point is available
        if len(infill) == 0:
            print(colored('Did not find an infill point!', 'yellow'))
            return

        # Extract new training data
        new_vars = np.atleast_2d(infill.extract_var())
        new_obj = np.atleast_2d(infill.extract_obj())

        # Transform objectives and constraints with bilog
        bilog_new_obj = bilog_transform(copy.deepcopy(new_obj), beta=1)

        # Train objective and constraint models with new training data
        for obj_cntr in range(self.problem.n_obj):
            if self.transform_list[obj_cntr]:
                obj = bilog_new_obj[:, obj_cntr]
            else:
                obj = new_obj[:, obj_cntr]
            try:
                # Objective surrogates
                self.surrogate.obj_surrogates[obj_cntr].add_points(new_vars, obj)
                self.surrogate.obj_surrogates[obj_cntr].train()
            except np.linalg.LinAlgError:
                print(colored(f'Objective regressor {obj_cntr} update failed!', 'red'))

    def split_population_rbf_tuning(self):
        # Combine optimum front with last four infilled points
        test_pop = Population.merge(copy.deepcopy(self.reps), copy.deepcopy(self.infilled_pop[-4:]))

        # Remove duplicates
        is_duplicate = self.check_duplications(test_pop, other=None, epsilon=self.duplication_tolerance)
        test_pop = test_pop[np.invert(is_duplicate)]

        # Determine train_pop
        is_duplicate = self.check_duplications(self.population, other=test_pop, epsilon=self.duplication_tolerance)
        train_pop = self.population[np.invert(is_duplicate)]

        print(f"N pop: {len(self.population)}, N train: {len(train_pop)}, N test: {len(test_pop)}")
        return train_pop, test_pop

    def survival_mechanism(self, population):
        population = copy.deepcopy(population)

        # Conduct non-dominated sorting and select first front only from feasible individuals
        best_front = NonDominatedSorting().do(np.atleast_2d(population.extract_obj()),
                                              only_non_dominated_front=True)
        opt = population[best_front]

        if len(opt) > self.problem.n_var:
            reps = copy.deepcopy(opt)
        else:
            # Conduct rank and crowding to maintain a fixed set of representative solutions
            reps = self.rank_and_crowd.do(self.problem, population, n_survive=self.problem.n_var)

        return opt, reps

    def reps_survival_mechanism(self, population):
        population = copy.deepcopy(population)

        # Conduct rank and crowding to maintain a fixed set of representative solutions
        survived = self.rank_and_crowd.do(self.problem, population, n_survive=self.problem.n_var)

        return survived

    def performance_indicator_and_data_storage(self, population):
        # Calculate performance indicator
        self.igd = self.indicator.do(self.problem, self.opt, self.evaluator.n_eval, return_value=True)

        # Store fronts and population
        self.data_extractor.add_generation(population, self.n_gen)
        self.data_extractor.add_front(self.opt, self.n_gen, indicator_values=np.array(self.igd))  # constraints=self.opt.extract_cons()

    def check_duplications(self, pop, other, epsilon=1e-3):
        dist = self.calc_dist(pop, other)
        dist[np.isnan(dist)] = np.inf

        is_duplicate = np.any(dist < epsilon, axis=1)
        return is_duplicate

    @staticmethod
    def calc_dist(pop, other=None):
        pop_var = np.atleast_2d(pop.extract_var())

        if other is None:
            dist = distance.cdist(pop_var, pop_var)
            dist[np.triu_indices(len(pop_var))] = np.inf
        else:
            other_var = np.atleast_2d(other.extract_var())
            if pop_var.ndim == 1:
                pop_var = pop_var[None, :]
            if other_var.ndim == 1:
                other_var = other_var[None, :]
            dist = distance.cdist(pop_var, other_var)
        return dist

    @staticmethod
    def plot_pareto(ref_vec=None, obj_array1=None, obj_array2=None, obj_array3=None, scaling=1.5, labels=None, block=True):
        n_obj = len(ref_vec[0])

        if labels is None:
            labels = ['Exact', 'Pop', 'Feasible']

        # 2D Plot
        if n_obj == 2:
            fig, ax = plt.subplots(1, 1, figsize=(9, 7))
            plt.xlabel('Obj 1', fontsize=14)
            plt.ylabel('Obj 2', fontsize=14)

            # Plot reference vectors
            if ref_vec is not None:
                origin = np.zeros(len(ref_vec))
                x_vec = scaling * np.vstack((origin, ref_vec[:, 0])).T
                y_vec = scaling * np.vstack((origin, ref_vec[:, 1])).T
                for i in range(len(x_vec)):
                    if i == 0:
                        ax.plot(x_vec[i], y_vec[i], color='black', linewidth=0.5, label='Ref Vec')
                    else:
                        ax.plot(x_vec[i], y_vec[i], color='black', linewidth=0.5)

            if obj_array1 is not None:
                obj_array1 = np.atleast_2d(obj_array1)
                ax.scatter(obj_array1[:, 0], obj_array1[:, 1], color=line_colors[6], s=50, label=labels[0])

            if obj_array2 is not None:
                obj_array2 = np.atleast_2d(obj_array2)
                ax.scatter(obj_array2[:, 0], obj_array2[:, 1], color=line_colors[1], s=25, label=labels[1])

            if obj_array3 is not None:
                obj_array3 = np.atleast_2d(obj_array3)
                ax.scatter(obj_array3[:, 0], obj_array3[:, 1], color=line_colors[2], s=15, label=labels[2])

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
                ax.scatter3D(obj_array1[:, 0], obj_array1[:, 1], obj_array1[:, 2], color=line_colors[0], s=50,
                             label=labels[0])

            if obj_array2 is not None:
                obj_array2 = np.atleast_2d(obj_array2)
                ax.scatter3D(obj_array2[:, 0], obj_array2[:, 1], obj_array2[:, 2], color=line_colors[1], s=25,
                             label=labels[1])

            if obj_array3 is not None:
                obj_array3 = np.atleast_2d(obj_array3)
                ax.scatter3D(obj_array3[:, 0], obj_array3[:, 1], obj_array3[:, 2], color=line_colors[2], s=15,
                             label=labels[2])

        plt.legend(loc='best', frameon=False)
        if block:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(3)
            plt.close()
        # plt.savefig('/home/juan/PycharmProjects/optimisation_framework/multi_obj/results/zdt1_mmode_gen_' + str(len(self.surrogate.population)) + '.png')

# if self.feasible_frac > 0.0:
#     # Select feasible population only
#     cons_array = copy.deepcopy(self.population.extract_cons())
#     cons_array[cons_array <= 0.0] = 0.0
#     cons_sum = np.sum(cons_array, axis=1)
#     feasible_mask = cons_sum == 0.0
#     population = self.population[feasible_mask]
# else:
#     population = self.population