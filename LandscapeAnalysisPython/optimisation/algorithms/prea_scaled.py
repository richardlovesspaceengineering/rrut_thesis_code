import copy
import random
import sys

from termcolor import colored

import numpy as np
from scipy.spatial import distance

from optimisation.model.population import Population
from optimisation.algorithms.sa_evolutionary_algorithm import SAEvolutionaryAlgorithm

from optimisation.operators.sampling.lhs_loader import LHSLoader
from optimisation.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from optimisation.operators.selection.random_selection import RandomSelection
from optimisation.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from optimisation.operators.mutation.de_mutation import DifferentialEvolutionMutation
from optimisation.operators.mutation.polynomial_mutation import PolynomialMutation
from optimisation.operators.survival.rank_and_crowding_survival import RankAndCrowdingSurvival
from optimisation.model.duplicate import DefaultDuplicateElimination
from optimisation.model.repair import BasicBoundsRepair, BounceBackBoundsRepair
from optimisation.model.evaluator import Evaluator
from optimisation.surrogate.rbf_tuner import RBFTuner
from optimisation.metrics.indicator import Indicator
from optimisation.output.generation_extractor import GenerationExtractor
from optimisation.util.non_dominated_sorting import NonDominatedSorting

# from optimisation.util.dominator import calculate_domination_matrix
from optimisation.util.misc import bilog_transform, min_prob_of_improvement, sp2log_transform


import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('seaborn-talk')
np.set_printoptions(suppress=True)
# matplotlib.use('TkAgg')
line_colors = ['green', 'blue', 'red', 'orange', 'cyan', 'lawngreen', 'm', 'orangered', 'sienna', 'gold', 'violet',
               'indigo', 'cornflowerblue']


class PREA(SAEvolutionaryAlgorithm):
    """
    PREA: An efficient, Probabilistic Regression Evolutionary Algorithm for expensive unconstrained multi-objective optimisation problems
    """
    def __init__(self,
                 ref_dirs=None,
                 n_population=100,
                 surrogate=None,
                 sampling=LHSLoader(),  # LatinHypercubeSampling(),
                 selection=RandomSelection(),
                 crossover=SimulatedBinaryCrossover(eta=20, prob=1.0),
                 mutation=PolynomialMutation(eta=20, prob=None),
                 eliminate_duplicates=DefaultDuplicateElimination(epsilon=1e-4),
                 **kwargs):

        self.ref_dirs = ref_dirs

        # Have to define here given the need to pass ref_dirs
        if 'survival' in kwargs:
            survival = kwargs['survival']
            del kwargs['survival']
        else:
            survival = RankAndCrowdingSurvival(filter_infeasible=True)

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

        # PREA Parameters
        self.last_minimised = -1
        self.n_init = self.evaluator.n_eval
        self.max_infill = self.max_gen - self.n_init
        self.n_infill = 1
        self.n_offspring = 300
        # self.w_max = self.w_max_0  # Initial number of internal offspring iterations
        self.w_max = [5, 10, 15, 20]
        self.w = 20

        # Local surrogates
        self.n_fixed_pop = 10 * self.problem.n_var
        self.training_pop = copy.deepcopy(self.population)
        self.f_min = None
        self.f_max = None

        # RBF regression surrogate hyperparameter tuner
        self.rbf_tuning_frac = 0.20
        self.rbf_tuner = RBFTuner(n_dim=self.problem.n_var, lb=self.problem.x_lower, ub=self.problem.x_upper,
                                  c=0.5, p_type='linear', kernel_type='gaussian',
                                  width_range=(0.1, 10), train_test_split=self.rbf_tuning_frac,
                                  max_evals=50, verbose=False)

        # Minimisation Exploitation of RBF surrogates
        self.minimise_frac = int(self.max_gen / self.problem.n_var)

        # Evaluator for predicted objective values
        self.surrogate_evaluator = Evaluator()
        self.surrogate_strategy = self.surrogate
        self.rank_and_crowd = RankAndCrowdingSurvival()

        # Data extractor
        self.filename = f"{self.problem.name.lower()}_{self.save_name}_maxgen_{round(self.max_f_eval)}_sampling_" \
                        f"{self.surrogate_strategy.n_training_pts}_seed_{self.surrogate_strategy.sampling_seed}"
        self.data_extractor = GenerationExtractor(filename=self.filename, base_path=self.output_path)
        self.indicator = Indicator(metric='igd')
        self.performance_indicator_and_data_storage(self.population)

        # Differential Evolution
        self.de_mutation = DifferentialEvolutionMutation(problem=self.problem, method='diversity')
        self.repair = BasicBoundsRepair()
        self.infilled_pop = Population(self.problem)

        # Re-initialise surrogates with scaled objective values
        self.initialise_regressor()
        print(self.f_min, self.f_max)

    def _next(self):
        # Conduct mating using the current population
        self.w = self.w_max[random.randint(0, len(self.w_max)-1)]
        self.offspring = self.generate_offspring(self.opt, n_gen=self.w, n_offspring=self.n_offspring)

        # Conduct probabilistic infill selection
        infill_population = self.probabilistic_infill_selection(self.offspring, self.opt, n_survive=self.n_infill)

        # Find the predicted surrogate extremes for the selected objective
        if (self.evaluator.n_eval % self.minimise_frac) == 0:
            # Minimisation of surrogate predictions
            extra_infill = self.minimise_regressors(self.offspring)
            infill_population = Population.merge(infill_population, extra_infill)
            print(colored('----------------- Minimising Regressor Surrogate Predictions ------------------', 'green'))

        # Evaluate infill point expensively
        infill_population = self.evaluator.do(self.problem.obj_func, self.problem, infill_population)
        print(colored(f"                                       w_max = {self.w}    New Obj: {infill_population.extract_obj()}", 'yellow'))

        # Merge the offspring with the current population
        # old_population = copy.deepcopy(self.population)
        self.population = Population.merge(self.population, infill_population)
        self.infilled_pop = Population.merge(self.infilled_pop, infill_population)

        # Update optimum (non-dominated solutions)
        # old_opt = copy.deepcopy(self.opt)
        fronts = NonDominatedSorting().do(self.population.extract_obj(), return_rank=False)
        self.opt = copy.deepcopy(self.population[fronts[0]])

        # Hyperparameter tuning of RBF widths (for each objective surrogate)
        if (self.n_gen % self.minimise_frac) == 0:
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

        # Calculate Performance Indicator and store generational data
        self.performance_indicator_and_data_storage(infill_population)

        # Update regressors with new infill point
        self.update_regressor(infill_population)

        # if (self.evaluator.n_eval % 50) == 0:
        #     self.plot_pareto(ref_vec=self.ref_dirs, scaling=1.5, labels=['Pareto', 'Total Pop', 'Local Pop'],
        #                      obj_array1=self.problem.pareto_set.extract_obj(),
        #                      obj_array2=self.population.extract_obj(),
        #                      obj_array3=self.training_pop.extract_obj())
        #
        # TODO: DEBUG PLOT ---------------------------------------------------------------------------------------------
        if self.evaluator.n_eval > (self.max_f_eval - 1):
            self.plot_pareto(ref_vec=self.ref_dirs, scaling=1.5, labels=['Pareto', 'Pop', 'Opt'],
                             obj_array1=self.problem.pareto_set.extract_obj(),
                             obj_array2=self.population.extract_obj(),
                             obj_array3=self.opt.extract_obj())

    def performance_indicator_and_data_storage(self, population):
        # Calculate performance indicator
        igd = self.indicator.do(self.problem, self.opt, self.evaluator.n_eval, return_value=True)
        print('IGD: ', np.round(igd, 4))

        # Store fronts and population
        self.data_extractor.add_generation(population, self.n_gen)
        self.data_extractor.add_front(self.opt, self.n_gen, indicator_values=np.array(igd))

    def initialise_regressor(self):
        # Initialise regressors
        obj_surrogates = self.surrogate.obj_surrogates

        # Create initial training data
        x_vars = self.population.extract_var()
        obj_array = self.population.extract_obj()

        # Determine objective bounds only once after the LHS
        self.f_min = np.min(obj_array, axis=0)
        self.f_max = np.max(obj_array, axis=0)

        # Normalise objective values
        # obj_array = (2 * obj_array - (self.f_max + self.f_min)) / (self.f_max - self.f_min)
        # print(np.min(obj_array), np.max(obj_array))

        # Re-initialise training data and fit model
        for obj_cntr in range(self.problem.n_obj):
            obj_surrogates[obj_cntr].reset()
            obj_surrogates[obj_cntr].add_points(x_vars, obj_array[:, obj_cntr])
            obj_surrogates[obj_cntr].train()

        return obj_surrogates

    def update_regressor(self, infill):
        # Ensure atleast one infill point is available
        if len(infill) == 0:
            print(colored('No infill point!, skipping surrogate update', 'yellow'))
            return

        # Extract update training data
        new_vars = np.atleast_2d(infill.extract_var())
        new_obj = np.atleast_2d(infill.extract_obj())

        # Normalise new objective values using initial objective ranges
        # new_obj = (2 * new_obj - (self.f_max + self.f_min)) / (self.f_max - self.f_min)
        # print(np.min(new_obj), np.max(new_obj))

        try:
            # Update training data and train objective models
            for obj_cntr in range(self.problem.n_obj):
                # Objective surrogates
                self.surrogate.obj_surrogates[obj_cntr].add_points(new_vars, new_obj[:, obj_cntr])
                self.surrogate.obj_surrogates[obj_cntr].train()
        except np.linalg.LinAlgError:
            print(colored(f'Objective regressor update failed! (obj_idx: {obj_cntr})', 'red'))

    def minimise_regressors(self, population):
        # Evaluate with surrogate models
        population = self.surrogate_evaluator.do(self.surrogate.obj_func, self.problem, population)
        obj_arr = np.atleast_2d(population.extract_obj())

        # Update objective counter
        self.last_minimised += 1
        if self.last_minimised == self.problem.n_obj:
            self.last_minimised = 0

        # For the specified objective, find extremes of the population
        print(colored('last minimised obj: ', 'blue'), self.last_minimised)
        idx = np.argmin(obj_arr[:, self.last_minimised])
        selected = [idx]

        # Filter out duplicates
        infill = population[np.array(selected)]
        infill = self.eliminate_duplicates.do(infill, self.population)

        return infill

    def generate_offspring(self, representatives, n_offspring=300, n_gen=10):
        representatives = representatives if len(representatives) > 3 else self.population
        survived = copy.deepcopy(representatives)
        for gen in range(n_gen-1):
            offspring = self.generate_ga_de_offspring(survived, n_offspring=n_offspring, evaluate=True)
            offspring = Population.merge(offspring, survived)

            # Select most promising individuals to survive
            survived = self.rank_and_crowd.do(self.problem, offspring, n_survive=int(n_offspring / 5))
            # best_front = NonDominatedSorting().do(offspring.extract_obj(), only_non_dominated_front=True, return_rank=False)
            # survived = offspring[best_front]

        # Create offspring of final n_offspring size
        offspring = self.generate_ga_de_offspring(survived, n_offspring=n_offspring, evaluate=True)

        # TODO: DEBUG --------------------------------------------------------------------
        # survived = self.surrogate_evaluator.do(self.problem.obj_func, self.problem, survived)
        # self.plot_pareto(ref_vec=self.ref_dirs, scaling=1, labels=['Survived', 'Reps', 'Pop'],
        #                  obj_array3=self.population.extract_obj(),
        #                  obj_array2=self.opt.extract_obj(),
        #                  obj_array1=survived.extract_obj())
        # TODO: DEBUG ---------------------------------------------------------------------
        return offspring

    def generate_ga_de_offspring(self, survived, n_offspring, evaluate=True):
        n_to_gen = min(len(survived),  int(n_offspring / 4))

        # Generate DE offspring (MODE)
        if n_to_gen > 3:
            archive = self.opt if len(self.opt) > 3 else self.population
            offspring1 = self.de_mutation.do(survived, archive, n_to_gen)
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

        return offspring

    def probabilistic_infill_selection(self, offspring, optimum, n_survive=1):
        # Determine which offspring are too close to existing population
        is_duplicate = self.check_duplications(offspring, self.population, epsilon=self.duplication_tolerance)
        offspring = offspring[np.invert(is_duplicate)[0]]

        # Calculate angle diversity between offspring and the current best front
        comparison_population = Population.merge(copy.deepcopy(self.opt), copy.deepcopy(self.infilled_pop[-10:]))
        min_angle = self.predict_angle_diversity(offspring, comparison_population)

        # Calculate the minimum Probability of Improvement (MPoI)
        mpoi_metric = min_prob_of_improvement(offspring, comparison_population)

        # # Calculate the EDN of each offspring individual
        # obj_arr = offspring.extract_obj()
        # dom_mat = calculate_domination_matrix(obj_arr)
        # dom_mat[dom_mat != 1] = 0
        # dom_ranks = np.sum(dom_mat, axis=1)
        # dom_ranks = dom_ranks / np.max(dom_ranks)

        # Combine metrics into scalar selection probability
        # probabilities = min_angle
        probabilities = mpoi_metric + 2 * min_angle
        # probabilities = dom_ranks + mpoi_metric + 2 * min_angle
        probabilities = probabilities - np.min(probabilities)
        # print(np.min(probabilities), np.max(probabilities))

        # Conduct probabilistic roulette-wheel infill selection
        rand_int = random.choices(population=list(range(len(offspring))), weights=probabilities, k=n_survive)
        # rand_int = np.array([np.argmax(min_angle)])
        infill_population = copy.deepcopy(offspring[rand_int])

        # # TODO: Plotting debug -----------------------------------------------------------------------------------------
        # if self.evaluator.n_eval > 248:
        #     offs_obj = offspring.extract_obj()
        #     reps_obj = np.atleast_2d(self.opt.extract_obj())
        #     fig, ax = plt.subplots(figsize=(9, 7))
        #     ax_contour = ax.tricontourf(offs_obj[:, 0], offs_obj[:, 1], probabilities, vmin=probabilities.min(), vmax=probabilities.max())
        #     ax.scatter(reps_obj[:, 0], reps_obj[:, 1], c='k', s=10)
        #     ax.scatter(offs_obj[:, 0], offs_obj[:, 1], c='r', s=10)
        #     plt.colorbar(ax_contour)
        #     plt.show()
        # # TODO: Plotting debug -----------------------------------------------------------------------------------------

        return infill_population

    def predict_angle_diversity(self, population, comparison_population):
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
        acute_angle = np.arccos(obj_arr @ real_obj_arr.T)
        min_angle = np.min(acute_angle, axis=1)

        return min_angle

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

            # ax.set_xlim((-0.05, 1.05))

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
        # plt.savefig('/home/juan/PycharmProjects/optimisation_framework/multi_obj/results/zdt1_mmode_gen_' + str(len(self.surrogate.population)) + '.png')

