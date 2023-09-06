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

from optimisation.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from optimisation.operators.mutation.polynomial_mutation import PolynomialMutation
from optimisation.operators.mutation.de_mutation import DifferentialEvolutionMutation
from optimisation.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from optimisation.operators.selection.random_selection import RandomSelection
from optimisation.operators.survival.self_adaptive_feasible_ratio_epsilon_survival import SelfAdaptiveFeasibleRatioEpsilonSurvival
from optimisation.operators.survival.rank_and_crowding_survival import RankAndCrowdingSurvival
from optimisation.operators.survival.two_ranking_survival import TwoRankingSurvival
from optimisation.operators.survival.population_based_epsilon_survival import PopulationBasedEpsilonSurvival

from optimisation.surrogate.rbf_tuner import RBFTuner

from optimisation.metrics.indicator import Indicator
from optimisation.output.generation_extractor import GenerationExtractor
from optimisation.util.non_dominated_sorting import NonDominatedSorting
from optimisation.util.misc import min_prob_of_improvement

import matplotlib.pyplot as plt
plt.style.use('seaborn-talk')
np.set_printoptions(suppress=True)
line_colors = ['green', 'blue', 'red', 'orange', 'cyan', 'lawngreen', 'm', 'orangered', 'sienna', 'gold', 'violet', 'indigo', 'cornflowerblue']


class DEVIL(SAEvolutionaryAlgorithm):
    """
    Baseline Constrained SAEA
    """

    def __init__(self,
                 ref_dirs=None,
                 n_population=20,
                 surrogate=None,
                 sampling=LatinHypercubeSampling(criterion='maximin', iterations=10000),  # LHSLoader()
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
        self.nondom = NonDominatedSorting()
        self.rank_and_crowd = RankAndCrowdingSurvival(filter_infeasible=False)
        self.two_rank_survival = TwoRankingSurvival(filter_infeasible=False)
        # self.rank_and_crowd = TwoRankingSurvival(filter_infeasible=False)
        self.safr_survival = SelfAdaptiveFeasibleRatioEpsilonSurvival(filter_infeasible=False)  # TODO: not working
        # self.rank_and_crowd = PopulationBasedEpsilonSurvival(filter_infeasible=False)  # TODO: not working
        self.opt = self.survival_mechanism(self.population, filter_feasible=True)

        # RBF regression surrogate hyperparameter tuner
        self.rbf_tuning_frac = 0.20
        self.rbf_tuner = RBFTuner(n_dim=self.problem.n_var, lb=self.problem.x_lower, ub=self.problem.x_upper,
                                  c=0.5, p_type='linear', kernel_type='gaussian',
                                  width_range=(0.1, 10), train_test_split=self.rbf_tuning_frac,
                                  max_evals=30, verbose=False)
        self.hopt_frac = int(self.max_gen / 10)

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
        # self.repair = BounceBackBoundsRepair()
        self.infilled_pop = Population(self.problem)
        self.offspring_archive = Population(self.problem)

        # Feasible Fractions
        self.feasible_frac, self.individual_feasible_frac = self.feasible_fraction(self.population, return_ind_feas_frac=True)

        # Auxiliary population parameters
        self.eps0 = 0.15 * self.get_max_cons_viol(self.population)
        self.eps = None
        self.cp = 5.0

    def _next(self):
        # Alternative between survival mechanisms every generation
        self.last_selected += 1
        if self.last_selected == 3:
            self.last_selected = 0

        # Reduce the dynamic constraint boundary
        self.eps = self.reduce_boundary(self.eps0, self.n_gen, self.max_gen, self.cp)

        # Internal Offspring Generation
        self.offspring = self.generate_offspring(self.opt, n_offspring=self.n_offspring)

        # Probabilistic Infill Point Selection
        infill = self.probabilistic_infill_selection(self.offspring, n_survive=self.n_infill)

        # Expensive Evaluation of Infill Point
        infill = self.evaluator.do(self.problem.obj_func, self.problem, infill)

        # Update Archives
        self.population = Population.merge(self.population, infill)
        self.performance_indicator_and_data_storage(infill)
        self.opt = self.survival_mechanism(self.population, filter_feasible=True)

        # Hyperparameter tuning of RBF widths (for each constraint surrogate)
        if (self.n_gen % self.hopt_frac) == 0:
            print(colored('----------------- Hyper-tuning Regressor Surrogate Widths ------------------', 'green'))
            # Conduct optimisation of each objective surrogate
            opt_models, opt_params = self.rbf_tuner.do_cons(problem=self.problem,
                                                            population=self.population,
                                                            n_cons=len(self.surrogate.cons_surrogates))
            # Set newly optimised RBF models
            for cntr, model in enumerate(opt_models):
                self.surrogate.cons_surrogates[cntr].model = model
            print(colored('----------------------------------------------------------------------------', 'green'))

        # Update Feasible Fraction
        self.feasible_frac = self.feasible_fraction(self.population)

        # Surrogate Management
        self.update_regressors(infill)

        # Debugging Output
        print(colored(f"\t\t\t\tIGD: {self.igd:.3f}\tw_max: {self.w}\tObj: {infill.extract_obj()}\tCons: {infill.extract_cons()}\tFeas: {self.feasible_frac:.3f}", 'yellow'))

        # if self.evaluator.n_eval > (self.max_f_eval - 1):
        #     self.plot_pareto(ref_vec=self.ref_dirs, scaling=1.5, labels=['Pareto', 'Pop', 'Opt'],
        #                      obj_array1=self.problem.pareto_set.extract_obj(),
        #                      obj_array2=self.population.extract_obj(),
        #                      obj_array3=self.opt.extract_obj())

    def generate_offspring(self, representatives, n_offspring):
        # Select number of internal offspring generations
        self.pos = random.randint(0, len(self.w_max)-1)
        self.w = self.w_max[self.pos]

        # Select representative solutions to generate offspring from
        reps = representatives if len(representatives) > 3 else self.population
        survived = copy.deepcopy(reps)

        # Conduct offspring generations
        for gen in range(self.w):
            # Generate GA and DE offspring
            offspring = self.generate_ga_de_offspring(survived, n_offspring=n_offspring, evaluate=True)

            # # TODO: Inexpensive constraints ----------------------------------------------------------------------------
            # real_offs = self.surrogate_evaluator.do(self.problem.obj_func, self.problem, copy.deepcopy(offspring))
            # offspring.assign_cons(real_offs.extract_cons())
            # # TODO: Inexpensive constraints ----------------------------------------------------------------------------

            # Select most promising individuals to survive
            if self.last_selected == 0:
                # print('rcrowd')
                survived = self.rank_and_crowd.do(self.problem, offspring, n_survive=int(n_offspring / 5))
            elif self.last_selected == 1:
                # print('thor')
                survived = self.two_rank_survival.do(self.problem, offspring, n_survive=int(n_offspring / 5))
            elif self.last_selected == 2:
                # print('eps-safr')
                eps_population = self.transform_population(offspring)
                max_abs_con_vals = self.evaluator.calc_max_abs_cons(eps_population, self.problem)
                eps_population = self.evaluator.sum_normalised_cons(eps_population, self.problem, max_abs_con_vals=max_abs_con_vals)
                selected = self.safr_survival._do(self.problem, eps_population, int(n_offspring / 5), cons_val=eps_population.extract_cons_sum())
                survived = offspring[selected]

        # self.plot_pareto(ref_vec=self.ref_dirs, scaling=1.5, labels=['Pareto', 'Pop', 'Opt'],
        #                  obj_array1=self.problem.pareto_set.extract_obj(),
        #                  obj_array2=self.opt.extract_obj(),
        #                  obj_array3=survived.extract_obj())

        return survived

    def generate_ga_de_offspring(self, survived, n_offspring, evaluate=True):
        n_to_gen = min(len(survived),  int(n_offspring / 4))

        # Generate DE offspring (MODE)
        if n_to_gen > 4:
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
        offspring = Population.merge(offspring, survived)

        return offspring

    def probabilistic_infill_selection(self, offspring, n_survive=1):
        # Determine which offspring are too close to existing population
        is_duplicate = self.check_duplications(offspring, self.population, epsilon=self.duplication_tolerance)
        offspring = offspring[np.invert(is_duplicate)[0]]

        # Calculate angle diversity between offspring and the current best front
        comparison_population = Population.merge(copy.deepcopy(self.opt), copy.deepcopy(self.infilled_pop[-10:]))
        min_angle = self.predict_angle_diversity(offspring, comparison_population)

        # Calculate the minimum Probability of Improvement (MPoI)
        mpoi_metric = min_prob_of_improvement(offspring, comparison_population)

        # Combine metrics into scalar selection probability
        probabilities = mpoi_metric + 2 * min_angle
        probabilities -= np.min(probabilities)

        # Conduct probabilistic roulette-wheel infill selection
        rand_int = random.choices(population=list(range(len(offspring))), weights=probabilities, k=n_survive)
        infill_population = copy.deepcopy(offspring[rand_int])

        return infill_population

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
        acute_angle = np.arccos(obj_arr @ real_obj_arr.T)
        min_angle = np.min(acute_angle, axis=1)

        return min_angle

    def update_regressors(self, infill):
        # Ensure atleast one infill point is available
        if len(infill) == 0:
            print(colored('Did not find an infill point!', 'yellow'))
            return

        # Extract new training data
        new_vars = np.atleast_2d(infill.extract_var())
        new_obj = np.atleast_2d(infill.extract_obj())
        new_cons = np.atleast_2d(infill.extract_cons())

        # Train objective and constraint models with new training data
        for obj_cntr in range(self.problem.n_obj):
            try:
                # Objective surrogates
                self.surrogate.obj_surrogates[obj_cntr].add_points(new_vars, new_obj[:, obj_cntr])
                self.surrogate.obj_surrogates[obj_cntr].train()
            except np.linalg.LinAlgError:
                print(colored(f'Objective regressor {obj_cntr} update failed!', 'red'))

        # Update training data and train constraint models
        for cons_cntr in range(self.problem.n_con):
            try:
                # Constraint surrogates
                self.surrogate.cons_surrogates[cons_cntr].add_points(new_vars, new_cons[:, cons_cntr])
                self.surrogate.cons_surrogates[cons_cntr].train()
            except np.linalg.LinAlgError:
                print(colored(f'Constraint regressor {cons_cntr} update failed!', 'red'))

    def survival_mechanism(self, population, filter_feasible=True):
        population = copy.deepcopy(population)
        survived = None

        # Filter by constraint violation
        feasible_frac = 0.0
        if filter_feasible:
            # Select feasible population only
            cons_array = copy.deepcopy(population.extract_cons())
            cons_array[cons_array <= 0.0] = 0.0
            cons_sum = np.sum(cons_array, axis=1)
            feasible_mask = cons_sum == 0.0
            feasible_frac = np.count_nonzero(feasible_mask) / len(population)
            if feasible_frac > 0.0:
                # Conduct non-dominated sorting and select first front only from feasible individuals
                best_front = NonDominatedSorting().do(np.atleast_2d(population[feasible_mask].extract_obj()),
                                                      only_non_dominated_front=True)
                survived = population[feasible_mask][best_front]

        if not filter_feasible or feasible_frac == 0.0:
            # Conduct survival on population for next generation
            survived = self.survival.do(self.problem, population, self.n_population, gen=self.n_gen, max_gen=self.max_gen)

        return survived

    def performance_indicator_and_data_storage(self, population):
        # Calculate performance indicator
        self.igd = self.indicator.do(self.problem, self.opt, self.evaluator.n_eval, return_value=True)

        # Store fronts and population
        self.data_extractor.add_generation(population, self.n_gen)
        self.data_extractor.add_front(self.opt, self.n_gen, indicator_values=np.array(self.igd))  # constraints=self.opt.extract_cons()

    def transform_population(self, population):
        # Create new objectives and transformed constraint
        self.old_cons = population.extract_cons()
        eps_cons_arr = copy.deepcopy(self.old_cons) - self.eps
        eps_cons_arr[eps_cons_arr <= 0.0] = 0.0

        # Create new population
        transformed_pop = copy.deepcopy(population)
        transformed_pop.assign_cons(eps_cons_arr)

        return transformed_pop

    @staticmethod
    def get_max_cons_viol(population):
        cons_arr = copy.deepcopy(population.extract_cons())
        cons_arr[cons_arr <= 0.0] = 0.0
        max_cons_viol = np.amax(cons_arr, axis=0)
        max_cons_viol[max_cons_viol == 0] = 1.0

        return max_cons_viol

    @staticmethod
    def reduce_boundary(eps0, n_gen, max_gen, cp, delta=1e-8):
        A = eps0 + delta
        base_num = np.log((eps0 + delta) / delta)
        B = max_gen / np.power(base_num, 1 / cp) + 1e-15
        eps = A * np.exp(-(n_gen / B) ** cp) - delta
        eps[eps < 0.0] = 0.0

        return eps

    def check_duplications(self, pop, other, epsilon=1e-3):
        dist = self.calc_dist(pop, other)
        dist[np.isnan(dist)] = np.inf

        is_duplicate = [np.any(dist < epsilon, axis=1)]
        return is_duplicate

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
    def feasible_fraction(population, return_ind_feas_frac=False):
        n_total = len(population)
        cons_array = copy.deepcopy(population.extract_cons())
        cons_array[cons_array <= 0.0] = 0.0
        cons_sum = np.sum(cons_array, axis=1)
        feasible_mask = cons_sum == 0.0
        feasible_frac = np.count_nonzero(feasible_mask) / n_total

        if return_ind_feas_frac:
            individual_feasible_frac = np.count_nonzero(cons_array <= 0.0, axis=0) / n_total
            return feasible_frac, individual_feasible_frac

        return feasible_frac

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