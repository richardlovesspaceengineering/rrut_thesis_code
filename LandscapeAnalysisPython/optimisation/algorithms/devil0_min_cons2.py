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
from optimisation.operators.survival.multiple_constraints_ranking_survival import MultipleConstraintsRankingSurvival

from optimisation.metrics.indicator import Indicator
from optimisation.output.generation_extractor import GenerationExtractor
from optimisation.util.non_dominated_sorting import NonDominatedSorting
from optimisation.util.misc import min_prob_of_improvement
from optimisation.util.misc import bilog_transform

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
                 # sampling=LatinHypercubeSampling(criterion='maximin', iterations=10000),
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
        self.last_minimised = -1
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
        # self.rank_and_crowd = TwoRankingSurvival(filter_infeasible=False)
        # self.rank_and_crowd = SelfAdaptiveFeasibleRatioEpsilonSurvival(filter_infeasible=False)
        # self.rank_and_crowd = PopulationBasedEpsilonSurvival(filter_infeasible=False)  # TODO: not working
        # self.rank_and_crowd = MultipleConstraintsRankingSurvival(filter_infeasible=False, use_generalised_mcr=True)
        self.opt = self.survival_mechanism(self.population, filter_feasible=True)

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

        # Feasible Fraction and active constraints
        self.feasible_frac, self.individual_feasible_frac = self.feasible_fraction(self.population, return_ind_feas_frac=True)
        self.n_active_cons = np.zeros(self.problem.n_con)
        self.n_con_original = self.problem.n_con
        self.n_obj_original = self.problem.n_obj

        # TODO
        self.active_cons = np.array([6])
        self.inactive_cons = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9])

        # TODO: Bilog Debug (retrain surrogates with bilog objectives)
        bilog_cons_arr = bilog_transform(self.population.extract_cons(), beta=1)
        for cons_cntr, model in enumerate(self.surrogate.cons_surrogates):
            model.reset()
            model.add_points(self.population.extract_var(), bilog_cons_arr[:, cons_cntr].flatten())
            model.train()

    def _next(self):
        # # TODO
        # if self.n_gen % 50 == 0:
        #     self.active_cons = np.argpartition(-self.n_active_cons, 2)[:2]
        #     self.inactive_cons = np.where(~np.in1d(np.arange(self.problem.n_con), self.active_cons))[0]
        #     print(f"Using act: {self.active_cons}, inact: {self.inactive_cons}")

        # Internal Offspring Generation
        self.offspring = self.generate_offspring(self.opt, n_offspring=self.n_offspring)

        # Probabilistic Infill Point Selection
        infill = self.probabilistic_infill_selection(self.offspring, n_survive=self.n_infill)

        # Expensive Evaluation of Infill Point
        infill = self.evaluator.do(self.problem.obj_func, self.problem, infill)
        
        # Calculate which constraints are active
        infill_cons = infill.extract_cons()
        active_mask = infill_cons > 0.0
        self.n_active_cons[active_mask] += 1.0
        print(f"Active cons: {self.n_active_cons}")

        # Update Archives
        self.population = Population.merge(self.population, infill)
        self.opt = self.survival_mechanism(self.population, filter_feasible=True)
        self.performance_indicator_and_data_storage(infill)

        # Update Feasible Fraction
        self.feasible_frac = self.feasible_fraction(self.population)

        # Surrogate Management
        self.update_regressors(infill)

        # Debugging Output
        cv_sum = self.calc_cv(self.population)
        cv_infill = self.calc_cv(infill)
        print(colored(f"min CV: {np.min(cv_sum):.3f} IGD: {self.igd:.3f} Obj: {infill.extract_obj()} Cons: {infill.extract_cons()} Feas: {self.feasible_frac:.3f} CV: {cv_infill}", 'yellow'))

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
            
            # # Create new population with n_obj active constraints as objectives and inactive as constraints
            # active_cons = np.argpartition(-self.n_active_cons, self.problem.n_obj)[:self.problem.n_obj]
            # inactive_cons = np.where(~np.in1d(np.arange(self.problem.n_con), active_cons))[0]
            # n_active_cons = len(active_cons)
            # n_inactive_cons = self.problem.n_con - n_active_cons
            # self.problem.n_con = n_inactive_cons
            # cons_arr = offspring.extract_cons()
            # mod_offspring = Population(self.problem, len(offspring))
            # mod_offspring.assign_var(self.problem, offspring.extract_var())
            # mod_offspring.assign_obj(cons_arr[:, active_cons])
            # mod_offspring.assign_cons(cons_arr[:, inactive_cons])

            # # TODO: minimise constraint selection (lowest CV so far)
            to_keep = np.array([4, 6, 7, 8, 9])
            mod_offspring = copy.deepcopy(offspring)
            cons_arr = mod_offspring.extract_cons()[:, to_keep]
            mod_offspring.assign_cons(cons_arr)
            cons_arr[cons_arr <= 0.0] = 0.0
            cv_sum = np.sum(cons_arr, axis=1)
            # cv_sum = np.max(cons_arr, axis=1)
            self.problem.n_con = len(to_keep)
            selected = self.rank_and_crowd._do(self.problem, mod_offspring, n_survive=int(n_offspring / 5), cons_val=cv_sum)
            survived = offspring[selected]
            self.problem.n_con = self.n_con_original

            # TODO: Minimise most act. cons vs. rest of cons CV sum (unconstrained, ignoring objectives)
            # self.active_cons = np.array([4, 6])
            # self.inactive_cons = np.array([0, 1, 2, 3, 5, 7, 8, 9])
            # cons_arr = offspring.extract_cons()
            # obj_arr = cons_arr[:, self.active_cons]
            # cons_arr = copy.deepcopy(cons_arr[:, self.inactive_cons])
            # cons_arr[cons_arr <= 0.0] = 0.0
            # cv_sum = np.sum(cons_arr, axis=1)
            # self.problem.n_con = len(self.inactive_cons)
            # mod_offspring = copy.deepcopy(offspring)
            # mod_offspring.assign_obj(obj_arr)
            # selected = self.rank_and_crowd._do(self.problem, mod_offspring, n_survive=int(n_offspring / 5), cons_val=cv_sum)
            # survived = offspring[selected]
            # self.problem.n_con = self.n_con_original

            # # TODO: All constraints as objectives
            # obj_arr = offspring.extract_cons()
            # obj_arr[obj_arr <= 0.0] = 0.0
            # best_front = self.nondom.do(obj_arr, only_non_dominated_front=True, return_rank=False)
            # survived = offspring[best_front]

            # TODO: CV vs. N_viol
            # cons_arr = copy.deepcopy(offspring.extract_cons())
            # cons_arr[cons_arr <= 0.0] = 0.0
            # cv_sum = np.sum(cons_arr, axis=1)
            # n_viol = np.max(cons_arr, axis=1)
            # obj_arr = np.hstack((cv_sum[:, None], n_viol[:, None]))
            # best_front = self.nondom.do(obj_arr, only_non_dominated_front=True, return_rank=False)
            # survived = offspring[best_front]

            # TODO: Min CV
            # cons_arr = copy.deepcopy(offspring.extract_cons())
            # cons_arr[cons_arr <= 0.0] = 0.0
            # cv_sum = np.sum(cons_arr, axis=1)
            # selected = np.argpartition(cv_sum, int(n_offspring / 5))[:int(n_offspring / 5)]
            # survived = offspring[selected]

            # self.plot_pareto(ref_vec=self.ref_dirs, scaling=1.5, labels=['Pareto', 'Pop', 'Opt'],
            #                  obj_array1=obj_arr)

            # Select most promising individuals to survive (using active constraints only)
            # survived = self.rank_and_crowd.do(self.problem, offspring, n_survive=int(n_offspring / 5))
            # self.problem.n_con = self.n_con_original

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
        # offspring = Population.merge(offspring, survived)

        return offspring

    def probabilistic_infill_selection(self, offspring, n_survive=1):
        # Determine which offspring are too close to existing population
        is_duplicate = self.check_duplications(offspring, self.population, epsilon=self.duplication_tolerance)
        offspring = offspring[np.invert(is_duplicate)[0]]

        # # Calculate angle diversity between offspring and the current best front
        # comparison_population = Population.merge(copy.deepcopy(self.opt), copy.deepcopy(self.infilled_pop[-10:]))
        # min_angle = self.predict_angle_diversity(offspring, comparison_population)
        #
        # # Calculate the minimum Probability of Improvement (MPoI)
        # mpoi_metric = min_prob_of_improvement(offspring, comparison_population)
        #
        # # Combine metrics into scalar selection probability
        # probabilities = mpoi_metric + 2 * min_angle
        # probabilities -= np.min(probabilities)
        #
        # # Conduct probabilistic roulette-wheel infill selection
        # rand_int = random.choices(population=list(range(len(offspring))), weights=probabilities, k=n_survive)
        # # rand_int = np.random.randint(0, len(offspring), 1)

        # Minimise CV
        cons_arr = np.atleast_2d(copy.deepcopy(offspring.extract_cons()))
        cons_arr[cons_arr <= 0.0] = 0.0
        cv_sum = np.sum(cons_arr, axis=1)
        # cv_sum = np.max(cons_arr, axis=1)
        rand_int = np.array([np.argmin(cv_sum)])

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

        # TODO: Bilog Debug (retrain surrogates with bilog constraints)
        bilog_new_cons = bilog_transform(new_cons, beta=1)

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
                self.surrogate.cons_surrogates[cons_cntr].add_points(new_vars, bilog_new_cons[:, cons_cntr])
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
                best_front = self.nondom.do(np.atleast_2d(population[feasible_mask].extract_obj()),
                                            only_non_dominated_front=True, return_rank=False)
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

    def check_duplications(self, pop, other, epsilon=1e-3):
        dist = self.calc_dist(pop, other)
        dist[np.isnan(dist)] = np.inf

        is_duplicate = [np.any(dist < epsilon, axis=1)]
        return is_duplicate

    @staticmethod
    def calc_cv(population):
        cons_arr = np.atleast_2d(copy.deepcopy(population.extract_cons()))
        cons_arr[cons_arr <= 0.0] = 0.0
        cv = np.sum(cons_arr, axis=1)

        return cv

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