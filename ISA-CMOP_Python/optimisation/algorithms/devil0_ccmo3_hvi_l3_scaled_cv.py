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

from optimisation.metrics.indicator import Indicator
from optimisation.output.generation_extractor import GenerationExtractor
from optimisation.util.non_dominated_sorting import NonDominatedSorting
from optimisation.util.calculate_hypervolume import calculate_hypervolume_pygmo

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
        self.two_rank_survival = TwoRankingSurvival(filter_infeasible=False)
        self.safr_survival = SelfAdaptiveFeasibleRatioEpsilonSurvival(filter_infeasible=False)
        self.opt = self.survival_mechanism(self.population)
        self.reps = self.reps_survival_mechanism(self.population)
        self.survival_list = ['rcrowd', 'tor', 'safr']

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
        self.max_cons_viol = self.get_max_cons_viol(self.population)
        self.eps0 = 0.15 * self.max_cons_viol
        self.eps = None
        self.cp = 5.0

        # TODO: Evaluate Global surrogate model accuracy
        # lhs = np.genfromtxt('/home/juan/PycharmProjects/optimisation_framework/multi_obj/1000.lhs')
        # self.large_lhs = (self.problem.x_upper - self.problem.x_lower) * lhs + self.problem.x_lower

        # TODO: Evaluate Local surrogate model accuracy (pareto front)
        self.large_lhs = np.genfromtxt('/home/juan/PycharmProjects/optimisation_framework/multi_obj/cases/MODAct_files/cs1_vars.txt')

        self.test_pop = Population(self.problem, len(self.large_lhs))
        self.test_pop.assign_var(self.problem, self.large_lhs)
        self.test_pop = self.surrogate_evaluator.do(self.problem.obj_func, self.problem, self.test_pop)

        self.surro_pop = self.surrogate_evaluator.do(self.surrogate.obj_func, self.problem, copy.deepcopy(self.test_pop))
        obj_rmse = (1 / len(self.large_lhs)) * np.sum(np.power(self.test_pop.extract_obj() - self.surro_pop.extract_obj(), 2), axis=0)
        cons_rmse = (1 / len(self.large_lhs)) * np.sum(np.power(self.test_pop.extract_cons() - self.surro_pop.extract_cons(), 2), axis=0)

        self.rmse_evals = np.array([self.evaluator.n_eval])
        self.rmse_obj = obj_rmse
        self.rmse_cons = cons_rmse
        print(f"OBJ RMSE: {obj_rmse}, CONS RMSE: {cons_rmse}")

    def _next(self):
        # Reduce the dynamic constraint boundary
        self.eps = self.reduce_boundary(self.eps0, self.n_gen, self.max_gen, self.cp)

        # Internal Offspring Generation
        self.offspring = self.generate_offspring(self.reps, n_offspring=self.n_offspring)

        # # TODO: DEBUG PLOT
        if (self.evaluator.n_eval-1) % 25 == 0:
            # self.plot_subplots(self.population, self.offspring, self.problem.pareto_set)
            self.surro_pop = self.surrogate_evaluator.do(self.surrogate.obj_func, self.problem, copy.deepcopy(self.test_pop))
            obj_rmse = (1 / len(self.large_lhs)) * np.sum(np.power(self.test_pop.extract_obj() - self.surro_pop.extract_obj(), 2), axis=0)
            cons_rmse = (1 / len(self.large_lhs)) * np.sum(np.power(self.test_pop.extract_cons() - self.surro_pop.extract_cons(), 2), axis=0)
            print(f"OBJ RMSE: {obj_rmse}, CONS RMSE: {cons_rmse}")

            self.rmse_evals = np.hstack((self.rmse_evals, self.evaluator.n_eval))
            self.rmse_obj = np.vstack((self.rmse_obj, obj_rmse))
            self.rmse_cons = np.vstack((self.rmse_cons, cons_rmse))

        # Probabilistic Infill Point Selection
        infill = self.probabilistic_infill_selection(self.offspring, n_survive=self.n_infill)

        # Expensive Evaluation of Infill Point
        infill = self.evaluator.do(self.problem.obj_func, self.problem, infill)

        # Update Archives
        self.population = Population.merge(self.population, infill)
        self.opt = self.survival_mechanism(self.population)
        self.reps = self.reps_survival_mechanism(self.population)
        self.performance_indicator_and_data_storage(infill)
        print(f"N opt: {len(self.opt)}, N reps: {len(self.reps)}")

        # Update Feasible Fraction
        self.feasible_frac = self.feasible_fraction(self.population)

        # Surrogate Management
        self.update_regressors(infill)

        # Debugging Output
        cv_sum = self.calc_cv(self.population)
        cv_infill = self.calc_cv(infill)
        print(colored(f"min CV: {np.min(cv_sum):.3f} IGD: {self.igd:.3f} Obj: {infill.extract_obj()} Cons: {infill.extract_cons()} Feas: {self.feasible_frac:.3f} CV: {cv_infill}", 'yellow'))

        if self.evaluator.n_eval > (self.max_f_eval - 1):
            self.plot_pareto(ref_vec=self.ref_dirs, scaling=1.5, labels=['Pareto', 'Pop', 'Opt'],
                             obj_array1=self.problem.pareto_set.extract_obj(),
                             obj_array2=self.population.extract_obj(),
                             obj_array3=self.opt.extract_obj())

            fig, ax = plt.subplots(1, 1, figsize=(9, 7))
            plt.xlabel('Obj 1', fontsize=14)
            plt.ylabel('Obj 2', fontsize=14)
            for i in range(self.problem.n_obj):
                ax.plot(self.rmse_evals, self.rmse_obj[:, i] / self.rmse_obj[0, i], '-o', label=f'Obj {i+1}')
            for j in range(self.problem.n_con):
                ax.plot(self.rmse_evals, self.rmse_cons[:, j] / self.rmse_cons[0, j], '-o', label=f'Cons {j+1}')
            plt.legend()
            plt.show()

    def generate_offspring(self, representatives, n_offspring):
        # Select number of internal offspring generations
        self.pos += 1
        if self.pos == len(self.w_max):
            self.pos = 0
        self.w = self.w_max[self.pos]

        # Alternative between survival mechanisms every generation
        self.last_selected += 1
        if self.last_selected == 3:
            self.last_selected = 0

        # Select representative solutions to generate offspring from
        survived = copy.deepcopy(representatives)

        # Conduct offspring generations
        print(f"Survival: {self.survival_list[self.last_selected]}, w_max: {self.w}")
        for gen in range(self.w):
            # Generate GA and DE offspring
            offspring = self.generate_ga_de_offspring(survived, n_offspring=n_offspring, evaluate=True)
            if gen < (self.w - 1):
                offspring = Population.merge(offspring, survived)

            # Select most promising individuals to survive
            cons_val = self.calc_cv(offspring, normalise=self.max_cons_viol)
            if self.last_selected == 0:
                selected = self.rank_and_crowd._do(self.problem, offspring, n_survive=int(n_offspring / 5), cons_val=cons_val)
            elif self.last_selected == 1:
                selected = self.two_rank_survival._do(self.problem, offspring, n_survive=int(n_offspring / 5), cons_val=cons_val)
            else:
                eps_offspring = self.transform_population(offspring)
                cons_val = self.calc_cv(eps_offspring, normalise=self.max_cons_viol)
                selected = self.safr_survival._do(self.problem, eps_offspring, int(n_offspring / 5),
                                                  cons_val=cons_val, gen=self.n_gen, max_gen=self.max_gen)
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
        offspring = offspring[np.invert(is_duplicate)[0]]

        # Calculate angle diversity between offspring and the current best front
        comparison_population = Population.merge(copy.deepcopy(self.reps), copy.deepcopy(self.infilled_pop[-4:]))
        # comparison_population = self.reps
        min_angle = self.predict_angle_diversity(offspring, comparison_population)

        # Calculate Hypervolume Improvement of each offspring
        hvi = self.calculate_hypervolume_improvement(offspring, self.reps)
        hvi = (hvi - np.min(hvi)) / (np.max(hvi) - np.min(hvi))
        probabilities = 2 * hvi + min_angle

        # Conduct probabilistic roulette-wheel infill selection
        # rand_int = random.choices(population=list(range(len(offspring))), weights=probabilities, k=n_survive)
        rand_int = np.array([np.argmax(probabilities)])
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

    def survival_mechanism(self, population):
        population = copy.deepcopy(population)
        survived = None

        # Select feasible population only
        cv_sum = self.calc_cv(population)
        feasible_mask = cv_sum == 0.0
        feasible_frac = np.count_nonzero(feasible_mask) / len(population)
        if feasible_frac > 0.0:
            # Conduct non-dominated sorting and select first front only from feasible individuals
            best_front = NonDominatedSorting().do(np.atleast_2d(population[feasible_mask].extract_obj()),
                                                  only_non_dominated_front=True)
            survived = population[feasible_mask][best_front]

        if feasible_frac == 0.0:
            # Select point with lowest cv violation
            lowest_cv = np.array([np.argmin(cv_sum)])
            survived = population[lowest_cv]

        return survived

    def reps_survival_mechanism(self, population):
        population = copy.deepcopy(population)
        survived = Population(self.problem)

        # Select feasible population only
        cv_sum = self.calc_cv(population)
        feasible_mask = cv_sum == 0.0
        feasible_frac = np.count_nonzero(feasible_mask) / len(population)

        if feasible_frac > 0.0:
            # Conduct non-dominated sorting and select first front only from feasible individuals
            best_front = NonDominatedSorting().do(np.atleast_2d(population[feasible_mask].extract_obj()),
                                                  only_non_dominated_front=True)
            candidates = population[feasible_mask][best_front]
            survived = Population.merge(survived, candidates)

        if feasible_frac == 0.0 or len(survived) < self.problem.n_var:
            # Conduct survival using CV vs. -dist from lowest_cv and non-dominated sorting
            dcv_obj_arr, self.lowest_cv = self.calc_dcv(population, return_lcv=True)
            dummy_pop = Population(self.problem, len(dcv_obj_arr))
            dummy_pop.assign_obj(dcv_obj_arr)
            # best_front = self.nondom.do(dcv_obj_arr, only_non_dominated_front=True)
            best_front = self.rank_and_crowd._do(self.problem, dummy_pop, n_survive=self.problem.n_var, cons_val=None)
            candidates = population[best_front]
            survived = Population.merge(survived, candidates)

        return survived

    def calc_dcv(self, population, lowest_cv=None, return_lcv=False):
        cv_sum = self.calc_cv(population)
        var_arr = population.extract_var()
        if lowest_cv is None:
            feasible = cv_sum == 0.0
            if np.count_nonzero(feasible) == 0:
                # Select lowest CV individual
                lcv = np.argpartition(cv_sum, 1)[:1]
            else:
                # Select feasible individuals
                lcv = np.where(feasible)[0]
            # Variable array
            lowest_cv = var_arr[lcv]

        # Calculate decision-space diversity component
        dist_mat = distance.cdist(np.atleast_2d(lowest_cv), np.atleast_2d(var_arr))
        diversity = -np.min(dist_mat, axis=0)

        # Concatenate objective vector
        obj_arr = np.hstack((cv_sum[:, None], diversity[:, None]))

        if return_lcv:
            return obj_arr, lowest_cv

        return obj_arr

    @staticmethod
    def calc_cv(population, normalise=None):
        cons_arr = np.atleast_2d(copy.deepcopy(population.extract_cons()))
        if normalise is not None:
            cons_arr = cons_arr / normalise
        cons_arr[cons_arr <= 0.0] = 0.0
        cv = np.sum(cons_arr, axis=1)

        return cv

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

    def plot_subplots(self, population, offspring, opt):
        # Evaluate expensively
        exp_offs = self.surrogate_evaluator.do(self.problem.obj_func, self.problem, copy.deepcopy(offspring))

        # Calculate CV values
        exp_cv = self.calc_cv(exp_offs)
        pred_cv = self.calc_cv(offspring)
        # print(exp_cv)
        # print(pred_cv)

        # Extract objectives
        pred_offs_obj = np.atleast_2d(offspring.extract_obj())
        exp_offs_obj = np.atleast_2d(exp_offs.extract_obj())
        pop_obj = np.atleast_2d(population.extract_obj())
        opt_obj = np.atleast_2d(opt.extract_obj())

        # Create plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
        fig.supxlabel('Obj 1', fontsize=14)
        fig.supylabel('Obj 2', fontsize=14)

        # First Scatter plot
        ax = fig.axes[0]
        ax.scatter(opt_obj[:, 0], opt_obj[:, 1], color='k', s=10, label="Pareto")
        # ax.scatter(pop_obj[:, 0], pop_obj[:, 1], color='b', s=20, label="Pop")
        ax_exp = ax.scatter(exp_offs_obj[:, 0], exp_offs_obj[:, 1], c=exp_cv, s=30,
                            vmin=np.min(exp_cv), vmax=np.max(exp_cv), label="Exp Offs")
        fig.colorbar(ax_exp, ax=ax)
        ax.set_title("Exact Evaluated")
        ax.legend(loc='best', frameon=False)

        # First Scatter plot
        ax = fig.axes[1]
        ax.scatter(opt_obj[:, 0], opt_obj[:, 1], color='k', s=10, label="Pareto")
        # ax.scatter(pop_obj[:, 0], pop_obj[:, 1], color='b', s=20, label="Pop")
        ax_pred = ax.scatter(pred_offs_obj[:, 0], pred_offs_obj[:, 1], c=pred_cv, s=30,
                             vmin=np.min(pred_cv), vmax=np.max(pred_cv), label="Pred Offs")
        fig.colorbar(ax_pred, ax=ax)
        ax.set_title("Surrogate Evaluated")
        ax.legend(loc='best', frameon=False)

        # plt.show()
        fig.savefig(f'/home/juan/Desktop/cs1_experimentation/exp/{self.problem.name}_gen_{self.evaluator.n_eval}.png')

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