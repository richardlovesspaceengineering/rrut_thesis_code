import numpy as np
from scipy.spatial import distance
import copy
import random
from collections import OrderedDict
from termcolor import colored

from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib
plt.style.use('seaborn-talk')
np.set_printoptions(suppress=True)
# matplotlib.use('TkAgg')
line_colors = ['green', 'blue', 'red', 'orange', 'cyan', 'lawngreen', 'm', 'orangered','sienna', 'gold', 'violet', 'indigo', 'cornflowerblue']

from optimisation.model.evaluator import Evaluator
from optimisation.algorithms.sa_evolutionary_algorithm import SAEvolutionaryAlgorithm

from optimisation.surrogate.rbf_tuner import RBFTuner
from optimisation.operators.sampling.lhs_loader import LHSLoader
from optimisation.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from optimisation.operators.selection.random_selection import RandomSelection
from optimisation.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from optimisation.operators.mutation.polynomial_mutation import PolynomialMutation

from optimisation.util.misc import calc_V
from optimisation.util.non_dominated_sorting import NonDominatedSorting
from optimisation.model.population import Population
from optimisation.model.duplicate import DefaultDuplicateElimination
from optimisation.model.repair import BasicBoundsRepair, BounceBackBoundsRepair
from optimisation.metrics.indicator import Indicator

from optimisation.util.dominator import get_relation

from optimisation.util.reference_directions import UniformReferenceDirection
from optimisation.util.calculate_hypervolume import calculate_hypervolume

from optimisation.output.generation_extractor import GenerationExtractor


class BISAEA(SAEvolutionaryAlgorithm):
    """
    BISAEA: "Bi-Indicator-Based Surrogate-Assisted Multi-Objective Evolutionary Algorithm for Computationally Expensive
    Problems": Wang, Dong 2022.
    DOI: https://dx.doi.org/10.2139/ssrn.4188470
    """

    def __init__(self,
                 ref_dirs=None,
                 n_population=100,
                 surrogate=None,
                 sampling=LHSLoader(),  # LatinHypercubeSampling(),
                 selection=RandomSelection(),
                 crossover=SimulatedBinaryCrossover(eta=20, prob=1.0),
                 mutation=PolynomialMutation(eta=20, prob=None),
                 survival=None,
                 eliminate_duplicates=DefaultDuplicateElimination(epsilon=1e-4),
                 **kwargs):

        # Reference directions
        self.ref_dirs = ref_dirs

        # Performance indicator metric
        self.indicator = Indicator(metric='igd')

        # Output path
        if 'output_path' in kwargs:
            self.output_path = kwargs['output_path']

        # Bounds repair
        self.repair = BasicBoundsRepair()

        # Population parameters
        self.n_population = n_population

        # Surrogate strategy instance
        # self.surrogate = surrogate

        # Sampling
        self.sampling = sampling

        # Survival
        self.survival = survival

        # Optimum position
        self.opt = None

        super().__init__(n_population=n_population,
                         sampling=sampling,
                         selection=selection,
                         crossover=crossover,
                         mutation=mutation,
                         survival=survival,
                         eliminate_duplicates=eliminate_duplicates,
                         surrogate=surrogate,
                         **kwargs)

    def _initialise(self):

        # Generation parameters
        self.max_f_eval = self.max_gen
        print(self.max_f_eval)

        # Reference vector parameters
        self.ideal = np.inf
        self.nadir = -np.inf
        self.V = calc_V(self.ref_dirs)

        # BISAEA Parameters
        self.n_infill = 3           # Number of infill points per generation
        self.n_offspring = 300      # TODO: Decide on an appropriate value
        self.w_max = 10             # Number of internal offspring generations
        self.k = 0.05               # Constant in indicator aggregation equation

        # Initialise from the evolutionary algorithm class
        super()._initialise()

        # Miscellaneous
        self.n_init = self.evaluator.n_eval
        self.history_count = []

        # RBF regression surrogate hyperparameter tuner
        self.rbf_tuning_frac = 0.20
        self.rbf_tuner = RBFTuner(n_dim=self.problem.n_var, lb=self.problem.x_lower, ub=self.problem.x_upper,
                                  c=0.5, p_type='linear', kernel_type='gaussian',
                                  width_range=(0.1, 10), train_test_split=self.rbf_tuning_frac,
                                  max_evals=50, verbose=False)

        # Minimisation Exploitation of RBF surrogates
        self.last_cluster = -2
        # self.last_minimised = -1
        self.minimise_frac = int(self.max_gen / self.problem.n_var)

        self.ref_dirs = UniformReferenceDirection(self.problem.n_obj, n_partitions=20).do()

        # Initialise archives of non-dominated solutions
        fronts = NonDominatedSorting().do(self.population.extract_obj(), return_rank=False)
        self.opt = copy.deepcopy(self.population[fronts[0]])
        self.representatives = self.opt
        igd = self.indicator.do(self.problem, self.opt, self.evaluator.n_eval, return_value=True)

        # Extract objective surrogates
        self.obj_surrogates = self.surrogate.obj_surrogates
        self.surrogate_evaluator = Evaluator()
        self.surrogate_strategy = self.surrogate

        # Data extractor
        self.filename = self.problem.name.lower() + '_' + str(self.save_name) + '_maxgen_' + \
                        str(round(self.max_f_eval)) + '_sampling_' + str(self.surrogate_strategy.n_training_pts) + \
                        '_seed_' + str(self.surrogate_strategy.sampling_seed)
        self.data_extractor = GenerationExtractor(filename=self.filename, base_path=self.output_path)
        self.data_extractor.add_generation(self.population, self.n_gen)
        self.data_extractor.add_front(self.opt, self.n_gen, indicator_values=np.array(igd))

    def _next(self):
        # Update normalisation bounds
        self.update_norm_bounds(self.population)

        # Generate offspring for w_max internal iterations
        self.offspring = self.generate_offspring(self.representatives, n_offspring=self.n_offspring, n_gen=self.w_max)

        # Conduct final selection for infill points
        infill_population = self.one_by_one_selection(self.offspring, n_survive=self.n_infill)

        # # Find the predicted surrogate extremes for the selected objective
        # if (self.evaluator.n_eval % self.minimise_frac) == 0:
        #     # Minimisation of surrogate predictions
        #     extra_infill = self.minimise_regressors(self.offspring)
        #     infill_population = Population.merge(infill_population, extra_infill)
        #     print(colored('----------------- Minimising Regressor Surrogate Predictions ------------------', 'green'))

        # Evaluate infill points expensively and merge with population
        infill = self.evaluator.do(self.problem.obj_func, self.problem, infill_population)

        # TODO: DEBUG ---------------------------------------------------------------------
        # self.plot_pareto(ref_vec=self.ref_dirs, scaling=1, labels=['Pop', 'Reps', 'Infill'],
        #                  obj_array1=self.population.extract_obj(),
        #                  obj_array2=self.representatives.extract_obj(),
        #                  obj_array3=infill.extract_obj())
        # TODO: DEBUG ---------------------------------------------------------------------

        # Update optimum front
        old_pop = copy.deepcopy(self.population)
        old_reps = copy.deepcopy(self.representatives)
        self.population = Population.merge(self.population, infill)
        fronts = NonDominatedSorting().do(self.population.extract_obj(), return_rank=False)
        self.opt = copy.deepcopy(self.population[fronts[0]])
        self.representatives = self.opt

        # Hyperparameter tuning of RBF widths (for each objective surrogate)
        if (self.n_gen % self.minimise_frac) == 0:
            print(colored('----------------- Hyper-tuning Regressor Surrogate Widths ------------------', 'green'))
            # Conduct optimisation of each objective surrogate
            opt_models, opt_params = self.rbf_tuner.do(problem=self.problem,
                                                       population=self.population,
                                                       n_obj=len(self.obj_surrogates),
                                                       fronts=fronts)
            # Set newly optimised RBF models
            for cntr, model in enumerate(opt_models):
                self.obj_surrogates[cntr].model = model
            print(colored('----------------------------------------------------------------------------', 'green'))

        # Calculate Performance Indicator
        igd = self.indicator.do(self.problem, self.opt, self.evaluator.n_eval, return_value=True)
        print('IGD: ', np.round(igd, 4))
        self.data_extractor.add_generation(infill, self.n_gen)
        self.data_extractor.add_front(self.opt, self.n_gen, indicator_values=np.array(igd))

        # Keep track of n_infill per generation
        self.history_count.append(self.evaluator.n_eval - self.n_init)

        # Update surrogate models
        new_x_var = np.atleast_2d(infill.extract_var())
        new_y_var = np.atleast_2d(infill.extract_obj())
        for obj_cntr in range(self.problem.n_obj):
            self.obj_surrogates[obj_cntr].add_points(new_x_var, new_y_var[:, obj_cntr])
            self.obj_surrogates[obj_cntr].train()

        # DEBUG PURPOSES ONLY
        # if self.evaluator.n_eval > 100:
        #     self.plot_pareto(ref_vec=self.ref_dirs, scaling=1, labels=['Reps', 'Pop', 'Infill'],
        #                      obj_array1=old_reps.extract_obj(),
        #                      obj_array2=old_pop.extract_obj(),
        #                      obj_array3=infill.extract_obj())

    def generate_offspring(self, representatives, n_offspring=300, n_gen=10):
        survived = copy.deepcopy(representatives)
        n_reps = len(survived)
        for gen in range(n_gen-1):
            # Create offspring and evaluate with surrogate
            offspring = self.mating.do(self.problem, survived, n_offspring)

            # Bounds repair
            offspring = self.repair.do(self.problem, offspring)

            # Merge offspring with parent individuals
            offspring = Population.merge(offspring, survived)

            # Evaluate predicted objective values with surrogates
            offspring = self.surrogate_evaluator.do(self.surrogate.obj_func, self.problem, offspring)

            # Select promising individuals
            fronts = NonDominatedSorting().do(offspring.extract_obj(), return_rank=False)
            survived = offspring[fronts[0]]

        # Create offspring of final n_offspring size
        survived = self.mating.do(self.problem, survived, n_offspring)
        survived = self.repair.do(self.problem, survived)
        survived = self.surrogate_evaluator.do(self.surrogate.obj_func, self.problem, survived)

        # TODO: DEBUG --------------------------------------------------------------------
        # survived = self.surrogate_evaluator.do(self.problem.obj_func, self.problem, survived)
        # self.plot_pareto(ref_vec=self.ref_dirs, scaling=1, labels=['Survived', 'Reps', 'Pop'],
        #                  obj_array3=self.population.extract_obj(),
        #                  obj_array2=self.representatives.extract_obj(),
        #                  obj_array1=survived.extract_obj())
        # TODO: DEBUG ---------------------------------------------------------------------

        return survived

    def one_by_one_selection(self, offspring, n_survive=1):
        offspring = copy.deepcopy(offspring)
        offspring = self.eliminate_duplicates.do(offspring, self.population)
        obj_array = offspring.extract_obj()

        # Update reference vector counter
        self.last_cluster += 2
        if self.last_cluster >= len(self.ref_dirs):
            if self.last_cluster % 2 == 0:
                self.last_cluster = 1
            else:
                self.last_cluster = 0
        print(colored('Selected Cluster: ', 'blue'), self.last_cluster, self.ref_dirs[self.last_cluster])

        weighted_obj = np.sum(obj_array * self.ref_dirs[self.last_cluster], axis=1)
        idx = np.argmin(weighted_obj)
        selected = [idx]

        # Filter out duplicates
        survived = offspring[np.array(selected)]

        # Select randomly if no infill point was found
        if len(survived) == 0:
            rand_val = np.random.uniform(0.0, 1.0, 1)
            w_select = 0.50
            if rand_val > w_select:
                rand_int = np.random.randint(0, len(self.offspring), 1)
                survived = copy.deepcopy(self.offspring[rand_int])
                print(colored('----------------- No infill point was found! Selecting randomly from offspring ----------------', 'yellow'))
            else:
                # Minimisation of surrogate predictions
                survived = self.minimise_regressors(self.offspring)
                print(colored('----------------- No infill point was found! Minimising Regressor Surrogate Predictions -------', 'yellow'))

        return survived

    def calc_convergence_indicator(self, population):
        # Extract objectives and normalise
        obj_array = population.extract_obj()
        obj_array = (obj_array - self.ideal) / (self.nadir - self.ideal)

        # Evaluate I_epsilon+ indicator
        n = len(population)
        I_plus = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    I_plus[i, j] = np.NaN
                else:
                    I_plus[i, j] = np.max(obj_array[i] - obj_array[j])

        # Calculate scalar term
        c_scalar = np.nanmax(I_plus, axis=1)

        # Aggregate to single fitness value per individual
        fitness = np.nansum(-np.exp(-I_plus / (c_scalar * self.k)), axis=1)

        # Simple sum of objectives as convergence
        # fitness = np.sum(obj_array, axis=1)

        # # EDN CONVERGENCE
        # # Calculate Pareto Domination matrix (-1: dominated, 1: dominates, 0: non-dominated)
        # dom_mat = calculate_domination_matrix(obj_array, flag='alpha')
        #
        # # Assign Expected Dominance Number (EDN)
        # dom_mat[dom_mat != 1] = 0
        # fitness = -np.sum(dom_mat, axis=1)

        return fitness

    def calc_diversity_indicator(self, population, ref_obj):
        # Extract objectives and normalise
        obj_array = population.extract_obj()
        obj_array = (obj_array - self.ideal) / (self.nadir - self.ideal)

        # Calculate minimum angles between population and reference vectors
        dist_to_ideal = np.linalg.norm(obj_array, axis=1)
        dist_to_ideal[dist_to_ideal < 1e-64] = 1e-64

        # Normalize by distance to ideal
        obj_prime = obj_array / dist_to_ideal[:, None]

        # Calculate for each solution the minimum acute angles to ref dirs
        acute_angle = np.arccos(obj_prime @ ref_obj.T)
        min_angle = np.min(acute_angle, axis=1)

        return min_angle

    def update_norm_bounds(self, population):
        # Extract objectives
        obj_array = population.extract_obj()

        # Find lower and upper bounds
        f_min = np.min(obj_array, axis=0)
        f_max = np.max(obj_array, axis=0)

        # Update the ideal and nadir points
        self.ideal = np.minimum(f_min, self.ideal)
        self.nadir = f_max

    @staticmethod
    def plot_pareto(ref_vec=None, obj_array1=None, obj_array2=None, obj_array3=None, fronts=None, scaling=1.5,
                    labels=None):

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
                        ax.scatter(obj_array1[frnt, 0], obj_array1[frnt, 1], color=line_colors[i], s=75,
                                   label=f"rank {i}")
                else:
                    ax.scatter(obj_array1[:, 0], obj_array1[:, 1], color=line_colors[6], s=150, label=labels[0])

            try:
                if obj_array2 is not None:
                    obj_array2 = np.atleast_2d(obj_array2)
                    if fronts is not None:
                        for i, frnt in enumerate(fronts):
                            ax.scatter(obj_array2[frnt, 0], obj_array2[frnt, 1], color=line_colors[i], s=75,
                                       label=f"rank {i}")
                    else:
                        ax.scatter(obj_array2[:, 0], obj_array2[:, 1], color=line_colors[1], s=50, label=labels[1])
            except:
                pass

            if obj_array3 is not None:
                obj_array3 = np.atleast_2d(obj_array3)
                if fronts is not None:
                    for i, frnt in enumerate(fronts):
                        ax.scatter(obj_array3[frnt, 0], obj_array3[frnt, 1], color=line_colors[i], s=75,
                                   label=f"rank {i}")
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
                        ax.scatter3D(obj_array1[frnt, 0], obj_array1[frnt, 1], obj_array1[frnt, 2],
                                     color=line_colors[i], s=50, label=f"rank {i}")
                else:
                    ax.scatter3D(obj_array1[:, 0], obj_array1[:, 1], obj_array1[:, 2], color=line_colors[0], s=50,
                                 label=f"theta_pop")

            if obj_array2 is not None:
                obj_array2 = np.atleast_2d(obj_array2)
                if fronts is not None:
                    for i, frnt in enumerate(fronts):
                        ax.scatter3D(obj_array2[frnt, 0], obj_array2[frnt, 1], obj_array2[frnt, 2],
                                     color=line_colors[i], s=50, label=f"rank {i}")
                else:
                    ax.scatter3D(obj_array2[:, 0], obj_array2[:, 1], obj_array2[:, 2], color=line_colors[1], s=25,
                                 label=f"pareto_pop")

            if obj_array3 is not None:
                obj_array3 = np.atleast_2d(obj_array3)
                if fronts is not None:
                    for i, frnt in enumerate(fronts):
                        ax.scatter3D(obj_array3[frnt, 0], obj_array3[frnt, 1], obj_array3[frnt, 2],
                                     color=line_colors[i], s=50, label=f"rank {i}")
                else:
                    ax.scatter3D(obj_array3[:, 0], obj_array3[:, 1], obj_array3[:, 2], color=line_colors[2], s=15,
                                 label=f"infill")

        plt.legend(loc='best', frameon=False)
        plt.show()
        # plt.savefig('/home/juan/PycharmProjects/optimisation_framework/multi_obj/results/zdt1_mmode_gen_' + str(len(self.surrogate.population)) + '.png')


def get_ref_vec_relation(c_i, c_j, d_i, d_j):
    # If not belonging to the same cluster: non-dominated
    if c_i != c_j:
        return 0
    else:
        # If same cluster and d_i < d_j: i dominates j
        if d_i < d_j:
            return 1
        else:
            # Otherwise if d_i > d_j: j dominates i
            return -1


def get_epsilon_relation(a, b, cv_a=None, cv_b=None, epsilon=0.1):
    # If constraint values are passed, compare them
    if cv_a is not None and cv_b is not None:
        if cv_a < cv_b:
            return 1
        elif cv_b < cv_a:
            return -1

    # If constraint values are equal for both individuals, or are not passed
    val = 0
    # Compare a and b
    for i in range(len(a)):
        # Alpha dominance terms

        if (a[i] - epsilon) < b[i]:
            if val == -1:
                # We are now indifferent because elements in both individuals dominate the other
                return 0
            val = 1
        elif b[i] < (a[i] - epsilon):
            if val == 1:
                # We are now indifferent because elements in both individuals dominate the other
                return 0
            val = -1
    return val


def get_alpha_relation(a, b, cv_a=None, cv_b=None, alpha=0.1):
    # If constraint values are passed, compare them
    if cv_a is not None and cv_b is not None:
        if cv_a < cv_b:
            return 1
        elif cv_b < cv_a:
            return -1

    # If constraint values are equal for both individuals, or are not passed
    val = 0
    # Compare a and b
    for i in range(len(a)):
        # Alpha dominance terms
        term_a = alpha * np.sum(a[np.arange(len(a)) != i])
        term_b = alpha * np.sum(b[np.arange(len(a)) != i])

        if (a[i] + term_a) < (b[i] + term_b):
            if val == -1:
                # We are now indifferent because elements in both individuals dominate the other
                return 0
            val = 1
        elif (b[i] + term_b) < (a[i] + term_a):
            if val == 1:
                # We are now indifferent because elements in both individuals dominate the other
                return 0
            val = -1
    return val


def get_c_alpha_relation(a, b, cv_a=None, cv_b=None, alpha=0.1):
    # If constraint values are passed, compare them
    if cv_a is not None and cv_b is not None:
        if cv_a < cv_b:
            return 1
        elif cv_b < cv_a:
            return -1

    # If constraint values are equal for both individuals, or are not passed
    val = 0
    # Compare a and b
    for i in range(len(a)):
        # CAlpha dominance terms
        a_dash = a[i] + alpha * np.sum(a[np.arange(len(a)) != i])
        b_dash = b[i] + alpha * np.sum(b[np.arange(len(b)) != i])

        a_hat = np.maximum(a_dash, np.sqrt(np.sum(a_dash[np.arange(len(a)) != i] ** 2)))
        b_hat = np.maximum(b_dash, np.sqrt(np.sum(b_dash[np.arange(len(a)) != i] ** 2)))

        if a_hat < b_hat:
            if val == -1:
                # We are now indifferent because elements in both individuals dominate the other
                return 0
            val = 1
        elif b_hat < a_hat:
            if val == 1:
                # We are now indifferent because elements in both individuals dominate the other
                return 0
            val = -1
    return val


def get_nlad_relation(a, b, cv_a=None, cv_b=None, alpha=0.1):
    # If constraint values are passed, compare them
    if cv_a is not None and cv_b is not None:
        if cv_a < cv_b:
            return 1
        elif cv_b < cv_a:
            return -1

    # If constraint values are equal for both individuals, or are not passed
    val = 0
    # Compare a and b
    for i in range(len(a)):
        # Alpha dominance terms
        term_a = alpha * a[i] ** 3 + np.sum(a[np.arange(len(a)) != i])
        term_b = alpha * b[i] ** 3 + np.sum(b[np.arange(len(a)) != i])
        if term_a < term_b:
            if val == -1:
                # We are now indifferent because elements in both individuals dominate the other
                return 0
            val = 1
        elif term_b < term_a:
            if val == 1:
                # We are now indifferent because elements in both individuals dominate the other
                return 0
            val = -1
    return val


def select_dom(flag):
    """
    Returns pointer to dominance relation method
    """
    # Switch to different domination relations
    if 'pareto' in flag:
        dom_func = get_relation
    elif 'alpha' in flag:
        dom_func = get_alpha_relation
    elif 'c_alpha' in flag:
        dom_func = get_c_alpha_relation
    elif 'nlad' in flag:
        dom_func = get_nlad_relation
    elif 'epsilon' in flag:
        dom_func = get_epsilon_relation
    else:
        dom_func = get_relation
    return dom_func


def calculate_domination_matrix(f, cv=None, flag=None):
    # Select dominance relation
    domination = select_dom(flag)

    # Number of individuals
    n = f.shape[0]

    # Check if constraint values are passed
    if cv is None:
        cv = [None for _ in range(n)]

    # Output matrix
    m = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            m[i, j] = domination(f[i, :], f[j, :], cv[i], cv[j])
            m[j, i] = -m[i, j]

    return m
