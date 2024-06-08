import numpy as np
from scipy.spatial import distance
import copy
import random
from collections import OrderedDict

from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib
plt.style.use('seaborn-talk')
np.set_printoptions(suppress=True)
# matplotlib.use('TkAgg')
line_colors = ['green', 'blue', 'red', 'orange', 'cyan', 'lawngreen', 'm', 'orangered','sienna', 'gold', 'violet', 'indigo', 'cornflowerblue']

from optimisation.model.evaluator import Evaluator
from optimisation.algorithms.evolutionary_algorithm import EvolutionaryAlgorithm

from optimisation.operators.sampling.random_sampling import RandomSampling
from optimisation.operators.selection.random_selection import RandomSelection
from optimisation.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from optimisation.operators.mutation.polynomial_mutation import PolynomialMutation
from optimisation.operators.survival.theta_survival import ThetaSurvival
from optimisation.util.misc import calc_V
from optimisation.util.non_dominated_sorting import NonDominatedSorting
from optimisation.model.population import Population
from optimisation.model.duplicate import DefaultDuplicateElimination
from optimisation.model.repair import BasicBoundsRepair, BounceBackBoundsRepair
from optimisation.metrics.indicator import Indicator

from optimisation.util.reference_directions import UniformReferenceDirection
from optimisation.util.calculate_hypervolume import calculate_hypervolume


class BISAEA(EvolutionaryAlgorithm):
    """
    BISAEA: "Bi-Indicator-Based Surrogate-Assisted Multi-Objective Evolutionary Algorithm for Computationally Expensive
    Problems": Wang, Dong 2022.
    DOI: https://dx.doi.org/10.2139/ssrn.4188470
    """

    def __init__(self,
                 ref_dirs=None,
                 n_population=100,
                 surrogate=None,
                 sampling=RandomSampling(),
                 selection=RandomSelection(),
                 crossover=SimulatedBinaryCrossover(eta=20, prob=1.0),
                 mutation=PolynomialMutation(eta=20, prob=None),
                 survival=None,
                 eliminate_duplicates=DefaultDuplicateElimination(epsilon=1e-2),
                 **kwargs):

        # Reference directions
        self.ref_dirs = ref_dirs

        # Performance indicator metric
        self.indicator = Indicator(metric='igd')

        # Bounds repair
        self.repair = BasicBoundsRepair()

        # Population parameters
        self.n_population = n_population

        # Surrogate strategy instance
        # self.surrogate = surrogate

        # Sampling
        self.sampling = sampling

        # Survival
        # survival = ThetaSurvival(ref_dirs=self.ref_dirs, theta=5.0, filter_infeasible=True)
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
        self.w_max_0 = 5           # Number of internal offspring generations
        self.w_max = self.w_max_0
        self.k = 0.05               # Constant in indicator aggregation equation

        # Initialise from the evolutionary algorithm class
        super()._initialise()

        # Miscellaneous
        self.n_init = self.evaluator.n_eval
        self.history_count = []
        self.use_kernel_switching = False

        # Initialise archives of non-dominated solutions
        fronts = NonDominatedSorting().do(self.population.extract_obj(), return_rank=False)
        self.opt = copy.deepcopy(self.population[fronts[0]])
        self.representatives = self.opt

        # Extract objective surrogates
        self.obj_surrogates = self.surrogate.obj_surrogates
        self.surrogate_evaluator = Evaluator()
        self.surrogate_strategy = self.surrogate

    def _next(self):
        # Update normalisation bounds
        self.update_norm_bounds(self.population)

        # Generate offspring for w_max internal iterations
        # self.w_max = int(self.w_max_0 * (1 - (self.evaluator.n_eval - self.n_init) / self.max_f_eval))
        # self.w_max = np.maximum(self.w_max, 2)
        # print('w: ', self.w_max)
        self.offspring = self.generate_offspring(self.representatives, n_offspring=self.n_offspring, n_gen=self.w_max)

        # Conduct final selection for infill points
        infill = self.one_by_one_selection(self.offspring, n_survive=self.n_infill)

        # Remove duplicates
        if len(infill) > 1:
            # Within the infill population
            infill = self.eliminate_duplicates.do(infill, to_itself=True)
            infill = self.repair.do(self.problem, infill)

        # Against the rest of the population
        infill = self.eliminate_duplicates.do(infill, self.population)
        print('n infill: ', len(infill))

        # Handle case of no points founds
        if len(infill) == 0:
            print('No infill points were found!')
            return

        # Evaluate infill points expensively and merge with population
        infill = self.evaluator.do(self.problem.obj_func, self.problem, infill)

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

        # Calculate Performance Indicator
        old_igd = self.indicator.previous_value()
        igd = self.indicator.do(self.problem, self.opt, return_value=True)
        print(f"IGD: {igd:.4f}")

        # Keep track of n_infill per generation
        self.history_count.append(self.evaluator.n_eval - self.n_init)

        # Kernel switching
        if self.use_kernel_switching:
            if (old_igd - igd) == 0:
                for obj_cntr in range(self.problem.n_obj):
                    kernels_to_keep = self.obj_surrogates[obj_cntr].predict_random_accuracy()
                    print(f"Obj {obj_cntr} using kernels: {kernels_to_keep}")

        # Update surrogate models
        new_x_var = np.atleast_2d(infill.extract_var())
        new_y_var = np.atleast_2d(infill.extract_obj())
        for obj_cntr in range(self.problem.n_obj):
            self.obj_surrogates[obj_cntr].add_points(new_x_var, new_y_var[:, obj_cntr])
            self.obj_surrogates[obj_cntr].train()

        # DEBUG PURPOSES ONLY
        # if self.evaluator.n_eval > 350:
        # self.plot_pareto(ref_vec=self.ref_dirs, scaling=1, labels=['Reps', 'Pop', 'Infill'],
        #                  obj_array1=old_reps.extract_obj(),
        #                  obj_array2=old_pop.extract_obj(),
        #                  obj_array3=infill.extract_obj())

    def generate_offspring(self, representatives, n_offspring=300, n_gen=20):
        survived = copy.deepcopy(representatives)
        n_reps = len(survived)
        for gen in range(n_gen-1):
            # Create offspring and evaluate with surrogate
            offspring = self.mating.do(self.problem, survived, n_offspring)
            offspring = self.repair.do(self.problem, offspring)

            # Merge offspring with parent individuals
            offspring = Population.merge(offspring, survived)

            # Evaluate predicted objective values with surrogates
            offspring = self.surrogate_evaluator.do(self.surrogate.obj_func, self.problem, offspring)

            # Select promising individuals with one-by-one-selection
            # survived = self.one_by_one_selection(offspring, n_survive=n_reps)
            fronts = NonDominatedSorting().do(offspring.extract_obj(), return_rank=False)
            survived = offspring[fronts[0]]

        # Create offspring of final n_offspring size
        survived = self.mating.do(self.problem, survived, n_offspring)
        survived = self.repair.do(self.problem, survived)
        survived = self.surrogate_evaluator.do(self.surrogate.obj_func, self.problem, survived)

        # TODO: DEBUG --------------------------------------------------------------------
        survived = self.surrogate_evaluator.do(self.problem.obj_func, self.problem, survived)
        self.plot_pareto(ref_vec=self.ref_dirs, scaling=1, labels=['Survived', 'Reps', 'Pop'],
                         obj_array3=self.population.extract_obj(),
                         obj_array2=self.representatives.extract_obj(),
                         obj_array1=survived.extract_obj())
        # TODO: DEBUG ---------------------------------------------------------------------

        return survived

    def one_by_one_selection(self, offspring, n_survive=1):
        offspring = copy.deepcopy(offspring)
        # Calculate CI measure
        CI = self.calc_convergence_indicator(offspring)
        # CI = -self.calc_hypervol_contrib(offspring, normalise=True)

        # Calculate DI measure
        DI = self.calc_diversity_indicator(offspring, self.V)

        # Combine measures for non-dominated ranking
        obj_arr = np.hstack((-DI[:, None], CI[:, None]))
        fronts = NonDominatedSorting().do(obj_arr, return_rank=False)
        best_indices = fronts[0]
        offspring = offspring[best_indices]

        # dummy_pop = Population(self.problem, len(offspring))
        # dummy_pop.assign_var(self.problem, offspring.extract_var())
        # dummy_pop.assign_obj(obj_arr)
        # offspring = self.survival.do(self.problem, dummy_pop, len(dummy_pop), first_rank_only=True)

        if len(offspring) == 0:
            breakpoint()

        # TODO: DEBUG ---------------------------------------------------------------------
        # fig, ax = plt.subplots(1, 1, figsize=(9, 7))
        # fig.supxlabel('DI', fontsize=14)
        # fig.supylabel('CI', fontsize=14)
        # for rank, indices in enumerate(fronts):
        #     ax.scatter(obj_arr[indices, 0], obj_arr[indices, 1], s=30, label=f"rank: {rank}")
        # ax.scatter(obj_arr[best_indices, 0], obj_arr[best_indices, 1], s=100, color='r')
        # # ax.scatter(offspring.extract_obj()[:, 0], offspring.extract_obj()[:, 1], s=100, color='r')
        # plt.show()
        # TODO: DEBUG ---------------------------------------------------------------------

        # Do one-by-one selection until n_survive infill points are identified
        survived = Population(self.problem, 0)
        population = copy.deepcopy(self.representatives)
        while len(survived) < n_survive:
            # Return if only one offspring left
            if len(offspring) == 1:
                survived = Population.merge(survived, offspring)
                break

            # Calculate maximin of the angle between first rank offspring and the population representatives
            vectors = calc_V(np.atleast_2d(population.extract_obj()))
            min_angles = self.calc_diversity_indicator(offspring, vectors)
            selected = np.argmax(min_angles)
            candidate = offspring[[selected]]

            # TODO: DEBUG ---------------------------------------------------------------------
            # self.plot_pareto(ref_vec=self.ref_dirs, scaling=1, labels=['Offs', 'Reps', 'Survived'],
            #                  obj_array3=candidate.extract_obj(),
            #                  obj_array2=population.extract_obj(),
            #                  obj_array1=offspring.extract_obj())
            # TODO: DEBUG ---------------------------------------------------------------------

            # Assign to survived population
            survived = Population.merge(survived, candidate)

            # Remove selected from offspring and add to representatives
            population = Population.merge(population, candidate)
            new_indices = np.arange(len(offspring)) != selected
            offspring = offspring[new_indices]

        # TODO: remove
        # Extract variables and re-assign to population without surrogate fitness values
        final_pop = Population(self.problem, len(survived))
        x_vars = np.atleast_2d(survived.extract_var())
        final_pop.assign_var(self.problem, x_vars)

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

    def calc_hypervol_contrib(self, population, normalise=True):
        # Extract objectives
        obj_array = population.extract_obj()
        n_obj = obj_array.shape[1]
        n_pop = len(population)

        if normalise:
            # Normalise to [0, 1] in R^n
            f_min = np.min(obj_array, axis=0)
            f_max = np.max(obj_array, axis=0)
            obj_array = (obj_array - f_min) / (f_max - f_min)

            # Reference point
            ref_point = np.ones(n_obj)
        else:
            # Extract reference point from largest objective in population
            ref_point = np.array([np.max(obj_array[:, i]) for i in range(n_obj)])

        # Calculate hypervolume contribution of each point: delta_HV = product(ref_point - obj_array)
        hv = np.prod(ref_point - obj_array, axis=1)

        # DEBUG ---------------------------------------------------------------------
        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.set_xlabel('F1')
        # ax.set_ylabel('F2')
        # ax.set_zlabel('HV')
        #
        # # x, y, z = plt.grid(obj_array[:, 0], obj_array[:, 1], hv)
        # # ax.scatter3D(obj_array[:, 0], obj_array[:, 1], hv)
        # ax.tricontourf(obj_array[:, 0], obj_array[:, 1], hv)
        #
        # # plt.legend(loc='upper right')
        # plt.show()
        # DEBUG ---------------------------------------------------------------------

        return hv

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
