import numpy as np
from scipy.spatial import distance
import copy
import random
import sys
from collections import OrderedDict

from mpi4py import MPI

from sklearn.svm import SVC, OneClassSVM

from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib
plt.style.use('seaborn-talk')
np.set_printoptions(suppress=True)
# matplotlib.use('TkAgg')
line_colors = ['green', 'blue', 'red', 'orange', 'cyan', 'lawngreen', 'm', 'orangered','sienna', 'gold', 'violet', 'indigo', 'cornflowerblue']

# from optimisation.model.algorithm import Algorithm
from optimisation.optimise import minimise
from optimisation.model.problem import Problem
from optimisation.setup import Setup
from optimisation.algorithms.evolutionary_algorithm import EvolutionaryAlgorithm
from optimisation.algorithms.multi_offspring_differential_evolution import MultiOffspringDE

from optimisation.surrogate.models.rbf_kernel_ensemble import RBFKernelEnsembleSurrogate
from optimisation.surrogate.models.ensemble import EnsembleSurrogate
from optimisation.surrogate.models.rbf import RadialBasisFunctions
from optimisation.surrogate.models.mars import MARSRegression
from optimisation.surrogate.models.rbf import RBF
from optimisation.surrogate.models.gp import GP

from optimisation.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from optimisation.operators.survival.rank_and_crowding_survival import RankAndCrowdingSurvival
from optimisation.util.non_dominated_sorting import NonDominatedSorting
from optimisation.model.population import Population
from optimisation.model.repair import BasicBoundsRepair

from optimisation.util.reference_directions import UniformReferenceDirection
from optimisation.util.calculate_hypervolume import calculate_hypervolume
from optimisation.operators.survival.reference_direction_survival import ReferenceDirectionSurvival


class MMODE(EvolutionaryAlgorithm):
    """
    Surrogate-Assisted Multi-objective Multi-Offspring Differential Evolution
    """

    def __init__(self,
                 n_population=None,
                 sampling=LatinHypercubeSampling(),
                 survival=RankAndCrowdingSurvival(),
                 # survival=ReferenceDirectionSurvival(),
                 **kwargs):

        # Population parameters
        self.n_population = n_population

        # # Surrogate strategy instance
        # self.surrogate = surrogate

        # Sampling
        self.sampling = sampling

        # Survival
        self.survival = survival

        # Optimum position
        self.opt = None

        super().__init__(n_population=n_population,
                         sampling=sampling,
                         survival=survival,
                         n_offspring=n_population,
                         **kwargs)

    def _initialise(self):

        # Generation parameters
        self.max_f_eval = self.max_gen + self.surrogate.n_training_pts
        print(self.max_f_eval)
        self.duplication_tolerance = 1e-2

        # MMODE Parameters
        self.n_infill = 3
        self.mode_f_gen = 11  # multiplier for number of max gen

        # # Calculate reference directions
        # self.ref_dirs = UniformReferenceDirection(self.problem.n_obj, n_partitions=self.problem.n_var).do()
        #
        # # Reference Vector Survival
        # self.survival = ReferenceDirectionSurvival(self.ref_dirs, filter_infeasible=True)

        # Initialise from the evolutionary algorithm class
        super()._initialise()

        self.population = copy.deepcopy(self.surrogate.population)

    def _next(self):

        # 1. Surrogate-Assisted MODE exploitation ----------------------------------------------------------------------
        seed = np.random.randint(low=0, high=10000, size=(1,))

        # Create Setup Instance with Surrogate
        setup = SurrogateSetup(self.surrogate.obj_surrogates)

        # Create Problem instance
        opt_prob = Problem('surrogate_exploitation', obj_func=setup.obj_func, map_internally=False, n_processors=1)

        # Set variables, constraints & objectives
        setup.do(opt_prob, parent_prob=self.problem)

        # Instance of Multi-offspring DE
        algorithm = MultiOffspringDE(n_population=self.n_population,
                                     max_gen=self.mode_f_gen*self.problem.n_var,  # 11
                                     init_population=self.surrogate.population,
                                     print=False,
                                     plot=False,
                                     save_results=False)
        minimise(opt_prob,
                 algorithm,
                 seed=seed,
                 hot_start=False,
                 x_init=None,
                 save_history=False)

        MPI.COMM_WORLD.barrier()

        # Extract non-dominated individuals from final population
        final_population = algorithm.pareto_archive

        # 2. Infill Selection Criteria from Optimal Surrogate Population -----------------------------------------------
        infill_population = self.knee_point_selection(final_population, n_survive=self.n_infill)

        # Duplicates within the exploit population
        is_duplicate = self.check_duplications(infill_population, other=None, epsilon=self.duplication_tolerance)
        infill_population = infill_population[np.invert(is_duplicate)[0]]

        # Duplicates with the surrogate population
        is_duplicate = self.check_duplications(infill_population, self.surrogate.population, epsilon=self.duplication_tolerance)
        infill_population = infill_population[np.invert(is_duplicate)[0]]
        print('n infill: ', len(infill_population))

        # Evaluate Infill Population expensively
        infill_population = self.evaluator.do(self.problem.obj_func, self.problem, infill_population)

        # 3. Kernel Switching ------------------------------------------------------------------------------------------
        old_opt = copy.deepcopy(self.opt)
        self.opt = self.survival.do(self.problem, self.surrogate.population, self.n_population, self.n_gen, self.max_gen)


        ##################################################33
        ranked_population = self.survival.do(self.problem, self.surrogate.population, len(self.surrogate.population))
        if len(self.surrogate.population) > 350:
            class_labels = self.pareto_classifier(ranked_population, infill_population)
        ##################################################33

        if isinstance(old_opt, Population):
            improvement = self.calculate_improvement(old_opt, self.opt)

            if improvement <= 0:
                for obj_cntr in range(self.problem.n_obj):
                    self.obj_surrogate = self.surrogate.obj_surrogates[obj_cntr]
                    # kernels_to_keep = self.obj_surrogate.predict_sep_accuracy(infill_population, obj_cntr)
                    kernels_to_keep = self.obj_surrogate.predict_random_accuracy()
                    print(f"Obj {obj_cntr} using kernels: {kernels_to_keep}")
            else:
                for obj_cntr in range(self.problem.n_obj):
                    self.surrogate.obj_surrogates[obj_cntr].clear_ranks()

        # 4. Surrogate Update ------------------------------------------------------------------------------------------
        self.surrogate.population = Population.merge(self.surrogate.population, infill_population)
        for obj_cntr in range(self.problem.n_obj):
            self.obj_surrogate = self.surrogate.obj_surrogates[obj_cntr]
            self.obj_surrogate.add_points(np.atleast_2d(infill_population.extract_var()), np.atleast_2d(infill_population.extract_obj())[:, obj_cntr])
            self.obj_surrogate.train()

        # # DEBUG PURPOSES ONLY
        # if len(self.surrogate.population) % 5 == 0 or len(self.surrogate.population) >= 360:
        #     self.plot_pf(final_population=self.opt)
        # if len(self.surrogate.population) > 70:
        #     self.plot_pf(final_population=self.opt)

    @staticmethod
    def calculate_improvement(old_opt, opt, infill_population=None):
        # TODO: Determine a metric to calculate the generational improvement between old_opt and opt
        # obj_dist_mat = distance.cdist(np.atleast_2d(infill_population.extract_obj()), old_opt.extract_obj())

        hv_old = calculate_hypervolume(old_opt.extract_obj())
        hv_new = calculate_hypervolume(opt.extract_obj())

        delta_hv = np.sum(hv_new) - np.sum(hv_old)
        print('HV improvement: ', delta_hv)

        # Return 1 if improvement was made, otherwise 0
        if delta_hv > 0:
            return 1
        else:
            return 0

    def plot_pf(self, final_population=None, infill_population=None, extreme_population=None, line=None, dividers=None, plot_pf=True):

        # Initialise Plot
        fig, ax = plt.subplots(1, 1, figsize=(9, 7))
        fig.supxlabel('Obj 1', fontsize=14)
        fig.supylabel('Obj 2', fontsize=14)

        # Knee selection stuff
        if line is not None:
            x_vals = line[0]
            y_vals = line[1]
            ax.plot(x_vals, y_vals, '-ok')

        if dividers is not None:
            x_vals = dividers[0]
            y_lines = dividers[1]
            for i in range(len(y_lines)):
                ax.plot(x_vals, y_lines[i], '-om')

        # Plot Non-dominated individuals and knee-point selection
        if extreme_population is not None:
            extreme_vars = extreme_population.extract_obj()
            obj_min = np.min(extreme_vars, axis=0)
            obj_max = np.max(extreme_vars, axis=0)
            extreme_vars = (extreme_vars - obj_min) / (obj_max - obj_min)
            ax.scatter(extreme_vars[:, 0], extreme_vars[:, 1], color='m', s=200, label='Extreme points')

        if final_population is not None:
            final_vars = final_population.extract_obj()
            obj_min = np.min(final_vars, axis=0)
            obj_max = np.max(final_vars, axis=0)
            final_vars = (final_vars - obj_min) / (obj_max - obj_min)
            ax.scatter(final_vars[:, 0], final_vars[:, 1], color='blue', s=75, label='Non-dominated Front')

        if infill_population is not None:
            infill_vars = np.atleast_2d(infill_population.extract_obj())
            obj_min = np.min(infill_vars, axis=0)
            obj_max = np.max(infill_vars, axis=0)
            infill_vars = (infill_vars - obj_min) / (obj_max - obj_min)
            ax.scatter(infill_vars[:, 0], infill_vars[:, 1], color='red', s=25, label='Knee-point Selection')

        if plot_pf:
            pareto_front = self.problem.variables['x_vars'][0].f_opt
            ax.plot(pareto_front[:, 0], pareto_front[:, 1], '-k')

        # ax.set_title(problem + ' ' + str(dim) + 'D, Distance-based')
        ax.set_aspect('equal')
        ax.set_xlim((-0.2, 1.2))
        ax.set_ylim((-0.2, 1.2))
        plt.legend(loc='best', frameon=False)
        plt.show()
        # plt.pause(3)
        # plt.close()
        # plt.savefig('/home/juan/PycharmProjects/optimisation_framework/multi_obj/results/zdt1_mmode_gen_' + str(len(self.surrogate.population)) + '.png')

    def knee_point_selection(self, population, n_survive=1):

        # Determine extreme points of Pareto
        obj_values = population.extract_obj()

        # Normalise objectives to [0, 1]
        obj_min = np.min(obj_values, axis=0)
        obj_max = np.max(obj_values, axis=0)
        obj_values = (obj_values - obj_min) / (obj_max - obj_min)

        # max_indices = (population.extract_crowding() >= 1e10) & (population.extract_rank() == 0)
        max_indices = np.argmax(obj_values, axis=0)
        extreme_points = obj_values[max_indices]

        # Add extreme points to survived population
        survived_population = population[max_indices]

        # Coefficients of hyperplane
        b = np.ones(len(extreme_points))
        w = np.linalg.solve(extreme_points, b)
        d = -np.dot(w, extreme_points[0])

        # Determination of type of PF (negative is convex / positive is concave)
        indicator = np.zeros(len(population))
        for i in range(len(population)):
            indicator[i] = np.dot(w, obj_values[i]) + d

        # Normal distance between points and hyperplane
        norm_dist = np.zeros(len(population))
        norm_w = np.linalg.norm(w)
        for idx in range(len(population)):
            norm_dist[idx] = np.abs(np.dot(obj_values[idx], w) + d) / norm_w
        norm_dist *= np.sign(indicator)

        # Split objective 1 domain into n_survive segments
        segments = np.linspace(0.0, 1.0, n_survive + 1)
        y_line = -(w[0] * segments + d) / w[1]
        bin_points = np.vstack((segments, y_line)).T

        # Calculate normal hyperplane normal vector
        w_norm = extreme_points[-1] - extreme_points[0]  # Normal coefficients to hyperplane

        # Dot product of objectives and the normal hyperplane
        products = np.zeros(len(population))
        for idx in range(len(population)):
            products[idx] = np.dot(w_norm, obj_values[idx])

        # Loop through individual segments
        bin_indices = []
        d_values = []
        for i in range(n_survive):
            # Intercepts of normal hyperplanes
            d1 = -np.dot(w_norm, bin_points[i])
            d2 = -np.dot(w_norm, bin_points[i+1])
            d_values.append([d1, d2])

            # Determine above or below hyperplane for population
            indicator1 = products + d1
            indicator2 = products + d2

            # Store indices that lie within the bin created by the two nearest hyperplanes
            bool_array = (indicator1 < 0) & (indicator2 >= 0)
            bin_indices.append(bool_array)

        # Loop through segments
        # survived_population = Population(self.problem, 0)
        for cntr in range(n_survive):
            # mask = (obj_values[:, 0] >= segments[cntr]) & (obj_values[:, 0] < segments[cntr + 1])
            mask = bin_indices[cntr]

            # Knee Point Selection (smallest negative distance from hyperplane)
            n_individuals = np.count_nonzero(mask)
            if n_individuals == 1:
                survived_population = Population.merge(survived_population, population[mask])
            elif n_individuals > 1:
                # # Select the point with the largest perpendicular distance to hyperplane   MMODE1
                # knee_point_indices = np.argpartition(norm_dist[mask], 1)[:1]

                # # Select at random from bin    MMODE 2
                # knee_point_indices = np.random.randint(0, np.count_nonzero(mask), 1)

                # Select at best-random from bin    MMODE 3
                rand_indices = np.random.randint(0, np.count_nonzero(mask), 3)
                knee_point_indices = np.argpartition(norm_dist[mask][rand_indices], 1)[:1]

                survived_population = Population.merge(survived_population,
                                                       population[mask][rand_indices][knee_point_indices])

        # # # DEBUG PURPOSES ONLY
        # y_lines = []
        # for i in range(len(d_values)):
        #     ds = d_values[i]
        #     l1 = -(w_norm[0]*segments + ds[0]) / w_norm[1]
        #     l2 = -(w_norm[0]*segments + ds[1]) / w_norm[1]
        #     y_lines.append(l1)
        #     y_lines.append(l2)
        # self.plot_pf(population, survived_population, population[max_indices], [segments, y_line], [segments, y_lines], plot_pf=False)

        return survived_population

    def bi_reference_point_selection(self, population, n_survived, points_per_ref=1):
        # Extract Objectives
        obj_array = population.extract_obj()

        # Select how many points to be survived per reference point
        n_obj = len(obj_array[0, :])
        n_ref = int(n_survived / points_per_ref / n_obj)
        intervals = np.linspace(0.0, 1.0, n_ref+1)[:-1]
        reference_points = np.zeros((n_obj*n_ref, n_obj))

        # Normalise Objectives
        cntr = 0
        for idx in range(n_obj):
            # Create Reference Point vector
            reference_points[cntr:cntr+n_obj, idx] = intervals
            cntr += n_obj

            # Normalise Objectives
            max_f = np.max(obj_array[:, idx], axis=0)
            min_f = np.min(obj_array[:, idx], axis=0)
            obj_array[:, idx] = (obj_array[:, idx] - min_f) / (max_f - min_f)

        # Find the closest points in PF to reference vectors
        survived_indices = None
        for i in range(len(reference_points)):
            dist = distance.cdist(np.atleast_2d(reference_points[i]), obj_array)[0]

            # Set a large distance if points already selected
            if survived_indices is not None:
                dist[survived_indices] = 1e9
            indices = np.argpartition(dist, points_per_ref)[:points_per_ref]

            # Store array of indices
            if survived_indices is not None:
                survived_indices = np.hstack((survived_indices, indices))
            else:
                survived_indices = indices

        # # Plot Check
        # fig, ax = plt.subplots(1, 1, figsize=(9, 7))
        # ax.scatter(obj_array[:, 1], obj_array[:, 0], color='black', marker='o', s=60, label='Offspring Population')
        # ax.scatter(obj_array[survived_indices, 1], obj_array[survived_indices, 0], color='cyan', marker='+', s=60, label='Reference Point Ranking')
        # ax.scatter(reference_var[:, 1], reference_var[:, 0], color='red', marker='o', s= 100, label='Reference Points')
        # ax.set_xlabel('Distance from Current Best (Negative)')
        # ax.set_ylabel('Surrogate Prediction')
        # ax.legend(loc='upper right')
        # plt.show()

        # Return Unique Survived Indices
        return np.unique(survived_indices)

    def rank_multiple_objectives(self, objectives_array):

        # Conduct ranking for each constraint
        rank_for_cons = np.zeros(cons_values.shape)
        for cntr in range(problem.n_con):
            cons_to_be_ranked = cons_values[:, cntr]
            fronts_to_be_ranked = NonDominatedSorting().do(cons_to_be_ranked.reshape((len(pop), 1)),
                                                           n_stop_if_ranked=len(pop))
            rank_for_cons[:, cntr] = self.rank_front_only(fronts_to_be_ranked, (len(pop)))
        rank_constraints = np.sum(rank_for_cons, axis=1)

        # Sum the normalised scores
        rank_objectives = np.sum(temp_obj, axis=1)

        # Simple Ranking
        objs_to_be_ranked = rank_objectives
        fronts_to_be_ranked = NonDominatedSorting().do(objs_to_be_ranked.reshape((len(temp_pop), 1)),
                                                       n_stop_if_ranked=len(temp_pop))
        rank_for_objs = self.rank_front_only(fronts_to_be_ranked, len(temp_pop))
        survivors = rank_for_objs.argsort()[:n_survive]

        # Survived population
        dummy_pop = Population(self.problem, len(survivors))
        dummy_pop.assign_obj(temp_obj[survivors])
        dummy_pop.assign_var(self.problem, temp_var[survivors, :])

        return dummy_pop

    @staticmethod
    def rank_front_only(fronts, n_survive):

        cntr_rank = 1
        rank = np.zeros(n_survive)
        for k, front in enumerate(fronts):

            # Save rank and crowding to the individuals
            for j, i in enumerate(front):
                rank[i] = cntr_rank

            cntr_rank += len(front)

        return rank

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

    def pareto_classifier(self, ranked_population, infill_population):

        # Extract ranks and objectives
        obj_vars = ranked_population.extract_obj()
        rank_vals = ranked_population.extract_rank().flatten()

        # Normalise Objectives [0, 1]
        max_f = np.max(obj_vars, axis=0)
        min_f = np.min(obj_vars, axis=0)
        obj_vars = (obj_vars - min_f) / (max_f - min_f)

        # Virtual points of population
        non_dominated = obj_vars[rank_vals == 0]
        dominated = obj_vars[rank_vals > 0]
        dist_matrix = distance.cdist(dominated, non_dominated)

        virtual_obj = np.zeros((len(dominated), self.problem.n_obj))
        virtual_rank = -rank_vals[rank_vals > 0]
        for idx in range(len(dominated)):
            # Closest two points to non-dominated front
            indices = np.argpartition(dist_matrix[idx, :], 2)[:2]
            virtual_obj[idx] = self.reflect_transform_2d(non_dominated[indices[0]], non_dominated[indices[1]], dominated[idx])

        # Training Dataset
        training_vars = np.vstack((obj_vars, virtual_obj))
        training_obj = np.hstack((rank_vals, virtual_rank))
        # training_vars = obj_vars
        # training_obj = rank_vals
        # training_obj = self.bilog_transform(training_obj, beta=1.0)

        # Train Regressor
        # classifier = RBF(self.problem.n_obj, c=0.5, p_type='constant', kernel_type='cubic')
        # classifier.fit(training_vars, training_obj)

        # Train Classifier
        training_obj[training_obj >= 0.] = 1
        training_obj[training_obj < 0.] = 0
        # print(training_obj)
        # classifier = SVC(C=1.0, kernel='rbf', probability=True)
        # classifier.fit(training_vars, training_obj)

        # One-class Classifier
        infill_vars = (infill_population.extract_obj() - min_f) / (max_f - min_f)
        training_vars = np.vstack((obj_vars, infill_vars))
        # training_labels = np.hstack((np.ones(len(obj_vars)), np.zeros(len(infill_population))))
        classifier = OneClassSVM(kernel='rbf', gamma=10.0, nu=0.01, shrinking=False)
        classifier.fit(obj_vars)

        # Predict Infill Points
        x_vars = infill_population.extract_obj()
        x_vars = (x_vars - min_f) / (max_f - min_f)
        class_labels = classifier.predict(x_vars)
        print(class_labels)

        # Visualisation Check
        if len(self.surrogate.population) > 0:
            fig, ax = plt.subplots(1, 1, figsize=(9, 7))
            fig.supxlabel('Obj 1', fontsize=14)
            fig.supylabel('Obj 2', fontsize=14)

            # Contour
            f1 = np.linspace(-0.5, 1.5, 100)
            f2 = np.linspace(-0.5, 1.5, 100)
            f11, f22 = np.meshgrid(f1, f2, indexing='ij')
            mesh = np.array((f11, f22)).T
            contour = np.zeros(np.shape(mesh[:, :, 0]))
            for i in range(np.shape(mesh)[0]):
                for j in range(np.shape(mesh)[1]):
                    contour[i, j] = classifier.predict(np.atleast_2d(mesh[i, j, :]))

            cmap = cm.get_cmap('Blues_r', 40)  # 'Blues_r'  'GnBu_r'  'PuBu_r'
            cmap = truncate_colormap(cmap, 0.0, 0.9)
            ax_surro = ax.contourf(f11, f22, contour.T, 40, cmap=cmap, label="Classifier")
            cbar = fig.colorbar(ax_surro)

            # Non-dominated Boundary
            ax_pf = ax.contour(f11, f22, contour.T, [0], colors='red', label="Non-dominated Boundary")

            # Old Points
            ax.scatter(obj_vars[:, 0], obj_vars[:, 1], color='blue', s=50, label='Population')
            # ax.scatter(virtual_obj[:, 0], virtual_obj[:, 1], color='m', s=50, label='Virtual')

            ax.scatter(obj_vars[rank_vals==0, 0], obj_vars[rank_vals==0, 1], color='red', s=50, label='Non-dominated Front')

            # New Points
            ax.scatter(x_vars[:, 0], x_vars[:, 1], color='black', s=50, label='Infill Points')

            plt.legend(loc='best', frameon=False)
            # plt.show()
            plt.savefig('/home/juan/PycharmProjects/optimisation_framework/multi_obj/results/pareto_classifier_one_svm_350.pdf')
            sys.exit(1)


        return class_labels

    @staticmethod
    def reflect_transform_2d(p1, p2, p3):
        """
        Householder transformation in R^2
        """
        # Vector P and Q
        Q = p2 - p1
        P = p3 - p1

        # reflection transformation
        X = p1 + (np.dot(P, Q) / np.dot(Q, Q)) * Q
        return (X - p3) * 2 + p3

    @staticmethod
    def bilog_transform(obj, beta=1.0):
        bilog_obj = np.zeros(np.shape(obj))

        for i in range(len(obj)):
            bilog_obj[i] = np.sign(obj[i]) * np.log(beta + np.abs(obj[i]))

        return bilog_obj


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    import matplotlib.colors as colors
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


class SurrogateSetup(Setup):

    def __init__(self, obj_surrogates, cons_surrogates=None,  use_constraints=False):
        super().__init__()

        self.obj_surrogates = obj_surrogates
        self.cons_surrogates = cons_surrogates
        self.use_constraints = use_constraints

    def set_variables(self, prob, parent_prob=None, **kwargs):

        # Copy variables from parent problem
        var_dict = extract_var_groups(parent_prob.variables)
        for i, key in enumerate(var_dict):
            prob.add_var_group(key, len(var_dict[key][0]), parent_prob.variables[key][0].type,
                               lower=var_dict[key][1], upper=var_dict[key][2],
                               value=var_dict[key][0], scale=parent_prob.variables[key][0].scale)

    def set_constraints(self, prob, parent_prob=None, **kwargs):

        if self.use_constraints:
            # Copy constraints from parent problem
            for key in parent_prob.constraints:
                prob.add_con(key)
        else:
            pass

    def set_objectives(self, prob, parent_prob=None, **kwargs):

        # Copy objectives from parent problem
        for key in parent_prob.objectives:
            prob.add_obj(key)

    def obj_func(self, x_dict, **kwargs):

        # Form design vector from input dict
        x = x_dict['x_vars']

        # Calculating objective function values
        obj = np.zeros(len(self.obj_surrogates))
        for i, model in enumerate(self.obj_surrogates):
            obj[i] = model.predict_model(x, 0)

        if self.cons_surrogates is not None and self.use_constraints:
            cons = np.ones(len(self.cons_surrogates))
            for i, model in enumerate(self.cons_surrogates):
                cons[i] = model.predict(x)
        else:
            cons = None

        performance = None

        return obj, cons, performance 

def extract_var_groups(vars):

    var_dict = OrderedDict()

    for i, key in enumerate(vars.keys()):

        # Extract variable values
        var_arr = np.zeros(len(vars[key]))
        lower_arr = np.zeros(len(vars[key]))
        upper_arr = np.zeros(len(vars[key]))
        for j in range(len(vars[key])):
            var_arr[j] = vars[key][j].value
            lower_arr[j] = vars[key][j].lower
            upper_arr[j] = vars[key][j].upper

        # Add variable to dict
        var_dict[key] = (var_arr, lower_arr, upper_arr)

        # De-scale variables
        if vars[key][0].type == 'c':
            var_dict[key] = var_dict[key]/vars[key][0].scale
        elif vars[key][0].type == 'i':
            var_dict[key] = round(var_dict[key]/vars[key][0].scale)
        elif vars[key][0].type == 'd':
            idx = np.round(var_dict[key]/vars[key][0].scale, 0).astype(int)
            var_dict[key] = np.asarray(vars[key][0].choices)[idx].tolist()

    return var_dict