import numpy as np
from scipy.spatial import distance
import copy
# import random
from collections import OrderedDict

from sklearn.cluster import KMeans

from mpi4py import MPI

# from optimisation.model.algorithm import Algorithm
from optimisation.optimise import minimise
from optimisation.model.problem import Problem
from optimisation.setup import Setup
from optimisation.algorithms.evolutionary_algorithm import EvolutionaryAlgorithm
from optimisation.algorithms.rvea import RVEA

from optimisation.surrogate.models.rbf_kernel_ensemble import RBFKernelEnsembleSurrogate
from optimisation.surrogate.models.ensemble import EnsembleSurrogate
from optimisation.surrogate.models.rbf import RadialBasisFunctions
from optimisation.surrogate.models.mars import MARSRegression
from optimisation.surrogate.models.rbf import RBF
from optimisation.surrogate.models.gp import GP

from optimisation.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from optimisation.operators.survival.rank_and_crowding_survival import RankAndCrowdingSurvival
from optimisation.operators.survival.adp_survival import APDSurvival
from optimisation.util.non_dominated_sorting import NonDominatedSorting
from optimisation.model.population import Population
from optimisation.model.repair import BasicBoundsRepair

from optimisation.util.reference_directions import UniformReferenceDirection
from optimisation.util.calculate_hypervolume import calculate_hypervolume
from optimisation.operators.survival.reference_direction_survival import ReferenceDirectionSurvival
from optimisation.util.misc import calc_V

from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib
plt.style.use('seaborn-talk')
np.set_printoptions(suppress=True)
matplotlib.use('TkAgg')
line_colors = ['green', 'blue', 'red', 'orange', 'cyan', 'lawngreen', 'm', 'orangered','sienna', 'gold', 'violet', 'indigo', 'cornflowerblue']


class K_RVEA(EvolutionaryAlgorithm):
    """
    K-RVEA: A Kriging-assisted Evolutionary Algorithm

    Publication: "A Surrogate-assisted Reference Vector Guided Evolutionary Algorithm for Computationally Expensive Many-objective
    Optimization"
    """

    def __init__(self,
                 ref_dirs,
                 n_population=None,
                 sampling=LatinHypercubeSampling(),
                 survival=None,
                 # survival=ReferenceDirectionSurvival(),
                 **kwargs):

        # Population parameters
        self.n_population = n_population

        # TODO: if throw-away surrogates, then will have to import GP and setup here
        # TODO: perhaps we can clear surrogate training data without completely going manual
        # # Surrogate strategy instance
        # self.surrogate = surrogate

        # Uniform reference vectors
        self.ref_dirs = ref_dirs

        # Sampling
        self.sampling = sampling

        # Survival
        self.survival = survival
        if self.survival is None:
            self.survival = APDSurvival(ref_dirs=self.ref_dirs)

        # Optimum position
        self.opt = None

        super().__init__(n_population=n_population,
                         sampling=sampling,
                         survival=survival,
                         n_offspring=n_population,
                         **kwargs)

    def _initialise(self):

        # Generation parameters
        self.max_f_eval = self.max_gen + self.surrogate.n_training_pts  # TODO: should be 300 FE + 11*n-1 training (outside?)
        print(self.max_f_eval)
        self.duplication_tolerance = 1e-2

        # K-RVEA Parameters
        self.n_population_kriging = 11 * self.problem.n_var - 1  # Reduced training size for Kriging Surrogate
        self.n_population_rvea = 50  # As per the paper
        self.n_infill = 5  # Number of infill points evaluated expensively every surrogate update
        self.w_max = 20  # Number of internal iterations for RVEA
        self.delta = 0.05 * len(self.ref_dirs)

        # Initialise from the evolutionary algorithm class
        super()._initialise()

        self.population = copy.deepcopy(self.surrogate.population)

        # Fixed and Adaptive Reference vectors
        self.V_adapt = calc_V(self.ref_dirs)
        self.V_fixed = copy.deepcopy(self.V_adapt)

        # Initialise two archives
        self.A1_archive = self.population
        self.A2_archive = self.population

    def _next(self):

        # 1. Surrogate-Assisted MODE exploitation ----------------------------------------------------------------------
        seed = np.random.randint(low=0, high=10000, size=(1,))

        # Create Setup Instance with Surrogate
        setup = SurrogateSetup(self.surrogate.obj_surrogates)

        # Create Problem instance
        opt_prob = Problem('surrogate_exploitation', obj_func=setup.obj_func, map_internally=False, n_processors=1)

        # Set variables, constraints & objectives
        setup.do(opt_prob, parent_prob=self.problem)

        # Instance of RVEA optimiser
        algorithm = RVEA(n_population=self.n_population_rvea,
                         max_gen=self.w_max,
                         ref_dirs=self.ref_dirs,  # TODO: should really be passing the self.V_adapt vectors here (kwargs ?)
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

        # Extract non-dominated individuals from final population / adaptive reference vectors
        final_population = algorithm.population
        self.V_adapt = algorithm.V  # TODO: Adapted on surrogate or on surrogate.population?

        # 2. Infill Criteria -------------------------------------------------------------------------------------------
        infill_population = self.infill_point_selection(final_population, n_survive=self.n_infill)
        print('n infill: ', len(infill_population))

        # Evaluate Infill Population expensively
        infill_population = self.evaluator.do(self.problem.obj_func, self.problem, infill_population)

        # 3. Update Archives --------------------------------------------------------------------------------
        self.A1_archive = self.manage_surrogate_archive(infill_population, n_maintain=self.n_population_kriging, seed=seed)
        self.A2_archive = self.survival.do(self.problem, self.surrogate.population, self.n_population)

        # 4. Surrogate Update ------------------------------------------------------------------------------------------
        self.surrogate.population = Population.merge(self.surrogate.population, infill_population)

        for obj_cntr in range(self.problem.n_obj):
            self.obj_surrogate = self.surrogate.obj_surrogates[obj_cntr]

            # Clear old data and use newly selected points
            self.obj_surrogate.reset()  # TODO: verify
            self.obj_surrogate.add_points(self.A1_archive.extract_var(), self.A1_archive.extract_obj().flatten())
            self.obj_surrogate.train()

    def infill_point_selection(self, population, n_survive=None):  # TODO
        # 1. Cluster active adaptive ref vectors into min(u, len(V_adapt)) clusters

        # 2. Individuals closest to active adaptive ref vectors in each cluster

        # 3. Assign I to fixed ref vectors and calculate num of inactive ref vectors |V_f^ia|

        # 4. Calculate delta {inactive fixed ref vectors from kth to k-1th}
        # if delta < self.delta
        # pick one individual with min ADP for each cluster
        # else
        # pick one individual with max Kriging uncertainty
        infill_population = None

        # TODO: move these to self.infill_point_selection ?
        # Duplicates within the exploit population
        is_duplicate = self.check_duplications(infill_population, other=None, epsilon=self.duplication_tolerance)
        infill_population = infill_population[np.invert(is_duplicate)[0]]

        # Duplicates with the surrogate population
        is_duplicate = self.check_duplications(infill_population, self.surrogate.population, epsilon=self.duplication_tolerance)
        infill_population = infill_population[np.invert(is_duplicate)[0]]

        return infill_population

    def manage_surrogate_archive(self, infill_population, n_maintain=None, seed=1):  # TODO
        if n_maintain is None:
            n_maintain = self.n_population_kriging

        # Filter individuals from surrogate population
        if len(self.surrogate.population) >= n_maintain:

            # Assign infill individuals to adaptive vectors
            infill_objectives = infill_population.extract_obj()
            active, inactive = self.split_by_vector_assignment(infill_objectives, self.V_adapt)

            # Assign entire population to inactive vectors
            assigned_to_inactive = None

            # If more individuals than allowed were selected
            if np.count_nonzero(assigned_to_inactive) > (n_maintain - len(infill_population)):
                pass
                # Do kmeans clustering of vectors?

            else:
                # Select randomnly via kmeans
                kmeans = KMeans(n_clusters=n_maintain - len(infill_population), random_state=seed)
                kmeans.fit(self.surrogate.population.extract_obj())
                cluster_labels = kmeans.labels_
                survived_indices = np.unique(cluster_labels)  # TODO: this should be a random selection within each cluster greater than 1
                training_data = self.surrogate.population[survived_indices]


            # Add new infill points to surrogate training set to achive required n_maintain points
            training_data = Population.merge(training_data, infill_population)

        # if self.A1_archive > self.n_population
        # assign u to self.V_adapt --> calculate inactive adaptive ref vectors V_a^ia
        # assign self.A1_archive \ u individuals to V_a^ia
        # Find active vectors now from the inactive adaptive set
        # Cluster set of inactive adaptive ref vectors V_a^ia into (self.n_population - self.n_infill) clusters
        # Select one individual from each cluster randomnly and remove the rest
        else:
            training_data = copy.deepcopy(self.surrogate.population)

        return training_data

    def split_by_vector_assignment(self, population, vectors):
        pass
        #




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