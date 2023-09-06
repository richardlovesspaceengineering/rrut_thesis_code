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

from optimisation.surrogate.surrogate_tuner import SurrogateTuner
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

        # Parameters
        self.n_infill = 1           # Number of infill points per generation
        self.n_offspring = 300      # TODO: Decide on an appropriate value
        self.w_max = 10             # Number of internal offspring generations

        # Initialise from the evolutionary algorithm class
        super()._initialise()

        # Miscellaneous
        self.n_init = self.evaluator.n_eval
        self.history_count = []

        # RBF regression surrogate hyperparameter tuner
        self.rbf_tuning_freq = int(self.max_gen / self.problem.n_var)
        self.rbf_tuning_frac = 0.20
        self.rbf_tuner = SurrogateTuner(n_dim=self.problem.n_var, lb=self.problem.x_lower, ub=self.problem.x_upper,
                                          problem=self.problem, c=0.5, p_type='linear', kernel_type='gaussian',
                                          width_range=(0.1, 10), train_test_split=self.rbf_tuning_frac,
                                          max_evals=75, verbose=False)

        # Minimisation Exploitation of RBF surrogates
        self.last_cluster = 0

        # Infill criterion parameters
        self.delta = 0.05  # % random variation in weighted sum

        # TODO: Move to main calling script
        if self.problem.n_obj == 2:
            self.n_ref_dirs = 20
        elif self.problem.n_obj == 3:
            self.n_ref_dirs = 6
        self.ref_dirs = UniformReferenceDirection(self.problem.n_obj, n_partitions=self.n_ref_dirs).do()
        self.cluster_range = np.arange(len(self.ref_dirs))

        # Initialise archives of non-dominated solutions
        self.opt = self.survive_best_front(self.population, return_fronts=False)
        self.representatives = self.opt
        igd = self.indicator.do(self.problem, self.opt, self.evaluator.n_eval, return_value=True)

        # Extract objective surrogates
        self.obj_surrogates = self.surrogate.obj_surrogates
        self.surrogate_evaluator = Evaluator()
        self.surrogate_strategy = self.surrogate

        # Results Output
        self.filename = f"{self.problem.name.lower()}_{self.save_name}_maxgen_{round(self.max_f_eval)}_sampling_" \
                        f"{self.surrogate_strategy.n_training_pts}_seed_{self.surrogate_strategy.sampling_seed}"
        self.data_extractor = GenerationExtractor(filename=self.filename, base_path=self.output_path)
        self.data_extractor.add_generation(self.population, self.n_gen)
        self.data_extractor.add_front(self.opt, self.n_gen, indicator_values=np.array(igd))

    def _next(self):
        # Update normalisation bounds
        self.update_norm_bounds(self.population)

        # Generate offspring for w_max internal iterations
        self.offspring = self.generate_offspring(self.representatives, n_offspring=self.n_offspring, n_gen=self.w_max)

        # Conduct final selection for infill points
        infill_population = self.infill_selection(self.offspring, n_survive=self.n_infill)

        # Evaluate infill points expensively and merge with population
        infill = self.evaluator.do(self.problem.obj_func, self.problem, infill_population)

        # Update optimum front
        old_pop = copy.deepcopy(self.population)
        old_reps = copy.deepcopy(self.representatives)
        self.population = Population.merge(self.population, infill)
        self.opt, fronts = self.survive_best_front(self.population, return_fronts=True)
        self.representatives = self.opt

        # Hyperparameter tuning of RBF widths (for each objective surrogate)
        if (self.n_gen % self.rbf_tuning_freq) == 0:
            self.regressor_tuning(fronts)
        else:
            # Update surrogate models
            self.update_regressor(infill_population)

        # Calculate Performance Indicator
        igd = self.indicator.do(self.problem, self.opt, self.evaluator.n_eval, return_value=True)
        print('IGD: ', np.round(igd, 4))
        self.data_extractor.add_generation(infill, self.n_gen)
        self.data_extractor.add_front(self.opt, self.n_gen, indicator_values=np.array(igd))

        # Keep track of n_infill per generation
        self.history_count.append(self.evaluator.n_eval - self.n_init)

        # DEBUG PURPOSES ONLY
        # if self.evaluator.n_eval % 25 == 0:
        #     self.plot_pareto(ref_vec=self.ref_dirs, scaling=1, labels=['Reps', 'Pop', 'Infill'],
        #                      obj_array1=old_reps.extract_obj(),
        #                      obj_array2=old_pop.extract_obj(),
        #                      obj_array3=infill.extract_obj())

    def generate_offspring(self, representatives, n_offspring=300, n_gen=10):
        survived = copy.deepcopy(representatives)
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
            survived = self.survive_best_front(offspring, return_fronts=False)

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

    def infill_selection(self, population, n_survive=1):
        # Extract offspring, eliminate duplicates and normalise objectives
        offspring = copy.deepcopy(population)
        offspring = self.eliminate_duplicates.do(offspring, self.population)
        obj_array = offspring.extract_obj()
        obj_array = (obj_array - self.ideal) / (self.nadir - self.ideal)

        # Update reference vector counter
        self.last_cluster, self.cluster_range = self.select_cluster(self.last_cluster, self.cluster_range,
                                                                    counter='skip2')
        # self.last_cluster += 2
        # if self.last_cluster >= len(self.ref_dirs):
        #     Cycle between odd and even ref_dirs
            # if self.last_cluster % 2 == 0:
            #     self.last_cluster = 1
            # else:
            #     self.last_cluster = 0

        # Select and perturb reference direction
        noise = np.random.uniform(-1.0, 1.0, self.problem.n_obj) * self.delta
        selected_weight = self.ref_dirs[self.last_cluster] + noise * 0.5
        selected_weight = selected_weight / np.sum(selected_weight)  # Ensure summation equals 1
        print(colored('Selected Cluster: ', 'blue'), self.last_cluster, colored('Ref Dir: ', 'blue'),
              self.ref_dirs[self.last_cluster], colored('--> ', 'blue'), selected_weight)

        # Select minimum of weighted sum of ref_dir and predicted objective values
        weighted_obj = np.sum(obj_array * selected_weight, axis=1)
        idx = np.argpartition(weighted_obj, n_survive)[:n_survive]
        survived = offspring[idx]

        if len(survived) == 0:
            # Select randomly if no infill point was found
            rand_int = np.random.randint(0, len(self.offspring), 1)
            survived = copy.deepcopy(self.offspring[rand_int])
            print(colored('--------- No infill point was found! Selecting randomly from offspring --------', 'yellow'))

        return survived

    def regressor_tuning(self, fronts):
        print(colored('----------------- Hyper-tuning Regressor Surrogate Widths ------------------', 'green'))
        # Conduct optimisation of each objective surrogate
        self.obj_surrogates = self.rbf_tuner.do(problem=self.problem,
                                                population=self.population,
                                                n_obj=len(self.obj_surrogates),
                                                fronts=fronts)
        print(colored('----------------------------------------------------------------------------', 'green'))

    def update_regressor(self, infill):
        # Create update training data
        new_vars = np.atleast_2d(infill.extract_var())
        new_obj = np.atleast_2d(infill.extract_obj())

        # Re-initialise training data and fit model
        for obj_cntr in range(self.problem.n_obj):
            self.obj_surrogates[obj_cntr].add_points(new_vars, new_obj[:, obj_cntr].flatten())
            try:
                self.obj_surrogates[obj_cntr].train()
            except:
                breakpoint()

    def update_norm_bounds(self, population):
        # Extract objectives
        obj_array = population.extract_obj()

        # Find lower and upper bounds
        f_min = np.min(obj_array, axis=0)
        f_max = np.max(obj_array, axis=0)

        # Update the ideal and nadir points
        self.ideal = np.minimum(f_min, self.ideal)
        self.nadir = f_max

    def select_cluster(self, last_idx, cluster_range, counter='skip2'):
        n_len = len(cluster_range)

        # Previous iteration cluster index
        c_ind = cluster_range[last_idx]

        # Loop through until a new cluster index is identified
        while c_ind == cluster_range[last_idx]:
            if 'random' in counter:
                index = random.randint(0, n_len-1)
            elif 'sequential' in counter:
                index = last_idx + 1
                if index == n_len:
                    index = 0
            elif 'seq_perm' in counter:
                index = last_idx + 1
                if index == n_len:
                    index = 0
                    np.random.shuffle(cluster_range)
            elif 'skip2' in counter:
                index = last_idx + 2
                if index >= n_len:
                    np.random.shuffle(cluster_range)
                    # Cycle between odd and even ref_dirs
                    if index % 2 == 0:
                        index = 1
                    else:
                        index = 0
            else:
                raise Exception(f"Counter strategy {counter} not implemented!")
            c_ind = cluster_range[index]

        return c_ind, cluster_range

    @staticmethod
    def survive_best_front(population, return_fronts=False):
        population = copy.deepcopy(population)
        fronts = NonDominatedSorting().do(population.extract_obj(), return_rank=False)
        survived = population[fronts[0]]
        if return_fronts:
            return survived, fronts

        return survived

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
