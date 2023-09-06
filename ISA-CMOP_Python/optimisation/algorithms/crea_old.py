import copy
import random
import numpy as np
from scipy.spatial import distance

# from keras.utils import to_categorical
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE

from optimisation.model.population import Population
from optimisation.algorithms.evolutionary_algorithm import EvolutionaryAlgorithm

from optimisation.operators.mutation.de_mutation import DifferentialEvolutionMutation
from optimisation.operators.sampling.random_sampling import RandomSampling
from optimisation.operators.selection.random_selection import RandomSelection
from optimisation.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from optimisation.operators.mutation.polynomial_mutation import PolynomialMutation
from optimisation.operators.survival.theta_survival import ThetaSurvival
from optimisation.operators.survival.rank_and_crowding_survival import RankAndCrowdingSurvival
from optimisation.model.duplicate import DefaultDuplicateElimination
from optimisation.model.repair import BasicBoundsRepair, BounceBackBoundsRepair
from optimisation.metrics.indicator import Indicator

from optimisation.surrogate.models.fnn_regressor_keras import DeepNeuralNetRegression
from optimisation.surrogate.classifiers.fnn_classifier_keras import DeepNeuralNetClassification
# from optimisation.surrogate.classifiers.fnn_classifier_pytorch import DeepNeuralNetClassification
from optimisation.surrogate.classifiers.xgboost_classifier import XGBoostClassification
from optimisation.surrogate.classifiers.catboost_classifier import CatBoostClassification
from optimisation.surrogate.classifiers.lightgbm_classifier import LightGBMClassification
from optimisation.surrogate.models.rbf import RadialBasisFunctions

# TODO: Move static methods from theta survival class to optimsation/utils/
from optimisation.util.calculate_hypervolume import calculate_hypervolume
from optimisation.util.hyperplane_normalisation import HyperplaneNormalisation
from optimisation.operators.survival.theta_survival import cluster_association, calc_pbi_func
from optimisation.util.non_dominated_sorting import NonDominatedSorting
from optimisation.util.dominator import get_relation
from optimisation.util.misc import calc_gamma, calc_V

import matplotlib.pyplot as plt
import matplotlib

plt.style.use('seaborn-talk')
np.set_printoptions(suppress=True)
matplotlib.use('TkAgg')
line_colors = ['green', 'blue', 'red', 'orange', 'cyan', 'lawngreen', 'm', 'orangered', 'sienna', 'gold', 'violet',
               'indigo', 'cornflowerblue']


class CREA(EvolutionaryAlgorithm):
    """
    CREA algorithm: Classification and Regression Evolutionary Algorithm
    """

    def __init__(self,
                 ref_dirs=None,
                 n_population=109,
                 surrogate=None,
                 sampling=RandomSampling(),
                 selection=RandomSelection(),
                 crossover=SimulatedBinaryCrossover(eta=30, prob=1.0),
                 mutation=PolynomialMutation(eta=20, prob=None),
                 eliminate_duplicates=DefaultDuplicateElimination(),
                 **kwargs):

        self.ref_dirs = ref_dirs
        self.surrogate_strategy = surrogate
        self.indicator = Indicator(metric='igd')
        self.repair = BasicBoundsRepair()

        # Classifier Parameters
        if 'classifier_params' in kwargs:
            self.classifier_params = kwargs['classifier_params']
        else:
            raise Exception('No classifier parameters given!')

        if 'classifier_type' in kwargs:
            self.classifier_type = kwargs['classifier_type'].lower()
            assert self.classifier_type in ['neural', 'xgboost', 'catboost', 'lightgbm']
        else:
            raise Exception('No classifier type was specified!')

        # PBI penalty
        if 'theta' in kwargs:
            self.theta = kwargs['theta']
        else:
            self.theta = 5.0

        # Have to define here given the need to pass ref_dirs
        if 'survival' in kwargs:
            survival = kwargs['survival']
            del kwargs['survival']
        else:
            survival = ThetaSurvival(ref_dirs=self.ref_dirs, theta=self.theta, filter_infeasible=True)
            # survival = RankAndCrowdingSurvival(filter_infeasible=True)

        super().__init__(n_population=n_population,
                         sampling=sampling,
                         selection=selection,
                         crossover=crossover,
                         mutation=mutation,
                         survival=survival,
                         eliminate_duplicates=eliminate_duplicates,
                         **kwargs)

    def _initialise(self):
        # Initialise from the evolutionary algorithm class
        super()._initialise()

        # Generation parameters
        self.max_f_eval = self.max_gen
        print('max_eval: ', self.max_f_eval)
        self.duplication_tolerance = 1e-2

        # Reference Vectors
        self.norm = HyperplaneNormalisation(self.ref_dirs.shape[1])
        self.ideal = np.full(self.problem.n_obj, np.inf)
        self.nadir = np.full(self.problem.n_obj, -np.inf)

        # MODE Offspring Parameters
        self.f = [0.8, 1.0, 1.0, 0.8, 0.8, 0.9, 0.4, 0.4]
        self.cr = [0.2, 0.1, 0.9, 0.8, 0.6, 0.2, 0.2, 0.9]
        self.create_offspring = DifferentialEvolutionMutation(F=self.f, Cr=self.cr, problem=self.problem)

        # t-DEA-DP Parameters & Neural-net initialisation
        self.oversampler = None
        self.last_cluster_index = -1
        self.cluster_range = np.arange(len(self.ref_dirs))
        self.clusters, self.d1_mat, self.d2_mat = None, None, None  # Theta parameters
        self.angle_mat, self.gamma_apd, self.alpha = None, None, 2.0  # APD parameters
        self.adapt_fr = 0.1  # APD vector frequency of adaptation
        self.n_init = self.evaluator.n_eval
        self.max_infill = self.max_gen - self.n_init
        self.n_infill = 1
        self.n_offspring = 7000
        self.q_max = 300
        self.gamma = 0.9
        self.epoch_init = 20
        self.n_train_max = 11 * self.problem.n_var + 24

        # Offspring generation flag
        self.offspring_flag = 'sbx_poly'  # 'sbx_poly', 'mode'

        if 'neural' in self.classifier_type:
            self.classifier_params = {
                "n_dim": self.problem.n_var,
                "n_outputs": len(self.ref_dirs),
                "n_layers": 3,
                "n_neurons": [20, 20, len(self.ref_dirs)],
                "activation_kernel": ['relu', 'relu', 'softmax'],  # 'sigmoid', 'softmax', None
                "weight_decay": 0.001,
                "lr": 0.01,
                "batch_size": 32,
                "epoch": 100,
                "class_weights": None,
            }
        elif 'catboost' in self.classifier_type:
            self.classifier_params = {
                "n_dim": self.problem.n_var,
                "loss_func": 'MultiClass',
                "epoch": 500,
                "task_type": "CPU",  # 'CPU' / 'GPU'
                "depth": 11,
                "lr": 0.17,
                "l2_leaf_reg": 3.0,
                "min_data_in_leaf": 35,
                "random_strength": 0.03,
                "border_count": 200,
                "colsample_bylevel": 0.65,
                "class_weights": None,
            }

        # Regression Neural Net
        self.regressor_type = 'rbf'  # 'neural', 'rbf'
        if 'neural' in self.regressor_type:
            self.regressor_params = {
                "n_dim": self.problem.n_var,
                "n_outputs": 1,
                "n_layers": 2,
                "n_neurons": [30, 1],
                "activation_kernel": ['relu', None],  # 'sigmoid', 'softmax', None
                "weight_decay": 0.001,
                "lr": 0.01,
                "batch_size": 32,
                "epoch": 100,
                "class_weights": None,
            }
        elif 'rbf' in self.regressor_type:
            self.regressor_params = {
                "c": 0.5,
                "p_type": 'linear',
                "kernel_type": 'cubic',
            }

        # Initialisation of surrogates
        self.classifier = self.initialise_classifier()
        self.regressor = self.initialise_regressor()

        # Initialise archives of non-dominated solutions
        fronts = NonDominatedSorting().do(self.population.extract_obj(), return_rank=False)
        self.opt = copy.deepcopy(self.population[fronts[0]])
        self.representatives = self.survival.do(self.problem, self.population, len(self.ref_dirs),
                                                first_rank_only=False)

    def _next(self):

        # Select clusters
        self.update_clusters()

        # Generate offspring and selection points for expensive evaluation
        self.offspring = self.generate_offspring(n_gen=10, n_offspring=300, nc=1, flag=self.offspring_flag)

        # Select infill points for expensive evaluation
        infill_population = self.infill_selection(self.offspring, n_survive=self.n_infill)

        # Select randomly if no infill point was found
        if len(infill_population) == 0:
            rand_int = np.random.randint(0, len(self.offspring), 1)
            infill_population = copy.deepcopy(self.offspring[rand_int])
            print('-----------------No infill point was found! Selecting randomly from offspring ------------------')

        # Evaluate infill point expensively
        infill_population = self.evaluator.do(self.problem.obj_func, self.problem, infill_population)

        # Merge the offspring with the current population
        old_population = copy.deepcopy(self.population)
        self.population = Population.merge(self.population, infill_population)

        # Truncate population for offspring generation
        old_reps = copy.deepcopy(self.representatives)
        self.representatives = self.survival.do(self.problem, self.population, len(self.ref_dirs),
                                                first_rank_only=False)

        # Update optimum (non-dominated solutions)
        fronts = NonDominatedSorting().do(self.population.extract_obj(), return_rank=False)
        old_opt = copy.deepcopy(self.opt)
        self.opt = copy.deepcopy(self.population[fronts[0]])

        # Calculate Performance Indicator
        igd = self.indicator.do(self.problem, self.opt, return_value=True)
        print(f"IGD: {igd:.4f}")

        # Update Neural-Net surrogates if necessary
        self.update_classifier()
        self.update_regressor()

        # DEBUG PLOT
        if self.evaluator.n_eval == 149 or self.evaluator.n_eval == 249 or self.evaluator.n_eval > 355:
            self.plot_pareto(ref_vec=self.ref_dirs, scaling=1, labels=['Reps', 'Pop', 'Infill'],
                             obj_array1=old_reps.extract_obj(),
                             obj_array2=old_population.extract_obj(),
                             obj_array3=infill_population.extract_obj())

    def initialise_classifier(self):
        # Initialise classifier class
        if 'neural' in self.classifier_type:
            classifier = DeepNeuralNetClassification(l_b=self.problem.x_lower, u_b=self.problem.x_upper,
                                                     **self.classifier_params)
        elif 'catboost' in self.classifier_type:
            classifier = CatBoostClassification(l_b=self.problem.x_lower, u_b=self.problem.x_upper,
                                                **self.classifier_params)
        elif 'xgboost' in self.classifier_type:
            classifier = XGBoostClassification(l_b=self.problem.x_lower, u_b=self.problem.x_upper,
                                               **self.classifier_params)
        elif 'lightgbm' in self.classifier_type:
            classifier = LightGBMClassification(l_b=self.problem.x_lower, u_b=self.problem.x_upper,
                                                **self.classifier_params)
        else:
            classifier = self.surrogate_strategy.obj_surrogates[0]

        # Create initial training data
        training_data, class_weights = self.prepare_decomposition_data(self.population)

        # Assign class weights (imbalanced classes)
        if 'neural' in self.classifier_type:
            self.classifier_params["class_weights"] = class_weights
        else:
            self.classifier_params["class_weights"] = self.assign_class_weights(training_data, flag=self.classifier_type)
        classifier.class_weights = self.classifier_params["class_weights"]

        # Assign training data to neural net and train
        classifier.add_points(training_data[0], training_data[1])
        classifier.train()

        return classifier

    def update_classifier(self):
        training_data, class_weights = self.prepare_decomposition_data(self.population)

        # Re-assign class weights to classifier
        if 'neural' in self.classifier_type:
            self.classifier_params["class_weights"] = class_weights
        else:
            self.classifier_params["class_weights"] = self.classifier_params(training_data, flag=self.classifier_type)
        self.classifier.class_weights = self.classifier_params["class_weights"]

        # Re-initialise training data and fit model
        self.classifier.reset()
        self.classifier.add_points(training_data[0], training_data[1])
        self.classifier.train()

    def initialise_regressor(self):
        # Initialise classifier class
        if 'neural' in self.regressor_type:
            regressor = DeepNeuralNetRegression(l_b=self.problem.x_lower, u_b=self.problem.x_upper,
                                                **self.regressor_params)
        elif 'rbf' in self.regressor_type:
            regressor = RadialBasisFunctions(self.problem.n_var, l_b=self.problem.x_lower, u_b=self.problem.x_upper,
                                             **self.regressor_params)
        else:
            regressor = self.surrogate_strategy.obj_surrogates[1]

        # Create initial training data
        training_data = self.prepare_indicator_data(self.population)

        # Assign training data to neural net and train
        regressor.add_points(training_data[0], training_data[1])
        regressor.train()

        return regressor

    def update_regressor(self):
        # Create update training data
        training_data = self.prepare_indicator_data(self.population)

        # Re-initialise training data and fit model
        self.regressor.reset()
        self.regressor.add_points(training_data[0], training_data[1])
        self.regressor.train()

    def generate_offspring(self, n_gen=5, n_offspring=300, nc=4, flag='sbx_poly'):
        # # Select a cluster index using: 'random', 'sequential', 'seq_perm' strategies
        # c_ind, cluster_indices = self.select_cluster(counter=counter)

        # Begin offspring generations from the current representatives
        survived = copy.deepcopy(self.representatives)

        # Internal generations of offspring generation
        for idx in range(n_gen - 1):
            # print('offs gen: ', idx)
            # Generate offspring for specified size
            if 'sbx_poly' in flag:
                offspring = self.mating.do(self.problem, survived, n_offspring)
            elif 'mode' in flag:
                offspring = self.create_offspring.do(self.opt, self.representatives)
                offspring = self.repair.do(self.problem, offspring)
            else:
                raise Exception(f"Offspring strategy {flag} is not recognised!")

            # DEBUG ---------------------------------------------------------------------
            # offspring = self.evaluator.do(self.problem.obj_func, self.problem, offspring)

            # Predict decomposition of offspring
            survived = self.infill_selection(offspring, n_survive=nc)

        # Final offspring of size n_offspring
        # print(len(survived))
        if 'sbx_poly' in flag:
            survived = self.mating.do(self.problem, survived, self.n_offspring)
        elif 'mode' in flag:
            survived = self.create_offspring.do(self.opt, self.representatives)
            survived = self.repair.do(self.problem, survived)

        # TODO: DEBUG ---------------------------------------------------------------------
        # survived = self.evaluator.do(self.problem.obj_func, self.problem, survived)
        # self.plot_pareto(ref_vec=self.ref_dirs, scaling=1, labels=['Offs', 'Reps', 'Pop'],
        #                  obj_array1=survived.extract_obj(),
        #                  obj_array2=self.representatives.extract_obj(),
        #                  obj_array3=self.population.extract_obj())
        # TODO: DEBUG ---------------------------------------------------------------------

        return survived

    def infill_selection(self, offspring, n_survive=1):
        # Predict decomposition of offspring
        off_vars = offspring.extract_var()
        cluster_labels = self.classifier.predict(off_vars)

        # Predict Hypervolume of offspring in cluster
        hypervolume = self.regressor.predict(off_vars).flatten()

        # For each cluster, find best points
        survived = Population(self.problem, 0)
        for i in np.unique(cluster_labels):
            indices = i == cluster_labels
            if np.count_nonzero(indices) > n_survive:
                # Select point/s with highest hypervolume value
                hp = hypervolume[indices]
                best_indices = np.argpartition(-hp, n_survive)[:n_survive]
                survived = Population.merge(survived, offspring[indices][best_indices])
            elif np.count_nonzero(indices) == n_survive:
                survived = Population.merge(survived, offspring[indices])

        # TODO: DEBUG ---------------------------------------------------------------------
        # fig, ax = plt.subplots(1, 1, figsize=(9, 7))
        # fig.supxlabel('Obj 1', fontsize=14)
        # fig.supylabel('Obj 2', fontsize=14)
        # for i in range(len(np.unique(cluster_labels))):
        #     indices = i == cluster_labels
        #     if np.count_nonzero(indices) > 1:
        #         c_offs = offspring[indices].extract_obj()
        #         hp = hypervolume[indices] * 100
        #         ax.scatter(c_offs[:, 0], c_offs[:, 1], s=hp, color=line_colors[i], label=f"clust: {i}")
        # reps = self.representatives.extract_obj()
        # ax.scatter(reps[:, 0], reps[:, 1], color='k', label='Reps')
        # surv = survived.extract_obj()
        # ax.scatter(surv[:, 0], surv[:, 1], marker='x', color='r', s=100, label='Survived')
        # plt.legend(loc='upper right')
        # plt.show()
        # TODO: DEBUG ---------------------------------------------------------------------

        return survived

    def prepare_decomposition_data(self, population):
        n = len(population)

        # Update Clusters
        self.update_clusters()

        # Assign population to clusters
        training_vars = np.zeros((n, self.problem.n_var))
        training_labels = np.zeros(n, dtype=int)

        count = 0
        weights = []
        for i, indices in enumerate(self.clusters):
            n_in_clust = len(indices)
            if n_in_clust > 0:
                weights.append(n_in_clust)
                for idx in indices:
                    training_vars[count] = population[idx].var
                    training_labels[count] = i
                    count += 1

        # Check if clusters have sufficient neighbours to perform over-sampling balancing of classes
        k_n = min(weights)-1
        if k_n > 1:
            # Conduct SMOTE Over-sampling
            self.oversampler = SMOTE(k_neighbors=k_n)
            training_vars, training_labels = self.oversampler.fit_resample(training_vars, training_labels)
            class_weights = dict(enumerate(np.ones(len(weights))))
        else:
            # Calculate class weights for imbalanced training
            n_class = len(weights)
            class_weights = dict(enumerate(n / (n_class * np.array(weights))))

        # Training data
        training_data = [training_vars, training_labels]

        # print('n_clusters: ', len(weights))

        return training_data, class_weights

    def prepare_indicator_data(self, population):
        # calculate hypervolume contribution of each point
        hypervol = self.calc_hypervol_contrib(population, normalise=True)

        # Form training data set
        training_x = population.extract_var()
        training_y = hypervol
        training_data = [training_x, training_y]

        return training_data

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

    def update_clusters(self):
        # Extract the objective function values from the population
        obj_array = self.population.extract_obj()

        # calculate the ideal and nadir points of the entire population
        f_min = np.min(obj_array, axis=0)
        f_max = np.max(obj_array, axis=0)
        self.ideal = np.minimum(f_min, self.ideal)
        self.nadir = f_max

        # TODO: Normalise population
        obj_array = (obj_array - self.ideal) / (self.nadir - self.ideal)

        # Assign clusters to population and calculate distances
        self.clusters, self.d1_mat, self.d2_mat = cluster_association(obj_array, self.ref_dirs)

    def select_cluster(self, counter='seq_perm'):
        n_len = len(self.cluster_range)

        # Previous iteration cluster index
        c_ind = self.cluster_range[self.last_cluster_index]

        # Loop through until a new cluster index is identified
        index = 0
        while c_ind == self.cluster_range[self.last_cluster_index]:
            if 'random' in counter:
                index = random.randint(0, n_len - 1)

            elif 'sequential' in counter:
                index = self.last_cluster_index + 1
                if index == n_len:
                    index = 0

            elif 'seq_perm' in counter:
                index = self.last_cluster_index + 1
                if index == n_len:
                    index = 0
                    np.random.shuffle(self.cluster_range)
            else:
                raise Exception(f"Counter strategy {counter} not implemented!")
            c_ind = self.cluster_range[index]

        # Assign new cluster and extract cluster indices
        self.last_cluster_index = index
        cluster_indices = np.array(self.clusters[c_ind])
        print('Using cluster index: ', c_ind)

        return c_ind, cluster_indices

    def best_from_cluster(self, flag=None):
        best_per_cluster = [[] for _ in range(len(self.clusters))]

        # For each of the individuals in the clusters find the one with the smallest pbi distance
        for c_ind, niche in enumerate(self.clusters):

            # if niche is not empty
            if len(niche) > 0:

                if 'theta' in flag:
                    # Calculate PBI
                    d1, d2 = self.d1_mat[c_ind], self.d2_mat[c_ind]
                    pbi = calc_pbi_func(d1, d2, np.array([self.theta]))
                    rep = niche[np.argmin(pbi)]
                    best_per_cluster[c_ind].append(rep)

                elif 'apd' in flag:
                    # Calculate APD
                    d1, angle = self.d1_mat[c_ind], self.angle_mat[c_ind]
                    apd = calc_apd_func(d1, angle, self.gamma_apd[c_ind], (self.evaluator.n_eval - self.n_init),
                                        self.max_infill, self.problem.n_obj, self.alpha)
                    rep = niche[np.argmin(apd)]
                    best_per_cluster[c_ind].append(rep)

        # Extract the best point per cluster
        best_reps = np.array([item[0] for item in best_per_cluster if item != []])

        return best_per_cluster, best_reps

    def adapt(self):
        # Re-calculate ideal and nadir points
        obj_arr = self.population.extract_obj()
        self.ideal = np.minimum(obj_arr.min(axis=0), self.ideal)
        self.nadir = obj_arr.max(axis=0)

        # Scale old vectors
        self.V = calc_V(calc_V(self.ref_dirs) * (self.nadir - self.ideal))

        # Pass onto survival class
        self.survival.V = copy.deepcopy(self.V)

    @staticmethod
    def assign_class_weights(training_data, flag=None):
        # Training data
        y_train = training_data[1]

        # # Reverse one-hot encoding
        # y_train = np.argmax(y_train, axis=1)

        # (Neural Nets)
        if 'neural' in flag:
            # Assign class weights
            n_total = len(y_train)
            n_class_1 = np.count_nonzero(y_train == 0)  # Non-dominated
            n_class_2 = np.count_nonzero(y_train == 1)  # Dominates
            n_class_3 = np.count_nonzero(y_train == 2)  # Dominated

            try:
                # Incase any class has no members
                weights = n_total / (3 * n_class_1), n_total / (3 * n_class_2), n_total / (3 * n_class_3)
            except ZeroDivisionError:
                return None

            # Return as a dictionary
            class_weights = dict(enumerate(weights))

        # Gradient Boosting
        elif 'xgboost' in flag or 'catboost' in flag or 'lightgbm' in flag:
            # Assign weight to each individual label (sklearn)
            class_weights = class_weight.compute_sample_weight('balanced', y_train)

        else:
            class_weights = None

        return class_weights

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


def find_cluster_index(clusters, ind):
    for c_index, items in enumerate(clusters):
        try:
            pos = items.index(ind)
            return c_index, pos
        except ValueError:
            continue


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


def calculate_theta_domination_matrix(obj_array, clusters, d1_mat, d2_mat, theta=5.0):
    # Number of individuals
    n = obj_array.shape[0]

    # Output matrix
    m = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            # Extract relevant cluster id and calculate pbi distance
            cluster_i, pos_i = find_cluster_index(clusters, i)
            cluster_j, pos_j = find_cluster_index(clusters, j)

            # Extract distances
            d1_i = d1_mat[cluster_i][pos_i]
            d2_i = d2_mat[cluster_i][pos_i]
            d1_j = d1_mat[cluster_j][pos_j]
            d2_j = d2_mat[cluster_j][pos_j]

            # Calculate PBI values
            pbi_i = calc_pbi_func(d1_i, d2_i, theta)
            pbi_j = calc_pbi_func(d1_j, d2_j, theta)

            # Theta domination
            m[i, j] = get_ref_vec_relation(cluster_i, cluster_j, pbi_i, pbi_j)
            m[j, i] = -m[i, j]

    return m


def calculate_apd_domination_matrix(obj_array, clusters, d1_mat, angle_mat, gamma, gen, max_gen, alpha=2.0):
    # Number of individuals
    n = obj_array.shape[0]
    n_obj = obj_array.shape[1] if obj_array.shape[1] > 2 else 1.0

    # Output matrix
    m = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            # Extract relevant cluster id and calculate pbi distance
            cluster_i, pos_i = find_cluster_index(clusters, i)
            cluster_j, pos_j = find_cluster_index(clusters, j)

            # Extract distances
            d1_i = d1_mat[cluster_i][pos_i]
            ang_i = angle_mat[cluster_i][pos_i]
            d1_j = d1_mat[cluster_j][pos_j]
            ang_j = angle_mat[cluster_j][pos_j]

            # Calculate APD values
            P_i = n_obj * ((gen / max_gen) ** alpha) * (ang_i / gamma[cluster_i])
            P_j = n_obj * ((gen / max_gen) ** alpha) * (ang_j / gamma[cluster_j])

            # calculate the angle-penalized penalized (APD)
            apd_i = d1_i * (1 + P_i)
            apd_j = d1_j * (1 + P_j)

            # APD domination
            m[i, j] = get_ref_vec_relation(cluster_i, cluster_j, apd_i, apd_j)
            m[j, i] = -m[i, j]

    return m


def apd_association(obj_arr, ref_dirs, ideal):
    # Update from new reference vectors
    ref_vec = calc_V(ref_dirs)
    gamma = calc_gamma(ref_vec)

    # store the ideal and nadir point estimation for adapt - (and ideal for transformation)
    ideal = np.minimum(obj_arr.min(axis=0), ideal)

    # translate the population to make the ideal point the origin
    obj_arr = obj_arr - ideal

    # the distance to the ideal point
    dist_to_ideal = np.linalg.norm(obj_arr, axis=1)
    dist_to_ideal[dist_to_ideal < 1e-64] = 1e-64

    # normalize by distance to ideal
    obj_prime = obj_arr / dist_to_ideal[:, None]

    # calculate for each solution the acute angles to ref dirs
    acute_angle = np.arccos(obj_prime @ ref_vec.T)

    # Indices of niches
    niches = acute_angle.argmin(axis=1)

    # Initialise clusters
    clusters = [[] for _ in range(len(ref_vec))]
    d1_distances = [[] for _ in range(len(ref_vec))]
    angle_distances = [[] for _ in range(len(ref_vec))]

    # Assign niches to clusters
    for k, i in enumerate(niches):
        clusters[i].append(k)
        d1_distances[i].append(dist_to_ideal[k])
        angle_distances[i].append(acute_angle[k, i])

    return clusters, d1_distances, angle_distances, gamma


def calc_apd_func(dist, angle, gamma, gen, max_gen, n_obj, alpha=2.0):
    # Angle Penalty
    penalty = n_obj * ((gen / max_gen) ** alpha) * (angle / gamma)

    # calculate the angle-penalized penalized (APD)
    apd = dist * (1 + penalty)

    return apd
