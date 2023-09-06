import copy
import random
import numpy as np

# from keras.utils import to_categorical

from optimisation.model.population import Population
from optimisation.algorithms.evolutionary_algorithm import EvolutionaryAlgorithm

from optimisation.operators.sampling.random_sampling import RandomSampling
from optimisation.operators.selection.random_selection import RandomSelection
from optimisation.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from optimisation.operators.mutation.polynomial_mutation import PolynomialMutation
from optimisation.operators.survival.theta_survival import ThetaSurvival
from optimisation.operators.survival.rank_and_crowding_survival import RankAndCrowdingSurvival
from optimisation.model.duplicate import DefaultDuplicateElimination

from optimisation.metrics.indicator import Indicator

# TODO: Move static methods from theta survival class to optimsation/utils/
from optimisation.util.hyperplane_normalisation import HyperplaneNormalisation
from optimisation.operators.survival.theta_survival import cluster_association, calc_pbi_func
from optimisation.util.dominator import calculate_domination_matrix, get_relation
from optimisation.util.non_dominated_sorting import NonDominatedSorting

import matplotlib.pyplot as plt
import matplotlib
plt.style.use('seaborn-talk')
np.set_printoptions(suppress=True)
# matplotlib.use('TkAgg')
line_colors = ['green', 'blue', 'red', 'orange', 'cyan', 'lawngreen', 'm', 'orangered','sienna', 'gold', 'violet', 'indigo', 'cornflowerblue']


class tDEADP(EvolutionaryAlgorithm):
    """
    t_DEA_DP algorithm:
    Yuan2022 "Expensive Multiobjective Evolutionary Optimization Assisted by Dominance Prediction"
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

        # Classifier Parameters
        if 'classifier_params' in kwargs:
            self.classifier_params = kwargs['classifier_params']
        else:
            raise Exception('No classifier parameters given!')

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

        # t-DEA-DP Parameters & Neural-net initialisation
        self.last_cluster_index = -1
        self.cluster_range = np.arange(len(self.ref_dirs))
        self.clusters, self.d1_mat, self.d2_mat = None, None, None
        self.n_infill = 1
        self.n_offspring = 7000
        self.q_max = 300
        self.gamma = 0.9
        self.epoch_init = 20
        self.n_train_max = 11 * self.problem.n_var + 24

        # Initialisation of Neural-Nets for dominance predictions
        self.training_data = dict()
        self.pareto_surrogate = self.initialise_neural_net(flag='pareto')
        self.theta_surrogate = self.initialise_neural_net(flag='theta')

        # Initialise Archive of non-dominated solutions
        self.opt = copy.deepcopy(self.population)
        self.representatives = self.survival.do(self.problem, self.population, len(self.ref_dirs), first_rank_only=False)
        # self.theta_archive = self.survival.do(self.population)

    def _next(self):

        # Conduct mating using the current population
        self.offspring = self.mating.do(self.problem, self.representatives, self.n_offspring)

        # Conduct two-stage pre-selection
        infill_population = self.two_stage_preselection(self.offspring, self.population, n_survive=self.n_infill)

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
        self.representatives = self.survival.do(self.problem, self.population, len(self.ref_dirs), first_rank_only=False)

        # Update optimum (non-dominated solutions)
        fronts = NonDominatedSorting().do(self.population.extract_obj(), return_rank=False)
        old_opt = copy.deepcopy(self.opt)
        self.opt = copy.deepcopy(self.population[fronts[0]])

        # Calculate Performance Indicator
        igd = self.indicator.do(self.problem, self.opt, return_value=True)
        print('IGD: ', np.round(igd, 4))

        # Update Clusters
        self.update_clusters()

        # Update Neural-Net surrogates if necessary
        self.update_neural_net(flag='pareto')
        self.update_neural_net(flag='theta')

        # DEBUG PLOT
        # self.plot_pareto(ref_vec=self.ref_dirs, scaling=3,
        #                  obj_array2=old_population.extract_obj(),
        #                  obj_array3=self.opt.extract_obj())

    def initialise_neural_net(self, flag=None):
        # Initialise neural-net classifier class
        if 'pareto' in flag:
            classifier = self.surrogate_strategy.obj_surrogates[0]
        else:  # theta
            classifier = self.surrogate_strategy.obj_surrogates[1]

        # Create initial training data
        training_data = self.prepare_training_data(self.population, domination=flag)
        self.training_data[flag] = training_data

        # np.savetxt('/home/juan/PycharmProjects/optimisation_framework/multi_obj/results/boosting_dataset_pareto.txt',
        #         np.hstack((training_data[0], training_data[1][:, None])), delimiter=',')

        # Assign class weights (imbalanced training in cross-entropy loss)
        self.classifier_params["class_weights"] = self.assign_class_weights(training_data)
        classifier.class_weights = self.classifier_params["class_weights"]

        # Assign training data to neural net and train
        classifier.add_points(training_data[0], training_data[1])
        classifier.train()

        return classifier

    def update_neural_net(self, flag=None):
        # Create updated training data
        archive = self.population[-self.n_train_max:]
        training_data = self.prepare_comparison_data(archive, domination=flag)

        # Evaluate label predictions for determining model accuracy
        prediction = self.predict_dominance(training_data[0], domination=flag, return_proba=False)

        # Calculate model accuracy by pairwise predictions of latest infill point with n_train_max individuals
        pred_accuracy = self.predict_model_accuracy(prediction, training_data[1])
        worst_accuracy = np.min(pred_accuracy)
        print('                                                         Class accuracies: ', np.round(pred_accuracy, 3))

        # If above threshold, continue without updating model
        if worst_accuracy >= self.gamma:
            print(f"{flag}-net continuing without update")
            return

        # Update number of training epochs for model update
        self.classifier_params["epoch"] = int(np.ceil(self.epoch_init * (1 - worst_accuracy / self.gamma)))

        # Assign class weights (imbalanced training in cross-entropy loss)
        self.classifier_params["class_weights"] = self.assign_class_weights(training_data)

        if self.classifier_params["class_weights"] is None:
            print(f"{flag}-net was not updated because class_weights were None!")
            return

        # Update neural-net classifier surrogate
        if 'pareto' in flag:
            # Re-assign parameters to surrogate
            self.pareto_surrogate.epochs = self.classifier_params["epoch"]
            self.pareto_surrogate.class_weights = self.classifier_params["class_weights"]

            # Re-initialise training data and fit model
            self.pareto_surrogate.reset()
            self.pareto_surrogate.add_points(training_data[0], training_data[1])
            self.pareto_surrogate.train()
        elif 'theta' in flag:
            # Re-assign parameters to surrogate
            self.theta_surrogate.epochs = self.classifier_params["epoch"]
            self.theta_surrogate.class_weights = self.classifier_params["class_weights"]

            # Re-initialise training data and fit model
            self.theta_surrogate.reset()
            self.theta_surrogate.add_points(training_data[0], training_data[1])
            self.theta_surrogate.train()

    def two_stage_preselection(self, offspring, population, n_survive=1, counter='seq_perm'):
        # Obtain the best indices for each cluster
        best_ind_per_cluster, best_reps = self.best_from_cluster()

        # Select a cluster index using: 'random', 'sequential', 'seq_perm' strategies
        c_ind, cluster_indices = self.select_cluster(counter=counter)

        # Select from Q1-3 is cluster is not null
        if len(cluster_indices) > 0:
            # Obtain Theta and Pareto-representative individuals in cluster
            theta_rep, pareto_rep = self.obtain_representatives(population, c_ind, best_reps, best_ind_per_cluster)

            # Split up offspring into Q1, Q2 and Q3 categories
            categories = self.preselection_strategy_one(offspring, population[theta_rep], population[pareto_rep],
                                                        flag='Q123')
        else:
            # Select from Q5 is cluster is null
            theta_rep = best_reps
            pareto_rep = []
            # Offspring non-dominated with theta representatives
            categories = self.preselection_strategy_one(offspring, population[theta_rep], population[pareto_rep],
                                                        flag='Q5')

        # Conduct final selection for one infill point
        infill_population = self.preselection_strategy_two(categories, n_survive=n_survive)

        # Plot Debug
        # self.plot_pareto(ref_vec=self.ref_dirs, scaling=5.0,
        #                  obj_array1=population[theta_rep].extract_obj(),
        #                  obj_array2=population[pareto_rep].extract_obj(),
        #                  obj_array3=population.extract_obj())

        return infill_population

    def preselection_strategy_one(self, offspring, t_reps, p_reps, flag=None):
        if 'Q123' in flag:
            # Obtain prediction labels and probabilities for pareto and theta representative-offspring comparisons
            p_labels, p_proba = self.predict_dominance_pairs(offspring, p_reps, flag='pareto')
            t_labels, t_proba = self.predict_dominance_pairs(offspring, t_reps, flag='theta')

            # 1d arrays
            p_labels = p_labels.flatten()
            t_labels = t_labels.flatten()
            p_proba = p_proba.flatten()
            t_proba = t_proba.flatten()

            # Determine Q1, Q2 and Q3 categories
            mask_one = (p_labels == 1) & (t_labels == 1)
            mask_two = (p_labels == 0) & (t_labels == 1)
            mask_three = (p_labels == 1) & (t_labels == 0)

            # Assign offspring to categories
            Q1, Q2, Q3 = offspring[[mask_one]], offspring[[mask_two]], offspring[[mask_three]]

            # Assign combined probabilities to offspring
            Q1_proba = p_proba[mask_one] + t_proba[mask_one]
            Q2_proba = p_proba[mask_two] + t_proba[mask_two]
            Q3_proba = p_proba[mask_three] + t_proba[mask_three]

            # Combine categories
            cat = [Q1, Q2, Q3]
            cat_proba = [Q1_proba, Q2_proba, Q3_proba]

        # Else assign offspring to category Q5
        elif 'Q5' in flag:

            # Obtain prediction labels and probabilities for theta representative-offspring comparisons
            t_labels, t_proba = self.predict_dominance_pairs(offspring, t_reps, flag='theta')

            # Find number of non-dominated individuals for each offspring
            n_non_dominated = np.count_nonzero(t_labels == 0, axis=1)

            # TODO: if maximum number of non-dominated individuals by the offspring is smaller than the representative
            # TODO: individuals then return nothing??
            max_val = np.max(n_non_dominated)
            if max_val < len(t_reps):
                return [[], [], []]

            # Otherwise return the indices with the maximum number of nondominated solutions
            indices = np.where(n_non_dominated == max_val)[0]
            sum_proba = np.sum(t_proba, axis=1).flatten()

            # Combine categories
            cat = [offspring[[indices]], [], []]
            cat_proba = [sum_proba[indices], [], []]

        # Return a maximum of Qmax offspring
        for i, prob in enumerate(cat_proba):
            if len(prob) == 0:
                cat[i] = prob
            else:
                max_prob_mask = np.argsort(-prob)[:self.q_max]
                cat[i] = cat[i][max_prob_mask]
                cat_proba[i] = prob[max_prob_mask]

        return cat

    def preselection_strategy_two(self, categories, n_survive=1):
        # Extract Qmax solutions in order of Q1 to Q3 or Q5
        if len(categories[0]) > 0:
            offspring = categories[0]
        elif len(categories[1]) > 0:
            offspring = categories[1]
        elif len(categories[2]) > 0:
            offspring = categories[2]
        else:
            offspring = Population(self.problem, 0)

        # Return nothing if no offspring are found within the categories
        if len(offspring) <= 1:
            return offspring

        # Predict labels and probabilities of offspring in selected categories
        p_labels, p_proba = self.predict_offspring_pairs(offspring, flag='pareto')
        t_labels, t_proba = self.predict_offspring_pairs(offspring, flag='theta')

        # Assign Expected Dominance Number (EDN)
        p_proba[p_labels != 1] = 0
        t_proba[t_labels != 1] = 0
        edn = np.sum(p_proba, axis=1) + np.sum(t_proba, axis=1)

        # Select one individual with the highest EDN value
        infill_ind = np.array([np.argmax(edn)])

        return offspring[infill_ind][:n_survive]

    def predict_dominance_pairs(self, offspring, representatives, flag=None):
        # Create prediction variables
        n = len(offspring)
        m = len(representatives)
        training_vars = np.zeros((n * m, 2 * self.problem.n_var))
        idx = 0
        for i in range(n):
            for j in range(m):
                training_vars[idx] = np.hstack((offspring[i].var, representatives[j].var))
                idx += 1

        # Predict labels for pair-wise comparisons
        labels, proba = self.predict_dominance(training_vars, domination=flag, return_proba=True)

        # Reshape into matrices
        training_labels = np.ones((n, m), dtype=np.int)
        training_proba = np.ones((n, m))
        idx = 0
        for i in range(n):
            for j in range(m):
                training_labels[i, j] = labels[idx]
                training_proba[i, j] = proba[idx]
                idx += 1

        return training_labels, training_proba

    def predict_offspring_pairs(self, offspring, flag=None):
        # Create prediction variables
        n = len(offspring)
        training_vars = np.zeros((int(n * (n-1) / 2), 2 * self.problem.n_var))
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                training_vars[idx] = np.hstack((offspring[i].var, offspring[j].var))
                idx += 1

        # Predict labels for pair-wise comparisons
        labels, proba = self.predict_dominance(training_vars, domination=flag, return_proba=True)

        # Reshape into matrices
        # TODO: Check if .reshape does the same job faster
        training_labels = np.zeros((n, n), dtype=np.int)
        training_proba = np.ones((n, n))
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                training_labels[i, j] = labels[idx]
                training_labels[j, i] = (labels[idx] if labels[idx] == 0 else 3 - labels[idx])  # Should be inverted
                training_proba[i, j] = proba[idx]
                training_proba[j, i] = proba[idx]
                idx += 1

        return training_labels, training_proba

    def predict_dominance(self, vars, domination=None, return_proba=True):
        # Prepare data for evaluation
        n = vars.shape[1] // 2
        uv_pairs = vars
        vu_pairs = np.hstack((vars[:, n:], vars[:, :n]))

        # Calculate labels and probabilities
        if 'pareto' in domination:
            # Pairs [u, v]
            uv_labels = self.pareto_surrogate.predict(uv_pairs)
            uv_probas = self.pareto_surrogate.predict_proba(uv_pairs)

            # Inverted pairs [v, u]
            vu_labels = self.pareto_surrogate.predict(vu_pairs)
            vu_probas = self.pareto_surrogate.predict_proba(vu_pairs)

        elif 'theta' in domination:
            # Pairs [u, v]
            uv_labels = self.theta_surrogate.predict(uv_pairs)
            uv_probas = self.theta_surrogate.predict_proba(uv_pairs)

            # Inverted pairs [v, u]
            vu_labels = self.theta_surrogate.predict(vu_pairs)
            vu_probas = self.theta_surrogate.predict_proba(vu_pairs)
        else:
            raise Exception('Specified surrogate is not recognised')

        # collapse probabilities
        uv_probas = np.max(uv_probas, axis=1)
        vu_probas = np.max(vu_probas, axis=1)

        # Initialise outputs
        labels = copy.deepcopy(uv_labels)
        probas = np.maximum(uv_probas, vu_probas)

        # Select the corresponding label/ probabilities with highest probabilities from the [u, v] and [v, u]
        # prediction pairs
        mask_p = uv_probas > vu_probas
        mask_n = vu_labels == 0  # Non-dominated label

        # [u,v] predicts higher p than [v,u] : t(u,v)
        labels[mask_p] = uv_labels[mask_p]

        # [v,u] predicts higher p than [u,v] & not non-dom : 3 - t(v,u)
        mask = ~mask_p & ~mask_n
        labels[mask] = 3 - vu_labels[mask]

        # [v,u] predicts higher p than [u,v] & non-dom : 0
        mask = ~mask_p & mask_n
        labels[mask] = 0

        if return_proba:
            return labels, probas

        return labels

    def prepare_comparison_data(self, population, domination=None):
        if domination is None:
            return None

        # Extract objective values from population
        obj_array = population.extract_obj()

        if 'theta' in domination:
            # Normalise objectives
            f_min = np.min(obj_array, axis=0)
            f_max = np.max(obj_array, axis=0)
            ideal = np.minimum(f_min, self.ideal)
            nadir = f_max

            # TODO: Normalise Objectives
            # obj_array = (obj_array - ideal) / (nadir - ideal)

            # Assign clusters to population and calculate distances
            clusters, d1_mat, d2_mat = cluster_association(obj_array, self.ref_dirs)

            # Calculate Theta Domination matrix (-1: dominated, 1: dominates, 0: non-dominated)
            dom_mat = calculate_theta_domination_matrix(obj_array, clusters, d1_mat, d2_mat, theta=self.theta)

        elif 'pareto' in domination:
            # Calculate Pareto Domination matrix (-1: dominated, 1: dominates, 0: non-dominated)
            dom_mat = calculate_domination_matrix(obj_array)
        else:
            raise Exception('Unrecognised dominance relation')

        # Construct training data arrays
        n = len(population)
        training_vars = np.zeros((2 * (n-1), 2 * self.problem.n_var))
        training_labels = np.zeros(2 * (n-1), dtype=int)

        idx = 0
        for i in range(n-1):
            for j in range(n-1, n):
                # Assign individual [u, v] and domination label to training data
                label = dom_mat[i, j]
                training_labels[idx] = label
                training_vars[idx] = np.hstack((population[i].var, population[j].var))
                idx += 1

                # Assign individual [v, u] and domination label to training data
                label = dom_mat[j, i]
                training_labels[idx] = label
                training_vars[idx] = np.hstack((population[j].var, population[i].var))
                idx += 1

        # Convert labels to one-hot encoded vector
        training_labels[training_labels == -1] = 2
        # training_one_hot = to_categorical(training_labels)
        # training_one_hot = training_one_hot.astype(int)
        training_data = [training_vars, training_labels]

        return training_data

    def prepare_training_data(self, population, domination=None):
        if domination is None:
            return None

        # Extract objective values from population
        obj_array = population.extract_obj()

        if 'theta' in domination:
            # Update Clusters
            self.update_clusters()

            # Calculate Theta Domination matrix (-1: dominated, 1: dominates, 0: non-dominated)
            dom_mat = calculate_theta_domination_matrix(obj_array, self.clusters, self.d1_mat, self.d2_mat, theta=self.theta)

        elif 'pareto' in domination:
            # Calculate Pareto Domination matrix (-1: dominated, 1: dominates, 0: non-dominated)
            dom_mat = calculate_domination_matrix(obj_array)
        else:
            raise Exception('Unrecognised dominance relation')

        # Initialise arrays
        n = len(population)
        training_vars = np.zeros((n * (n - 1), 2 * self.problem.n_var))
        training_labels = np.zeros(n * (n - 1), dtype=int)

        # Construct training dataset of n * (n - 1) pair-wise domination comparisons
        idx = 0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                # If not on diagonal, assign individual [u,v] and domination label to training data
                label = dom_mat[i, j]
                training_labels[idx] = label
                training_vars[idx] = np.hstack((population[i].var, population[j].var))
                idx += 1

        # Convert labels to one-hot encoded vector
        training_labels[training_labels == -1] = 2
        # training_one_hot = to_categorical(training_labels)
        # training_one_hot = training_one_hot.astype(int)
        training_data = [training_vars, training_labels]

        return training_data

    def update_clusters(self):
        # Extract the objective function values from the population
        obj_array = self.population.extract_obj()

        f_min = np.min(obj_array, axis=0)
        f_max = np.max(obj_array, axis=0)

        self.ideal = np.minimum(f_min, self.ideal)
        self.nadir = f_max

        # TODO: Normalise population
        # obj_array = (obj_array - self.ideal) / (self.nadir - self.ideal)

        # Assign clusters to population and calculate distances
        self.clusters, self.d1_mat, self.d2_mat = cluster_association(obj_array, self.ref_dirs)

    def obtain_representatives(self, population, c_ind, best_reps, best_ind_per_cluster):
        # Obtain theta-representative individual
        theta_rep = best_ind_per_cluster[c_ind]
        theta_nondominated = population[best_reps]

        # Obtain pareto-representative solution from theta individuals in cluster
        obj_arr = theta_nondominated.extract_obj()
        fronts = NonDominatedSorting().do(obj_arr, return_rank=False)
        pareto_nondominated = theta_nondominated[fronts[0]]
        if theta_rep in best_reps[fronts[0]]:
            pareto_rep = theta_rep
        else:
            pareto_rep = None
            min_dist = np.inf
            ref_to_cluster = self.ref_dirs[c_ind]
            # Find individual that pareto-dominates theta-rep with smallest angle between ref vectors
            for i in range(len(pareto_nondominated)):
                dom = get_relation(pareto_nondominated[i].obj, population[theta_rep[0]].obj)
                if dom == 1:
                    dist = np.sum((self.ref_dirs[i] - ref_to_cluster) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        pareto_rep = best_reps[i]
            # Return as a list
            pareto_rep = [pareto_rep]

        return theta_rep, pareto_rep

    def select_cluster(self, counter='seq_perm'):
        n_len = len(self.cluster_range)

        # Previous iteration cluster index
        c_ind = self.cluster_range[self.last_cluster_index]

        # Loop through until a new cluster index is identified
        index = 0
        while c_ind == self.cluster_range[self.last_cluster_index]:
            if 'random' in counter:
                index = random.randint(0, n_len-1)

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

    def best_from_cluster(self):
        best_per_cluster = [[] for _ in range(len(self.clusters))]

        # For each of the individuals in the clusters find the one with the smallest pbi distance
        for c_ind, niche in enumerate(self.clusters):

            # if niche is not empty
            if len(niche) > 0:

                # Calculate PBI
                d1, d2 = self.d1_mat[c_ind], self.d2_mat[c_ind]
                pbi = calc_pbi_func(d1, d2, np.array([self.theta]))
                theta_rep = niche[np.argmin(pbi)]
                best_per_cluster[c_ind].append(theta_rep)

        # Extract the best point per cluster
        best_reps = np.array([item[0] for item in best_per_cluster if item != []])

        return best_per_cluster, best_reps

    @staticmethod
    def assign_class_weights(training_data):
        # Training data
        y_train = training_data[1]

        # # Reverse one-hot encoding
        # y_train = np.argmax(y_train, axis=1)

        # Assign weights
        n_total = len(y_train)
        n_class_1 = np.count_nonzero(y_train == 0)   # Non-dominated
        n_class_2 = np.count_nonzero(y_train == 1)   # Dominates
        n_class_3 = np.count_nonzero(y_train == 2)   # Dominated

        # Incase any class has no members
        try:
            weights = n_total / (3 * n_class_1), n_total / (3 * n_class_2), n_total / (3 * n_class_3)
        except ZeroDivisionError:
            return None

        # Return as a dictionary
        class_weights_dict = dict(enumerate(weights))

        return class_weights_dict

    @staticmethod
    def predict_model_accuracy(predicted_labels, encoded_true_labels):

        accuracy = []
        # decoded_true_labels = np.argmax(encoded_true_labels, axis=1)  # Decoding one-hot vector data
        decoded_true_labels = encoded_true_labels
        for i in range(3):
            # Number of predicted labels i
            true_mask = decoded_true_labels == i
            pred_mask = predicted_labels[true_mask] == i

            frac_label = np.count_nonzero(pred_mask)

            # Number of true labels i
            total_label = np.count_nonzero(true_mask)

            # Accuracy for each class prediction
            if total_label == 0:
                accuracy.append(1.0)
            else:
                try:
                    # Fraction of correct labels between predicted and true labels
                    acc = frac_label / total_label
                except ZeroDivisionError:
                    acc = 0.0
                accuracy.append(acc)

        return accuracy

    @staticmethod
    def plot_pareto(ref_vec=None, obj_array1=None, obj_array2=None, obj_array3=None, fronts=None, scaling=1.5):

        n_obj = len(ref_vec[0])

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
                    ax.scatter(obj_array1[:, 0], obj_array1[:, 1], color=line_colors[0], s=150, label=f"theta_pop")

            try:
                if obj_array2 is not None:
                    obj_array2 = np.atleast_2d(obj_array2)
                    if fronts is not None:
                        for i, frnt in enumerate(fronts):
                            ax.scatter(obj_array2[frnt, 0], obj_array2[frnt, 1], color=line_colors[i], s=75, label=f"rank {i}")
                    else:
                        ax.scatter(obj_array2[:, 0], obj_array2[:, 1], color=line_colors[1], s=75, label=f"pareto_pop")
            except:
                pass

            if obj_array3 is not None:
                obj_array3 = np.atleast_2d(obj_array3)
                if fronts is not None:
                    for i, frnt in enumerate(fronts):
                        ax.scatter(obj_array3[frnt, 0], obj_array3[frnt, 1], color=line_colors[i], s=75, label=f"rank {i}")
                else:
                    ax.scatter(obj_array3[:, 0], obj_array3[:, 1], color=line_colors[2], s=25, label=f"infill")

            ax.set_xlim((-0.05, 1.05))

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
                                 label=f"theta_pop")

            if obj_array2 is not None:
                obj_array2 = np.atleast_2d(obj_array2)
                if fronts is not None:
                    for i, frnt in enumerate(fronts):
                        ax.scatter3D(obj_array2[frnt, 0], obj_array2[frnt, 1], obj_array2[frnt, 2], color=line_colors[i], s=50, label=f"rank {i}")
                else:
                    ax.scatter3D(obj_array2[:, 0], obj_array2[:, 1], obj_array2[:, 2], color=line_colors[1], s=25,
                                 label=f"pareto_pop")

            if obj_array3 is not None:
                obj_array3 = np.atleast_2d(obj_array3)
                if fronts is not None:
                    for i, frnt in enumerate(fronts):
                        ax.scatter3D(obj_array3[frnt, 0], obj_array3[frnt, 1], obj_array3[frnt, 2], color=line_colors[i], s=50, label=f"rank {i}")
                else:
                    ax.scatter3D(obj_array3[:, 0], obj_array3[:, 1], obj_array3[:, 2], color=line_colors[2], s=15,
                                 label=f"infill")

        plt.legend(loc='best', frameon=False)
        plt.show()
        # plt.savefig('/home/juan/PycharmProjects/optimisation_framework/multi_obj/results/zdt1_mmode_gen_' + str(len(self.surrogate.population)) + '.png')


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
            m[i, j] = get_theta_relation(cluster_i, cluster_j, pbi_i, pbi_j)
            m[j, i] = -m[i, j]

    return m


def get_theta_relation(c_i, c_j, d_i, d_j):
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


def find_cluster_index(clusters, ind):
    for c_index, items in enumerate(clusters):
        try:
            pos = items.index(ind)
            return c_index, pos
        except ValueError:
            continue

