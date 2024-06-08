import copy
import random
import numpy as np

from scipy.spatial import distance

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
matplotlib.use('TkAgg')
line_colors = ['green', 'blue', 'red', 'orange', 'cyan', 'lawngreen', 'm', 'orangered','sienna', 'gold', 'violet', 'indigo', 'cornflowerblue']


class tDEADP(EvolutionaryAlgorithm):
    """
    t_DEA_DP algorithm: REGRESSION BASED
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
        self.n_train_max = 11 * self.problem.n_var + 24

        # Initialisation of Neural-Nets for dominance predictions
        self.obj_surrogates = self.surrogate_strategy.obj_surrogates
        self.initialise_surrogates()
        self.update_clusters()

        # Initialise Archive of non-dominated solutions
        self.opt = copy.deepcopy(self.population)
        self.representatives = self.survival.do(self.problem, self.population, len(self.ref_dirs), first_rank_only=False)

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

        # Duplicates with the surrogate population
        is_duplicate = self.check_duplications(infill_population, self.population, epsilon=self.duplication_tolerance)
        infill_population = infill_population[np.invert(is_duplicate)[0]]

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

        # Update Surrogates
        self.update_surrogates(infill_population)

        # DEBUG PLOT
        # self.plot_pareto(ref_vec=self.ref_dirs, scaling=3,
        #                  obj_array2=old_population.extract_obj(),
        #                  obj_array3=infill_population.extract_obj())

    def initialise_surrogates(self):
        # Create initial training data
        training_vars = self.population.extract_var()
        training_obj = self.population.extract_obj()

        # Assign training data to surrogate and train
        for i, surrogate in enumerate(self.obj_surrogates):
            surrogate.add_points(training_vars, training_obj[:, i].flatten())
            surrogate.train()

    def update_surrogates(self, archive):
        # Create updated training data
        # archive = self.population[-self.n_train_max:]
        training_vars = np.atleast_2d(archive.extract_var())
        training_obj = np.atleast_2d(archive.extract_obj())

        # Update surrogate by clearing the data
        for i, surrogate in enumerate(self.obj_surrogates):
            # surrogate.reset()
            surrogate.add_points(training_vars, training_obj[:, i].flatten())
            surrogate.train()

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
                max_prob_mask = np.argsort(prob)[:self.q_max]
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
        infill_ind = np.array([np.argmin(edn)])

        return offspring[infill_ind][:n_survive]

    def predict_dominance_pairs(self, offspring, representatives, flag=None):
        n = len(offspring)
        m = len(representatives)
        training_labels = np.ones((n, m), dtype=np.int)
        training_proba = np.ones((n, m))
        for i in range(n):
            for j in range(m):
                # Evaluate surrogate predicted objective values and calculate labels and prediction confidence
                label, proba = self.predict_dominance(offspring[i].var, representatives[j].var, domination=flag)

                # Assign labels and probabilities
                training_labels[i, j] = label
                training_proba[i, j] = proba

        return training_labels, training_proba

    def predict_offspring_pairs(self, offspring, flag=None):
        # Create prediction variables
        n = len(offspring)
        training_labels = np.zeros((n, n), dtype=np.int)
        training_proba = np.ones((n, n))
        for i in range(n):
            for j in range(i+1, n):
                # Predict labels for pair-wise comparisons
                label, proba = self.predict_dominance(offspring[i].var, offspring[j].var, domination=flag)

                # Assign labels and probabilities
                training_labels[i, j] = label
                training_labels[j, i] = (label if label == 0 else 3 - label)  # Should be inverted
                training_proba[i, j] = proba
                training_proba[j, i] = proba

        return training_labels, training_proba

    def predict_dominance(self, offs_vars, rep_vars, domination=None, return_proba=True):
        # Surrogate predictions
        offspring_objectives = np.zeros(len(self.obj_surrogates))
        rep_objectives = np.zeros(len(self.obj_surrogates))

        # Uncertainty predictions
        offspring_uncertainties = np.zeros(len(self.obj_surrogates))
        rep_uncertainties = np.zeros(len(self.obj_surrogates))

        # Evaluate for all objectives
        for i, surrogate in enumerate(self.obj_surrogates):
            # Mean predictions
            offspring_objectives[i] = surrogate.predict(offs_vars)
            rep_objectives[i] = surrogate.predict(rep_vars)

            # Uncertainty predictions (IDW Distance uncertainty)
            offspring_uncertainties[i] = surrogate.predict_idw(offs_vars, delta=2)
            rep_uncertainties[i] = surrogate.predict_idw(rep_vars, delta=2)

        # Calculate labels and probabilities
        if 'pareto' in domination:
            label = get_relation(offspring_objectives, rep_objectives)
        elif 'theta' in domination:  # TODO
            # 2d arrays
            offspring_objectives = np.atleast_2d(offspring_objectives)
            rep_objectives = np.atleast_2d(rep_objectives)

            # PBI Distances and cluster id
            clust_i, d1_i, d2_i = cluster_association(offspring_objectives, self.ref_dirs)
            clust_j, d1_j, d2_j = cluster_association(rep_objectives, self.ref_dirs)
            clust_i, pos = extract_value(clust_i)
            d1_i, d2_i = d1_i[pos], d2_i[pos]
            clust_j, pos = extract_value(clust_j)
            d1_j, d2_j = d1_j[pos], d2_j[pos]

            # Theta relation
            pbi_i = calc_pbi_func(d1_i[0], d2_i[0], self.theta)
            pbi_j = calc_pbi_func(d1_j[0], d2_j[0], self.theta)
            label = get_theta_relation(clust_i, clust_j, pbi_i, pbi_j)

        else:
            raise Exception('Specified surrogate is not recognised')

        # Modify
        proba = np.sum(offspring_uncertainties) + np.sum(rep_uncertainties)  # TODO:
        if label == -1:
            label = 2

        if return_proba:
            return label, proba

        return label

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


def extract_value(array):
    for idx, item in enumerate(array):
        if len(item) > 0:
            return item, idx
    return None
