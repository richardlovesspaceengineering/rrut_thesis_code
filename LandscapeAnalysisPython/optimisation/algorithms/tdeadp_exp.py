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

from optimisation.surrogate.classifiers.fnn_classifier_keras import DeepNeuralNetClassification
# from optimisation.surrogate.classifiers.fnn_classifier_pytorch import DeepNeuralNetClassification
from optimisation.surrogate.classifiers.xgboost_classifier import XGBoostClassification
from optimisation.surrogate.classifiers.catboost_classifier import CatBoostClassification
from optimisation.surrogate.classifiers.lightgbm_classifier import LightGBMClassification

# TODO: Move static methods from theta survival class to optimsation/utils/
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

        # Change the first and second dominance relation criterion
        self.dom1_flag = 'pareto'  # 'pareto', 'alpha', 'epsilon', 'nlad'
        self.dom2_flag = 'theta'  # 'theta', 'apd'

        self.diversity_params = {
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

        self.catboost_params = {
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

        # Initialisation of Neural-Nets for dominance predictions
        self.training_data = dict()
        self.dom1_surrogate = self.initialise_classifier(flag='dom1')
        self.dom2_surrogate = self.initialise_classifier(flag='div1')

        # Initialise Archives of non-dominated solutions
        fronts = NonDominatedSorting().do(self.population.extract_obj(), return_rank=False)
        self.opt = copy.deepcopy(self.population[fronts[0]])
        self.representatives = self.survival.do(self.problem, self.population, len(self.ref_dirs),
                                                first_rank_only=False)

        # Initialise list to count occurrences of Q123 and Q5
        self.category_count = [0, 0]

    def _next(self):

        # Conduct mating using the current population
        # old_offspring = self.generate_offspring(n_gen=10, n_offspring=200)
        self.offspring = self.generate_offspring(n_gen=5, n_offspring=300, flag=self.offspring_flag)
        # self.offspring = self.generate_offspring(n_gen=1, n_offspring=7000, flag=self.offspring_flag)
        # print(len(old_offspring), len(self.offspring))

        # self.evaluator.do(self.problem.obj_func, self.problem, self.offspring)
        # self.evaluator.do(self.problem.obj_func, self.problem, old_offspring)
        # self.plot_pareto(ref_vec=self.ref_dirs, scaling=3,
        #                  obj_array1=self.offspring.extract_obj(),
        #                  obj_array2=old_offspring.extract_obj(),
        #                  obj_array3=self.representatives.extract_obj(),
        #                  labels=['10_200', '5_300', 'Reps'])

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
        self.representatives = self.survival.do(self.problem, self.population, len(self.ref_dirs),
                                                first_rank_only=False)

        # Update optimum (non-dominated solutions)
        fronts = NonDominatedSorting().do(self.population.extract_obj(), return_rank=False)
        old_opt = copy.deepcopy(self.opt)
        self.opt = copy.deepcopy(self.population[fronts[0]])

        # Calculate Performance Indicator
        igd = self.indicator.do(self.problem, self.opt, return_value=True)
        print('IGD: ', np.round(igd, 4))

        # Update Clusters
        self.update_clusters(flag=self.dom2_flag)

        # Update Neural-Net surrogates if necessary
        self.update_classifier(flag='dom1')
        self.update_classifier(flag='div1')

        # DEBUG PLOT
        # if self.evaluator.n_eval == 300 or self.evaluator.n_eval == 110 or self.evaluator.n_eval == 150:
        #     self.plot_pareto(ref_vec=self.ref_dirs, scaling=1,
        #                      obj_array1=self.representatives.extract_obj(),
        #                      obj_array2=old_population.extract_obj(),
        #                      obj_array3=infill_population.extract_obj())

    def initialise_classifier(self, flag=None):
        # Initialise neural-net classifier class
        if 'dom1' in flag:
            classifier = self.surrogate_strategy.obj_surrogates[0]
        elif 'div1' in flag:
            # classifier = DeepNeuralNetClassification(l_b=self.problem.x_lower, u_b=self.problem.x_upper, **self.diversity_params)
            classifier = CatBoostClassification(l_b=self.problem.x_lower, u_b=self.problem.x_upper, **self.catboost_params)
        else:
            raise Exception('Specified surrogate does not exist')

        # Create initial training data
        if 'dom' in flag:
            training_data = self.prepare_training_data(self.population, domination=flag)
        elif 'div' in flag:
            training_data, class_weights = self.diversity_training_data(self.population)
        self.training_data[flag] = training_data

        # Assign class weights (imbalanced training in cross-entropy loss)
        if 'dom' in flag:
            self.classifier_params["class_weights"] = self.assign_class_weights(training_data, flag=self.classifier_type)
            classifier.class_weights = self.classifier_params["class_weights"]
        elif 'div' in flag:
            # self.diversity_params["class_weights"] = class_weights
            self.diversity_params["class_weights"] = self.assign_class_weights(training_data, flag="catboost")
            classifier.class_weights = self.diversity_params["class_weights"]

        # Assign training data to neural net and train
        classifier.add_points(training_data[0], training_data[1])
        classifier.train()

        return classifier

    def update_classifier(self, flag=None):
        # Create updated training data
        if 'dom' in flag:
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

            if 'neural' in self.classifier_type:
                # Update number of training epochs for model update
                self.classifier_params["epoch"] = int(np.ceil(self.epoch_init * (1 - worst_accuracy / self.gamma)))
                # self.classifier_params["epoch"] = self.epoch_init

        # Assign class weights (imbalanced training in cross-entropy loss)
        if 'dom' in flag:
            self.classifier_params["class_weights"] = self.assign_class_weights(training_data, flag=self.classifier_type)

            if self.classifier_params["class_weights"] is None:
                print(f"{flag}-net was not updated because class_weights were None!")
                return

        # Update neural-net classifier surrogate
        if 'dom1' in flag:
            # Re-assign parameters to surrogate
            if 'neural' in self.classifier_type:
                self.dom1_surrogate.epochs = self.classifier_params["epoch"]
            self.dom1_surrogate.class_weights = self.classifier_params["class_weights"]

            # Re-initialise training data and fit model
            self.dom1_surrogate.reset()
            self.dom1_surrogate.add_points(training_data[0], training_data[1])
            self.dom1_surrogate.train()
        elif 'div1' in flag:
            training_data, class_weights = self.diversity_training_data(self.population)

            # Re-assign parameters to surrogate
            if 'neural' in self.classifier_type:
                self.dom2_surrogate.epochs = self.classifier_params["epoch"]
            self.dom2_surrogate.class_weights =class_weights

            # Re-initialise training data and fit model
            self.dom2_surrogate.reset()
            self.dom2_surrogate.add_points(training_data[0], training_data[1])
            self.dom2_surrogate.train()

    def generate_offspring(self, n_gen=5, n_offspring=300, nr=4, flag='sbx_poly'):
        # Begin offspring generations from the current representatives
        survived = copy.deepcopy(self.representatives[[-2, -1]])
        n_survive = len(self.representatives)  # n_reps

        # conduct n_gen-1 generations of offspring generation
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
            categories = [offspring, [], []]

            # Predict cluster assignment of offspring
            offspring = self.evaluator.do(self.problem.obj_func, self.problem, offspring)
            off_vars = offspring.extract_var()
            cluster_labels = self.dom2_surrogate.predict(off_vars)

            # For each cluster, extract best points
            survived = Population(self.problem, 0)
            for i in range(len(self.clusters)):
                indices = i == cluster_labels
                if np.count_nonzero(indices) >= nr:
                    # Preselection 2
                    p_labels, p_proba = self.predict_offspring_pairs(offspring[indices], flag='dom1')
                    p_proba[p_labels != 1] = 0
                    edn = np.sum(p_proba, axis=1)
                    select = np.argsort(-edn)[:nr]
                    survived = Population.merge(survived, offspring[indices][select])

            # TODO: DEBUG ---------------------------------------------------------------------
            fig, ax = plt.subplots(1, 1, figsize=(9, 7))
            fig.supxlabel('Obj 1', fontsize=14)
            fig.supylabel('Obj 2', fontsize=14)

            for i in range(len(self.clusters)):
                indices = i == cluster_labels
                if np.count_nonzero(indices) > 1:
                    c_offs = offspring[indices].extract_obj()
                    ax.scatter(c_offs[:, 0], c_offs[:, 1], s=50, color=line_colors[i], label=f"clust: {i}")

            reps = self.representatives.extract_obj()
            ax.scatter(reps[:, 0], reps[:, 1], color='k', label='Reps')

            surv = survived.extract_obj()
            ax.scatter(surv[:, 0], surv[:, 1], marker='x', color='k', s=100, label='Survived')

            plt.legend(loc='upper right')
            plt.show()
            # TODO: DEBUG ---------------------------------------------------------------------

        # Final offspring of size n_offspring
        if 'sbx_poly' in flag:
            survived = self.mating.do(self.problem, survived, self.n_offspring)
        elif 'mode' in flag:
            survived = self.create_offspring.do(self.opt, self.representatives)
            survived = self.repair.do(self.problem, survived)

        # DEBUG ---------------------------------------------------------------------
        print(self.clusters)
        # copy_offs = self.evaluator.do(self.problem.obj_func, self.problem, copy_offs)
        survived = self.evaluator.do(self.problem.obj_func, self.problem, survived)
        self.plot_pareto(ref_vec=self.ref_dirs, scaling=1,
                         obj_array3=self.population.extract_obj(),
                         obj_array2=self.representatives[[-2, -1]].extract_obj(),
                         obj_array1=survived.extract_obj(), labels=['Offs', 'Reps', 'Pop'])

        return survived

    def two_stage_preselection(self, offspring, population, n_survive=1, counter='seq_perm'):
        # Obtain the best indices for each cluster
        best_ind_per_cluster, best_reps = self.best_from_cluster(flag=self.dom2_flag)

        # Select a cluster index using: 'random', 'sequential', 'seq_perm' strategies
        c_ind, cluster_indices = self.select_cluster(counter=counter)

        # Select from Q1-3 is cluster is not null
        if len(cluster_indices) > 0:
            # Obtain dom2 and dom1 - representative individuals in cluster
            dom1_rep, dom2_rep = self.obtain_representatives(population, c_ind, best_reps, best_ind_per_cluster)

            # Split up offspring into Q1, Q2 and Q3 categories
            categories = self.preselection_strategy_one(offspring, population[dom1_rep], population[dom2_rep],
                                                        flag='Q123')
        else:
            # Select from Q5 is cluster is null
            dom1_rep = best_reps
            dom2_rep = []
            # Offspring non-dominated with all dom2 representatives
            categories = self.preselection_strategy_one(offspring, population[dom1_rep], population[dom2_rep],
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
            print('Using Category 1,2 or 3')
            self.category_count[0] = self.category_count[0] + 1

            # Obtain prediction labels and probabilities for dom1 and dom2 representative-offspring comparisons
            p_labels, p_proba = self.predict_dominance_pairs(offspring, p_reps, flag='dom1')
            # t_labels, t_proba = self.predict_dominance_pairs(offspring, t_reps, flag='dom2')

            # 1d arrays
            p_labels = p_labels.flatten()
            # t_labels = t_labels.flatten()
            p_proba = p_proba.flatten()
            # t_proba = t_proba.flatten()

            # Determine Q1, Q2 and Q3 categories
            # mask_one = (p_labels == 1) & (t_labels == 1)
            # mask_two = (p_labels == 0) & (t_labels == 1)
            # mask_three = (p_labels == 1) & (t_labels == 0)
            mask_one = p_labels == 1
            mask_two = p_labels == 0
            mask_three = p_labels == 2

            # Assign offspring to categories
            Q1, Q2, Q3 = offspring[[mask_one]], offspring[[mask_two]], offspring[[mask_three]]

            # Assign combined probabilities to offspring
            Q1_proba = p_proba[mask_one] # + t_proba[mask_one]
            Q2_proba = p_proba[mask_two] # + t_proba[mask_two]
            Q3_proba = p_proba[mask_three] # + t_proba[mask_three]

            # Combine categories
            cat = [Q1, Q2, Q3]
            cat_proba = [Q1_proba, Q2_proba, Q3_proba]

        # Else assign offspring to category Q5
        elif 'Q5' in flag:
            print('Using Category 5')
            self.category_count[1] = self.category_count[1] + 1

            # Obtain prediction labels and probabilities for theta representative-offspring comparisons
            t_labels, t_proba = self.predict_dominance_pairs(offspring, t_reps, flag='dom1')

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

        # # DEBUG PURPOSES
        # self.evaluator.do(self.problem.obj_func, self.problem, offspring)
        # self.evaluator.do(self.problem.obj_func, self.problem, cat[0])
        # self.plot_pareto(ref_vec=self.ref_dirs, scaling=3,
        #                  obj_array1=offspring.extract_obj(),
        #                  obj_array2=cat[0].extract_obj(),
        #                  obj_array3=t_reps.extract_obj(),
        #                  labels=['offs', 'cat 123/5', 'Reps'])

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
        p_labels, p_proba = self.predict_offspring_pairs(offspring, flag='dom1')
        # t_labels, t_proba = self.predict_offspring_pairs(offspring, flag='dom2')

        # Assign Expected Dominance Number (EDN)
        p_proba[p_labels != 1] = 0
        # t_proba[t_labels != 1] = 0
        edn = np.sum(p_proba, axis=1) #  + np.sum(t_proba, axis=1)

        if n_survive == 1:
            # Select one individual with the highest EDN value
            infill_ind = np.array([np.argmax(edn)])
        else:
            # Retirn n_survive points with highest EDN values
            infill_ind = np.argsort(-edn)[:n_survive]

        return offspring[infill_ind]

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
        training_vars = np.zeros((int(n * (n - 1) / 2), 2 * self.problem.n_var))
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
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
            for j in range(i + 1, n):
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
        if 'dom1' in domination:
            # Pairs [u, v]
            uv_labels = self.dom1_surrogate.predict(uv_pairs)
            uv_probas = self.dom1_surrogate.predict_proba(uv_pairs)

            # Inverted pairs [v, u]
            vu_labels = self.dom1_surrogate.predict(vu_pairs)
            vu_probas = self.dom1_surrogate.predict_proba(vu_pairs)

        elif 'dom2' in domination:
            # Pairs [u, v]
            uv_labels = self.dom2_surrogate.predict(uv_pairs)
            uv_probas = self.dom2_surrogate.predict_proba(uv_pairs)

            # Inverted pairs [v, u]
            vu_labels = self.dom2_surrogate.predict(vu_pairs)
            vu_probas = self.dom2_surrogate.predict_proba(vu_pairs)
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

        # Extract objective values from population
        obj_array = population.extract_obj()
        dom_mat = None
        if 'dom1' in domination:
            # Calculate Pareto Domination matrix (-1: dominated, 1: dominates, 0: non-dominated)
            dom_mat = calculate_domination_matrix(obj_array, flag=self.dom1_flag)

        elif 'dom2' in domination:
            # Normalise objectives
            f_min = np.min(obj_array, axis=0)
            f_max = np.max(obj_array, axis=0)
            ideal = np.minimum(f_min, self.ideal)
            nadir = f_max

            # TODO: Normalise Objectives
            obj_array = (obj_array - ideal) / (nadir - ideal)

            if 'theta' in self.dom2_flag:
                # Assign clusters to population and calculate distances
                clusters, d1_mat, d2_mat = cluster_association(obj_array, self.ref_dirs[[0, -1]])

                # Calculate Theta Domination matrix (-1: dominated, 1: dominates, 0: non-dominated)
                dom_mat = calculate_theta_domination_matrix(obj_array, clusters, d1_mat, d2_mat, theta=self.theta)

            elif 'apd' in self.dom2_flag:
                # Assign clusters to population and calculate distances
                clusters, d1_mat, angle_mat, gamma = apd_association(obj_array, self.ref_dirs, ideal)

                # Calculate APD Domination matrix (-1: dominated, 1: dominates, 0: non-dominated)
                dom_mat = calculate_apd_domination_matrix(obj_array, clusters, d1_mat, angle_mat, gamma,
                                                          (self.evaluator.n_eval - self.n_init), self.max_infill,
                                                          alpha=self.alpha)
        else:
            raise Exception('Unrecognised dominance relation')

        # Construct training data arrays
        n = len(population)
        training_vars = np.zeros((2 * (n - 1), 2 * self.problem.n_var))
        training_labels = np.zeros(2 * (n - 1), dtype=int)

        idx = 0
        for i in range(n - 1):
            for j in range(n - 1, n):
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

    def diversity_training_data(self, population):
        n = len(population)

        # Update Clusters
        self.update_clusters(flag=self.dom2_flag)

        # Assign population to clusters
        training_vars = np.zeros((n, self.problem.n_var))
        training_labels = np.zeros(n, dtype=int)

        n_class = len(self.clusters)
        count = 0
        weights = []
        for i, indices in enumerate(self.clusters):
            weights.append(len(indices))
            for idx in indices:
                training_vars[count] = population[idx].var
                training_labels[count] = i
                count += 1

        # Conduct SMOTE Over-sampling
        # oversampler = SMOTE(k_neighbors=3)
        # training_vars, training_labels = oversampler.fit_resample(training_vars, training_labels)

        # Training data
        training_data = [training_vars, training_labels]

        # Calculate class weights
        class_weights = dict(enumerate(n / (n_class * np.array(weights))))

        return training_data, class_weights

    def prepare_training_data(self, population, domination=None):
        if domination is None:
            return None

        # Extract objective values from population
        obj_array = population.extract_obj()
        dom_mat = None
        if 'dom1' in domination:
            # Calculate Pareto Domination matrix (-1: dominated, 1: dominates, 0: non-dominated)
            dom_mat = calculate_domination_matrix(obj_array, flag=self.dom1_flag)
        elif 'dom2' in domination:
            # Update Clusters
            self.update_clusters(flag=self.dom2_flag)

            if 'theta' in self.dom2_flag:
                # Calculate Theta Domination matrix (-1: dominated, 1: dominates, 0: non-dominated)
                dom_mat = calculate_theta_domination_matrix(obj_array, self.clusters, self.d1_mat, self.d2_mat,
                                                            theta=self.theta)
            elif 'apd' in self.dom2_flag:
                dom_mat = calculate_apd_domination_matrix(obj_array, self.clusters, self.d1_mat, self.angle_mat,
                                                          self.gamma_apd, (self.evaluator.n_eval - self.n_init),
                                                          self.max_infill, alpha=self.alpha)
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

    def update_clusters(self, flag=None):
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
        if 'theta' in flag:
            self.clusters, self.d1_mat, self.d2_mat = cluster_association(obj_array, self.ref_dirs[[-2, -1]])
        elif 'apd' in flag:
            self.clusters, self.d1_mat, self.angle_mat, self.gamma_apd = apd_association(obj_array, self.ref_dirs,
                                                                                         self.ideal)
            # Conduct reference vector adaptation
            if self.n_gen % np.ceil(self.max_gen * self.adapt_fr) == 0:
                self.adapt()
        else:
            raise Exception(f"Provided dominance {flag} is not recognised!")

    def obtain_representatives(self, population, c_ind, best_reps, best_ind_per_cluster):
        # Obtain theta-representative individual
        theta_rep = best_ind_per_cluster[c_ind]
        theta_nondominated = population[best_reps]

        # Obtain pareto-representative solution from theta individuals in cluster
        obj_arr = theta_nondominated.extract_obj()
        fronts = NonDominatedSorting().do(obj_arr, return_rank=False)
        pareto_nondominated = theta_nondominated[fronts[0]]

        # Handle when the theta-rep is not in the nondominated set
        if theta_rep in best_reps[fronts[0]]:
            pareto_rep = theta_rep
        else:
            pareto_rep = None
            min_dist = np.inf
            ref_to_cluster = self.ref_dirs[c_ind]
            # Find individual that pareto-dominates theta-rep with smallest angle between ref vectors
            for i in range(len(pareto_nondominated)):
                # Select dominance relation
                domination = select_dom(self.dom1_flag)
                dom = domination(pareto_nondominated[i].obj, population[theta_rep[0]].obj)

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
