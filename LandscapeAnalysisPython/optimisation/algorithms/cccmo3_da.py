import copy
import random
import numpy as np

from scipy.stats import cauchy, norm
from scipy.spatial.distance import cdist

from optimisation.algorithms.evolutionary_algorithm import EvolutionaryAlgorithm

from optimisation.operators.sampling.random_sampling import RandomSampling
from optimisation.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from optimisation.operators.selection.tournament_selection import TournamentSelection, comp_by_cv_then_random
from optimisation.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from optimisation.operators.mutation.polynomial_mutation import PolynomialMutation
from optimisation.operators.mutation.de_mutation import DifferentialEvolutionMutation
from optimisation.operators.survival.rank_and_crowding_survival import RankAndCrowdingSurvival
from optimisation.operators.survival.two_ranking_survival import TwoRankingSurvival
from optimisation.operators.survival.self_adaptive_feasible_ratio_epsilon_survival import SelfAdaptiveFeasibleRatioEpsilonSurvival
from optimisation.model.duplicate import DefaultDuplicateElimination

from optimisation.model.duplicate import DefaultDuplicateElimination
from optimisation.model.population import Population
from optimisation.model.repair import BasicBoundsRepair
from optimisation.model.normalisation import Normalisation

from optimisation.util.non_dominated_sorting import NonDominatedSorting
from optimisation.util import dominator

"""
"A Coevolutionary Framework for Constrained Multiobjective Optimization Problems" - Tian2021
"""

import matplotlib.pyplot as plt

plt.style.use('seaborn-talk')
line_colors = ['green', 'blue', 'red', 'orange', 'cyan', 'lawngreen', 'm', 'orangered', 'sienna', 'gold',
               'violet',
               'indigo', 'cornflowerblue']


class CCMO3(EvolutionaryAlgorithm):

    def __init__(self,
                 ref_dirs=None,
                 n_population=100,
                 sampling=LatinHypercubeSampling(),
                 selection=TournamentSelection(comp_func=comp_by_cv_then_random),
                 crossover=SimulatedBinaryCrossover(eta=30, prob=1.0),
                 mutation=PolynomialMutation(eta=20, prob=None),
                 eliminate_duplicates=DefaultDuplicateElimination(),
                 survival1=RankAndCrowdingSurvival(filter_infeasible=False),
                 survival2=SelfAdaptiveFeasibleRatioEpsilonSurvival(filter_infeasible=False),
                 survival3=SelfAdaptiveFeasibleRatioEpsilonSurvival(filter_infeasible=False),
                 **kwargs):

        self.ref_dirs = ref_dirs

        if 'save_results' in kwargs:
            self.save_results = kwargs['save_results']

        if 'save_name' in kwargs:
            self.save_name = kwargs['save_name']

        self.survival1 = survival1
        self.survival2 = survival2
        self.survival3 = survival3

        super().__init__(n_population=n_population,
                         sampling=sampling,
                         selection=selection,
                         crossover=crossover,
                         mutation=mutation,
                         survival=survival1,
                         eliminate_duplicates=eliminate_duplicates,
                         **kwargs)

    def _initialise(self):
        # Initialise from the evolutionary algorithm class -------------------------------------------------------------
        # Instantiate population
        self.population = Population(self.problem, self.n_population)

        # Population initialisation
        if self.hot_start:
            # Initialise population using hot-start
            self.hot_start_initialisation()
        else:

            # Initialise surrogate modelling strategy
            if self.surrogate is not None:
                self.surrogate.initialise(self.problem, self.sampling)

            # Compute sampling
            self.sampling.do(self.n_population, self.problem.x_lower, self.problem.x_upper, self.seed)

            # Assign sampled design variables to population
            self.population.assign_var(self.problem, self.sampling.x)
            if self.x_init:
                self.population[0].set_var(self.problem, self.problem.x_value)
            if self.x_init_additional and self.problem.x_value_additional is not None:
                for i in range(len(self.problem.x_value_additional)):
                    self.population[i+1].set_var(self.problem, self.problem.x_value_additional[i, :])

        # Evaluate initial population
        if self.surrogate is not None:
            self.population = self.evaluator.do(self.surrogate.obj_func, self.problem, self.population)
        else:
            self.population = self.evaluator.do(self.problem.obj_func, self.problem, self.population)

        # Calculate maximum constraint values across the population
        self.max_abs_con_vals = self.evaluator.calc_max_abs_cons(self.population, self.problem)

        # Assign rank and crowding to population
        self.population.assign_rank_and_crowding()

        # Create co-evolution population sets
        self.population2 = copy.deepcopy(self.population)
        self.population3 = copy.deepcopy(self.population)
        self.reps = copy.deepcopy(self.population)

        # Dummy survival call to ensure population is ranked prior to mating selection
        self.population = self.survival1.do(self.problem, self.population, self.n_population, self.n_gen, self.max_gen)
        self.population2 = self.survival2.do(self.problem, self.population2, self.n_population, self.n_gen, self.max_gen)
        self.population3 = self.survival3.do(self.problem, self.population3, self.n_population, self.n_gen, self.max_gen)

        # Number of offspring to generate by individual co-evolution
        self.n_merged = int(0.5 * self.n_population)

        # Auxiliary population stuff
        self.eps0 = 0.15 * self.get_max_cons_viol(self.population)
        self.eps = None
        self.cp = 5.0
        self.n_active_cons = np.zeros(self.problem.n_con)
        self.n_original_cons = self.problem.n_con

        # DE offspring
        self.repair = BasicBoundsRepair()
        self.de_mutation = DifferentialEvolutionMutation(problem=self.problem, method='balanced')
        self.f = [0.6, 0.8, 1.0]
        self.cr = [0.1, 0.2, 1.0]

        self.offspring1 = None
        self.offpsring2 = None
        self.offspring3 = None
        self.div_archive1 = Population(self.problem)
        self.div_archive2 = Population(self.problem)
        self.div_archive3 = Population(self.problem)

    def _next(self):
        # Reduce the dynamic constraint boundary
        self.eps = self.reduce_boundary(self.eps0, self.n_gen, self.max_gen, self.cp)
        # self.eps = self.reduce_boundary_linear(self.eps0, self.n_gen, self.max_gen)  # ccmo_safr_linear_eps_%
        # self.eps = self.reduce_boundary_exp(self.eps0, self.n_gen, self.max_gen)
        # self.eps = self.reduce_boundary_sinusoid(self.eps0, self.n_gen, self.max_gen)
        print("Eps: ", self.eps)

        # Calculate which constraints are active
        # cons = self.population2.extract_cons()
        # active = np.count_nonzero(cons >= 0.0, axis=0)
        # self.n_active_cons += active
        # print(f"Active cons: {self.n_active_cons}")

        # Merge populations with diversity archives
        self.population = Population.merge(self.population, self.div_archive1)
        self.population2 = Population.merge(self.population2, self.div_archive2)
        self.population3 = Population.merge(self.population3, self.div_archive3)

        # Generate offspring
        # self.offspring = self.mating.do(self.problem, self.reps, self.n_merged)
        self.offspring1 = self.generate_ga_de_offspring(self.population, int(self.n_population / 3))
        self.offspring2 = self.generate_ga_de_offspring(self.population2, int(self.n_population / 3))
        self.offspring3 = self.generate_ga_de_offspring(self.population3, int(self.n_population / 3))
        # self.offspring = Population.merge_multiple(self.offspring1, self.offspring2, self.offspring3)

        # Evaluate offspring
        self.offspring1 = self.evaluator.do(self.problem.obj_func, self.problem, self.offspring1, self.max_abs_con_vals)
        self.offspring2 = self.evaluator.do(self.problem.obj_func, self.problem, self.offspring2, self.max_abs_con_vals)
        self.offspring3 = self.evaluator.do(self.problem.obj_func, self.problem, self.offspring3, self.max_abs_con_vals)

        # Merge the offspring with the current population
        self.population = Population.merge_multiple(self.population, self.offspring1, self.offspring2, self.offspring3)
        self.population2 = Population.merge_multiple(self.population2, self.offspring1, self.offspring2, self.offspring3)
        self.population3 = Population.merge_multiple(self.population3, self.offspring1, self.offspring2, self.offspring3)

        # Environmental control of Population1 (Feasible-first survival)
        self.population = self.survival1.do(self.problem, self.population, self.n_population, self.n_gen, self.max_gen)

        # Environment control of Population2 (Balanced Rank-based survival)
        self.population2 = self.survival2.do(self.problem, self.population2, self.n_population, self.n_gen, self.max_gen)

        # Environment control of Population3 (Epsilon-relaxed survival)
        eps_population = self.transform_population(self.population3)
        max_abs_con_vals = self.evaluator.calc_max_abs_cons(eps_population, self.problem)
        eps_population = self.evaluator.sum_normalised_cons(eps_population, self.problem, max_abs_con_vals=max_abs_con_vals)
        survived = self.survival3._do(self.problem, eps_population, self.n_population, cons_val=eps_population.extract_cons_sum())
        self.population3 = self.population3[survived]

        # Update diversity archives
        self.div_archive1 = self.update_diversity_archive(self.offspring1, self.population, self.div_archive1)
        self.div_archive2 = self.update_diversity_archive(self.offspring2, self.population2, self.div_archive2)
        self.div_archive3 = self.update_diversity_archive(self.offspring3, self.population3, self.div_archive3)
        print('Archive sizes: ', len(self.div_archive1), len(self.div_archive2), len(self.div_archive3))

        # Merge representatives
        # self.reps = Population.merge(self.population, self.population2)
        # self.reps = Population.merge(self.reps, self.population3)

        # TODO: DEBUG PLOTTING -----------------------------------------------------------------------------------------
        if (self.n_gen-1) % 100 == 0 or self.n_gen == self.max_gen:
            self._plot_populations(pop1=self.population, pop2=self.population2, pop3=self.population3)
        # TODO: DEBUG PLOTTING -----------------------------------------------------------------------------------------

        if self.n_gen == self.max_gen:
            final_population = Population.merge_multiple(self.population, self.population2, self.population3)
            self.survival1.filter_infeasible = True
            self.population = self.survival1.do(self.problem, final_population, self.n_population, self.n_gen, self.max_gen)

    def generate_ga_de_offspring(self, survived, n_offspring):
        # Generate GA-SBX offpsring
        offspring = self.mating.do(self.problem, survived, int(n_offspring / 4))
        offspring = self.repair.do(self.problem, offspring)

        # Generate DE offspring (MODE)
        offspring1 = self.de_mutation.do(survived, self.population3, int(np.ceil(n_offspring / 10)))
        offspring1 = self.repair.do(self.problem, offspring1)

        # Generate transfer-DE offspring (URCMO)
        # offspring2 = self.create_de_offspring(self.population, self.population3, int(np.ceil(n_offspring / 4)))

        # Merge all offspring together
        offspring = Population.merge(offspring, offspring1)
        # offspring = Population.merge(offspring, offspring2)

        return offspring

    def create_de_offspring(self, population, archive, n_offspring):
        # Extract variables from populations
        var1_array = population.extract_var()
        var2_array = archive.extract_var()

        # Obtain F and CR values from list
        f, cr = self.assign_f_and_cr_from_pool()
        f = cauchy.rvs(f, 0.1, 1)
        cr = norm.rvs(cr, 0.1, 1)

        # Select DE operator
        rand_int = np.random.uniform(0.0, 1.0, 1)

        if rand_int < 0.5:
            # Generate offspring from DE/transfer/1
            offspring = self.transfer_to_1_bin(var2_array, var1_array, f, cr, n_offspring)
        else:
            # Select one random individual from top 100p% individuals in population1
            fitness = self.spea2_fitness(population)
            p = np.random.randint(1, len(population))
            sorted_indices = np.argpartition(fitness, p)[:p]
            opbest_indices = np.random.choice(sorted_indices, 1)
            opbest_arr = var1_array[opbest_indices]

            # Generate offspring from DE/current-to-opbest/1
            offspring = self.current_to_opbest_1_bin(var2_array, opbest_arr, f, cr, n_offspring)

        return offspring

    def transfer_to_1_bin(self, var1_array, var2_array, f, cr, population_size):
        mutant_array = np.zeros(var1_array.shape)
        trial_array = np.zeros(var1_array.shape)

        # Randomly permutate arrays to ensure fair chance of all individuals
        perm_mask = np.random.permutation(list(range(len(var1_array))))
        var1_array = var1_array[perm_mask]
        var2_array = var2_array[perm_mask]

        for idx in range(population_size):
            rand_indices = self._select_random_indices(population_size, 1, current_index=idx)
            mutant_array[idx, :] = var1_array[rand_indices, :]

            for var_idx in range(self.problem.n_var):
                rand = np.random.random(1)
                j_rand = np.random.randint(0, self.problem.n_var)
                if rand < cr or var_idx == j_rand:
                    trial_array[idx, var_idx] = mutant_array[idx, var_idx]
                else:
                    trial_array[idx, var_idx] = var2_array[idx, var_idx]

        offspring = Population(self.problem, population_size)
        offspring.assign_var(self.problem, trial_array)
        offspring = self.repair.do(self.problem, offspring)

        return offspring

    def current_to_opbest_1_bin(self, var2_array, opbest_array, f, cr, population_size):
        mutant_array = np.zeros((population_size, self.problem.n_var))
        trial_array = np.zeros((population_size, self.problem.n_var))

        # Randomly permutate arrays to ensure fair chance of all individuals
        perm_mask = np.random.permutation(list(range(len(var2_array))))
        var2_array = var2_array[perm_mask]

        # loop through the population
        for idx in range(population_size):
            rand_indices = self._select_random_indices(population_size, 2, current_index=idx)
            mutant_array[idx, :] = var2_array[idx, :] + \
                                   f * (opbest_array - var2_array[idx, :]) + \
                                   f * (var2_array[rand_indices[0], :] - var2_array[rand_indices[1], :])

            # Do not perform crossover
            trial_array = mutant_array

        offspring = Population(self.problem, population_size)
        offspring.assign_var(self.problem, trial_array)
        offspring = self.repair.do(self.problem, offspring)

        return offspring

    def assign_f_and_cr_from_pool(self):
        ind = random.randint(0, len(self.f) - 1)
        f = self.f[ind]
        cr = self.cr[ind]

        return f, cr

    def update_diversity_archive(self, offspring, population, archive):
        # Extract non-dominated from offspring ignoring constraints
        selected = self.survival1._do(self.problem, offspring, self.n_population, cons_val=None)
        non_dominated = offspring[selected]

        # Determine most diverse individuals with respect to existing population (constrained)
        pop_obj_arr = population.extract_obj()
        obj_arr = non_dominated.extract_obj()

        current_min = np.min(pop_obj_arr, axis=0)
        current_max = np.max(pop_obj_arr, axis=0)
        lower_extreme = np.where(np.any(obj_arr < current_min, axis=0))[0]
        upper_extreme = np.where(np.any(obj_arr > current_max, axis=0))[0]
        extremes = np.unique(np.hstack((lower_extreme, upper_extreme)))

        # Combine new diverse individuals with archive
        if len(extremes) > 0:
            diverse_individuals = copy.deepcopy(non_dominated[extremes])
            merged_archive = Population.merge(archive, diverse_individuals)
            selected = NonDominatedSorting().do(np.atleast_2d(merged_archive.extract_obj()), only_non_dominated_front=True, return_rank=False)
            new_archive = merged_archive[selected]
        else:
            new_archive = archive

        return new_archive

    @staticmethod
    def spea2_fitness(pop):
        # Extract objectives
        obj_val = np.atleast_2d(pop.extract_obj())
        cons_val = pop.extract_cons_sum()

        # Calculate domination matrix
        M = dominator.calculate_domination_matrix(obj_val, cons_val, domination_type="pareto")

        # Number of solutions each individual dominates
        S = np.sum(M == 1, axis=0)

        # The raw fitness of each solution (strength of its dominators)
        R = np.sum(((M == -1) * S), axis=1)

        # Determine the k-th nearest neighbour
        k = int(np.sqrt(len(pop)))
        if k >= len(pop):
            k -= 1

        _pop = Normalisation().do(copy.deepcopy(pop), recalculate=True)
        _obj_val = np.atleast_2d(_pop.extract_obj())

        # Calculate distance matrix and sort by nearest neighbours
        dist_mat = cdist(_obj_val, _obj_val)
        np.fill_diagonal(dist_mat, np.inf)
        sorted_dist_mat = np.sort(dist_mat, axis=1)

        # Inverse distance metric
        D = 1.0 / (sorted_dist_mat[:, k] + 2.0)

        # SPEA2 fitness
        fitness = R + D

        return fitness

    @staticmethod
    def _select_random_indices(population_size, nr_indices, current_index=None):
        index_list = list(range(population_size))
        if current_index is not None:
            index_list.pop(current_index)
        selected_indices = random.sample(index_list, nr_indices)

        return selected_indices

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
    def feasible_fraction(population, return_individual_feasible_frac=False):
        cons_array = population.extract_cons_sum()
        feasible_mask = cons_array == 0.0
        feasible_frac = np.count_nonzero(feasible_mask) / len(population)

        if return_individual_feasible_frac:
            cons_array = population.extract_cons()
            individual_feasible_frac = np.count_nonzero(cons_array <= 0.0, axis=0) / len(population)
            return feasible_frac, individual_feasible_frac

        return feasible_frac

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

    @staticmethod
    def reduce_boundary_linear(eps0, n_gen, max_gen):
        eps = eps0 - (eps0 / max_gen) * n_gen

        return eps

    @staticmethod
    def reduce_boundary_exp(eps0, n_gen, max_gen):
        eps = eps0 * np.exp(-n_gen / 400)

        return eps

    @staticmethod
    def reduce_boundary_sinusoid(eps0, n_gen, max_gen, offset=0.1):
        eps = eps0 * np.sin(np.pi * n_gen / max_gen) - offset * n_gen / max_gen + offset

        return eps

    def _find_cross_survived_population(self, offspring, survived):
        offs_obj = copy.deepcopy(np.atleast_2d(offspring.extract_obj()))
        surv_obj = copy.deepcopy(np.atleast_2d(survived.extract_obj()))

        where_is_equal = []
        for i in range(len(survived)):
            n_equal = np.count_nonzero(surv_obj[i] == offs_obj, axis=1)
            if any(n_equal):
                where_is_equal.append(i)

        return copy.deepcopy(survived[where_is_equal])

    def _plot_populations(self, pop1, pop2, pop3):
        fig, ax = plt.subplots(1, 1, figsize=(9, 7))
        fig.supxlabel('Obj 1', fontsize=14)
        fig.supylabel('Obj 2', fontsize=14)

        # extract objs
        pop1_obj = copy.deepcopy(np.atleast_2d(pop1.extract_obj()))
        pop2_obj = copy.deepcopy(np.atleast_2d(pop2.extract_obj()))
        pop3_obj = copy.deepcopy(np.atleast_2d(pop3.extract_obj()))

        # Exact front
        exact_obj = self.problem.pareto_set.extract_obj()
        ax.scatter(exact_obj[:, 0], exact_obj[:, 1], color='k', s=75, label="Exact PF")

        # Primary population & cross-survival from auxiliary population
        ax.scatter(pop1_obj[:, 0], pop1_obj[:, 1], color='blue', s=50, alpha=0.5, label="Pop1")
        ax.scatter(pop2_obj[:, 0], pop2_obj[:, 1], color='green', s=50, alpha=0.5, label="Pop2")
        ax.scatter(pop3_obj[:, 0], pop3_obj[:, 1], color='red', s=50, alpha=0.5, label="Pop3")

        plt.legend(loc='best', frameon=False)
        plt.savefig(f'/home/juan/Desktop/ccmo_viz/ccmo3_da_{self.problem.name}_gen_{self.n_gen-1}.png')
        # plt.show()
