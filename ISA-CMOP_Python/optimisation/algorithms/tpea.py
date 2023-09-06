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

from optimisation.util import dominator
from optimisation.util.non_dominated_sorting import NonDominatedSorting

"""
"Constrained Multiobjective Optimization with Escape and Expansion Forces" - Liu2023
"""

import matplotlib.pyplot as plt

plt.style.use('seaborn-talk')
line_colors = ['green', 'blue', 'red', 'orange', 'cyan', 'lawngreen', 'm', 'orangered', 'sienna', 'gold',
               'violet', 'indigo', 'cornflowerblue']


class TPEA(EvolutionaryAlgorithm):

    def __init__(self,
                 ref_dirs=None,
                 n_population=100,
                 sampling=LatinHypercubeSampling(),
                 selection=TournamentSelection(comp_func=comp_by_cv_then_random),
                 crossover=SimulatedBinaryCrossover(eta=30, prob=1.0),
                 mutation=PolynomialMutation(eta=20, prob=None),
                 eliminate_duplicates=DefaultDuplicateElimination(),
                 survival1=RankAndCrowdingSurvival(filter_infeasible=False),
                 survival2=RankAndCrowdingSurvival(filter_infeasible=False),
                 survival3=RankAndCrowdingSurvival(filter_infeasible=False),
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

        # DE offspring
        self.repair = BasicBoundsRepair()
        self.de_mutation = DifferentialEvolutionMutation(problem=self.problem, method='balanced')
        self.f = [0.6, 0.8, 1.0]
        self.cr = [0.1, 0.2, 1.0]

        # Extras
        self.offspring1 = None
        self.offpsring2 = None
        self.offspring3 = None
        self.max_abs_con_vals2 = copy.deepcopy(self.max_abs_con_vals)
        self.max_abs_con_vals3 = copy.deepcopy(self.max_abs_con_vals)
        self.nondom = NonDominatedSorting()

    def _next(self):
        # Generate offspring
        self.offspring1 = self.generate_ga_de_offspring(self.population, int(self.n_population / 3))
        self.offspring2 = self.generate_ga_de_offspring(self.population2, int(self.n_population / 3))
        self.offspring3 = self.generate_ga_de_offspring(self.population3, int(self.n_population / 3))

        # Evaluate offspring
        self.offspring1 = self.evaluator.do(self.problem.obj_func, self.problem, self.offspring1, self.max_abs_con_vals)
        self.offspring2 = self.evaluator.do(self.problem.obj_func, self.problem, self.offspring2, self.max_abs_con_vals2)
        self.offspring3 = self.evaluator.do(self.problem.obj_func, self.problem, self.offspring3, self.max_abs_con_vals3)

        # Merge the offspring with the current population
        self.population = Population.merge_multiple(self.population, self.offspring1, self.offspring2, self.offspring3)
        self.population2 = Population.merge_multiple(self.population2, self.offspring1, self.offspring2, self.offspring3)
        self.population3 = Population.merge_multiple(self.population3, self.offspring1, self.offspring2, self.offspring3)

        # Environmental control of Population1 (Feasible-first survival)
        self.population = self.survival1.do(self.problem, self.population, self.n_population, self.n_gen, self.max_gen)

        # Environment control of Population2 (Inner infeasible solutions)
        self.population2 = self.survive_pop2(self.population2, self.n_population)

        # Environment control of Population3 (Outter infeasible solutions, respectively)
        self.population3 = self.survive_pop3(self.population3, self.n_population)

        # # TODO: DEBUG PLOTTING -----------------------------------------------------------------------------------------
        if (self.n_gen-1) % 100 == 0:
            self._plot_populations(pop1=self.population, pop2=self.population2, pop3=self.population3)
        if self.n_gen == self.max_gen:
            self._plot_populations(pop1=self.population, pop2=self.population, pop3=self.population)
        # # TODO: DEBUG PLOTTING -----------------------------------------------------------------------------------------

        if self.n_gen == self.max_gen:
            final_population = Population.merge_multiple(self.population, self.population2, self.population3)
            self.survival1.filter_infeasible = True
            self.population = self.survival1.do(self.problem, final_population, self.n_population, self.n_gen, self.max_gen)

    def survive_pop2(self, population, n_survive):
        # Find non-dominated solutions from modified opt problem (MOP 4)
        cv_obj = self.calc_cv_obj(population)
        obj_arr = population.extract_obj()
        mod_obj = np.hstack((obj_arr, cv_obj[:, None]))
        best_front = self.nondom.do(mod_obj, only_non_dominated_front=True, return_rank=False)

        # Select only the infeasible from the non-dominated front
        infeasible = np.where(cv_obj[best_front] > 0.0)[0]
        infeasible_population = copy.deepcopy(population[best_front][infeasible])

        # Conduct rank & crowding ignoring constraints (Pop2)
        selected = self.survival2._do(self.problem, infeasible_population, n_survive, cons_val=None)
        survived = infeasible_population[selected]

        return survived

    def survive_pop3(self, population, n_survive):
        # Find non-dominated solutions from modified opt problem (MOP 4)
        cv_obj = self.calc_cv_obj(population)
        obj_arr = population.extract_obj()
        mod_obj = np.hstack((obj_arr, cv_obj[:, None]))
        best_front = self.nondom.do(mod_obj, only_non_dominated_front=True, return_rank=False)

        # Select only the infeasible from the non-dominated front
        infeasible = np.where(cv_obj[best_front] > 0.0)[0]
        infeasible_population = copy.deepcopy(population[best_front][infeasible])

        # Modify infeasible population to maximise objectives
        mod_infeasible_pop = copy.deepcopy(infeasible_population)
        mod_infeasible_pop.assign_obj(-mod_infeasible_pop.extract_obj())

        # Conduct rank & crowding ignoring constraints (Pop3)
        selected = self.survival2._do(self.problem, mod_infeasible_pop, n_survive, cons_val=None)
        survived = infeasible_population[selected]

        return survived

    def generate_ga_de_offspring(self, survived, n_offspring):
        # Generate GA-SBX offpsring
        offspring = self.mating.do(self.problem, survived, int(n_offspring / 4))
        offspring = self.repair.do(self.problem, offspring)

        # Generate DE offspring (MODE)
        offspring1 = self.de_mutation.do(survived, self.population3, int(np.ceil(n_offspring / 10)))
        offspring1 = self.repair.do(self.problem, offspring1)

        # Merge all offspring together
        offspring = Population.merge(offspring, offspring1)

        return offspring

    @staticmethod
    def calc_cv_obj(population):
        cons_array = np.atleast_2d(copy.deepcopy(population.extract_cons()))
        cons_array[cons_array <= 0.0] = 0.0
        cv_obj = np.sum(cons_array, axis=1)

        return cv_obj

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
        plt.savefig(f'/home/juan/Desktop/tpea_viz/tpea_{self.problem.name}_gen_{self.n_gen-1}.png')
        # plt.show()