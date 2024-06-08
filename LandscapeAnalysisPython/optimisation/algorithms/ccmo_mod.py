import copy

import numpy as np

from optimisation.algorithms.evolutionary_algorithm import EvolutionaryAlgorithm

from optimisation.operators.sampling.random_sampling import RandomSampling
from optimisation.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from optimisation.operators.mutation.de_mutation import DifferentialEvolutionMutation
from optimisation.operators.mutation.polynomial_mutation import PolynomialMutation
from optimisation.operators.selection.tournament_selection import TournamentSelection, comp_by_cv_then_random
from optimisation.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from optimisation.operators.survival.rank_and_crowding_survival import RankAndCrowdingSurvival
from optimisation.operators.survival.two_ranking_survival import TwoRankingSurvival

from optimisation.model.duplicate import DefaultDuplicateElimination
from optimisation.model.population import Population
from optimisation.model.repair import BasicBoundsRepair

"""
"A Coevolutionary Framework for Constrained Multiobjective Optimization Problems" - Tian2021
"""

import matplotlib.pyplot as plt

plt.style.use('seaborn-talk')
line_colors = ['green', 'blue', 'red', 'orange', 'cyan', 'lawngreen', 'm', 'orangered', 'sienna', 'gold',
               'violet',
               'indigo', 'cornflowerblue']


class CCMO(EvolutionaryAlgorithm):

    def __init__(self,
                 ref_dirs=None,
                 n_population=100,
                 sampling=LatinHypercubeSampling(),
                 selection=TournamentSelection(comp_func=comp_by_cv_then_random),
                 crossover=SimulatedBinaryCrossover(eta=30, prob=1.0),
                 mutation=PolynomialMutation(eta=20, prob=None),
                 eliminate_duplicates=DefaultDuplicateElimination(),
                 survival1=TwoRankingSurvival(filter_infeasible=False),
                 survival2=TwoRankingSurvival(filter_infeasible=False),
                 **kwargs):

        self.ref_dirs = ref_dirs

        if 'save_results' in kwargs:
            self.save_results = kwargs['save_results']

        if 'save_name' in kwargs:
            self.save_name = kwargs['save_name']

        self.survival1 = survival1
        self.survival2 = survival2

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

        # Create co-evolution population sets
        self.population2 = copy.deepcopy(self.population)

        # Dummy survival call to ensure population is ranked prior to mating selection
        self.population = self.survival1.do(self.problem, self.population, self.n_population, self.n_gen, self.max_gen)
        self.population2 = self.survival2.do(self.problem, self.population2, self.n_population, self.n_gen, self.max_gen)

        # Calculate maximum constraint values across the population
        self.max_abs_con_vals = self.evaluator.calc_max_abs_cons(self.population, self.problem)

        # Assign rank and crowding to population
        self.population.assign_rank_and_crowding()

        # Number of offspring to generate by individual co-evolution
        self.n_half_offspring = int(self.n_offspring / 2)

        # DE offspring
        self.repair = BasicBoundsRepair()
        self.de_mutation = DifferentialEvolutionMutation(problem=self.problem, method='diversity')

        # Auxiliary population stuff
        self.eps0 = 0.25 * self.get_max_cons_viol(self.population)
        self.eps = None
        self.cp = 5.0
        self.n_active_cons = np.zeros(self.problem.n_con)

        self.aux_population = 'eps'  # 'unconstrained', 'eps', 'active'
        self.n_original_cons = self.problem.n_con

    def _next(self):
        # Update Current feasible fraction in population
        self.feasible_frac = self.feasible_fraction(self.population, return_individual_feasible_frac=True)
        print(f"Individual Feasible Fractions: {self.feasible_frac}")

        # Calculate which constraints are active
        cons = self.population2.extract_cons()
        active = np.count_nonzero(cons >= 0.0, axis=0)
        self.n_active_cons += active
        print(f"Active cons: {self.n_active_cons}")

        # Reduce the dynamic constraint boundary
        self.eps = self.reduce_boundary(self.eps0, self.n_gen, self.max_gen, self.cp)
        # self.eps = self.reduce_boundary_linear(self.eps0, self.n_gen, self.max_gen)  # ccmo_safr_linear_eps_%
        # self.eps = self.reduce_boundary_exp(self.eps0, self.n_gen, self.max_gen)       
        print("Eps: ", self.eps)

        # Generate offspring
        self.offspring1 = self.mating.do(self.problem, self.population, self.n_half_offspring)
        self.offspring2 = self.mating.do(self.problem, self.population2, self.n_half_offspring)

        # Evaluate offspring
        if self.surrogate is not None:
            self.offspring1 = self.evaluator.do(self.surrogate.obj_func, self.problem, self.offspring1, self.max_abs_con_vals)
            self.offspring2 = self.evaluator.do(self.surrogate.obj_func, self.problem, self.offspring2, self.max_abs_con_vals)
        else:
            self.offspring1 = self.evaluator.do(self.problem.obj_func, self.problem, self.offspring1, self.max_abs_con_vals)
            self.offspring2 = self.evaluator.do(self.problem.obj_func, self.problem, self.offspring2, self.max_abs_con_vals)

        # Merge the offspring with the current population
        self.population = Population.merge(copy.deepcopy(self.population), self.offspring1)
        self.population = Population.merge(copy.deepcopy(self.population), self.offspring2)
        self.population2 = Population.merge(copy.deepcopy(self.population2), self.offspring1)
        self.population2 = Population.merge(copy.deepcopy(self.population2), self.offspring2)

        # Conduct environmental control (principal population)
        self.population = self.survival1.do(self.problem, self.population, self.n_population, self.n_gen, self.max_gen)

        # Conduct environmental control (auxiliary population)
        if self.aux_population == 'unconstrained':
            self.population2 = self.population2[self.survival2._do(self.problem, self.population2, self.n_population,
                                                                   cons_val=None)]
        elif self.aux_population == 'eps':
            # Apply the DC-NSGA2 epsilon-relaxation to population
            aux_population = self.transform_population(self.population2)
            max_abs_con_vals = self.evaluator.calc_max_abs_cons(aux_population, self.problem)
            aux_population = self.evaluator.sum_normalised_cons(aux_population, self.problem, max_abs_con_vals=max_abs_con_vals)
            survived = self.survival2._do(self.problem, aux_population, self.n_population, cons_val=aux_population.extract_cons_sum())
            self.population2 = self.population2[survived]

        elif self.aux_population == 'active':
            # Remove inactive constraints
            active_mask = np.argwhere(self.n_active_cons > self.n_population).flatten()
            active_cons = self.population2.extract_cons()
            aux_pop = copy.deepcopy(self.population2)
            if len(active_mask) > 0:
                active_cons = active_cons[:, active_mask]
                print('                                                ', active_mask)
                aux_pop.assign_cons(active_cons)
                self.problem.n_con = len(active_mask)

            # Perform survival
            max_abs_con_vals = self.evaluator.calc_max_abs_cons(aux_pop, self.problem)
            aux_pop = self.evaluator.sum_normalised_cons(aux_pop, self.problem, max_abs_con_vals=max_abs_con_vals)
            survived = self.survival2._do(self.problem, aux_pop, self.n_population, cons_val=aux_pop.extract_cons_sum())
            self.population2 = self.population2[survived]
            self.problem.n_con = self.n_original_cons

        # Resetting active counts
        if (self.n_gen % 50) == 0:
            self.n_active_cons[:] = 0.0

        # TODO: DEBUG PLOTTING -----------------------------------------------------------------------------------------
        if (self.n_gen-1) % 50 == 0:
            aux_in_pop = self._find_cross_survived_population(self.offspring2, self.population)
            pop_in_aux = self._find_cross_survived_population(self.offspring1, self.population2)
            self._plot_populations(pop=self.population, aux=self.population2, aux_in_pop=aux_in_pop, pop_in_aux=pop_in_aux)
        # TODO: DEBUG PLOTTING -----------------------------------------------------------------------------------------

    def generate_ga_de_offspring(self, survived, n_offspring):
        offspring = self.mating.do(self.problem, survived, int(n_offspring / 2))    # GA-SBX
        offspring = self.repair.do(self.problem, offspring)
        offspring1 = self.de_mutation.do(survived, self.population, int(n_offspring / 2))  # DE
        offspring1 = self.repair.do(self.problem, offspring1)
        offspring = Population.merge(offspring, offspring1)

        return offspring

    def transform_population(self, population):
        # Create new objectives and transformed constraint
        cons_arr = population.extract_cons()
        eps_cons_arr = copy.deepcopy(cons_arr) - self.eps
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

    def _find_cross_survived_population(self, offspring, survived):
        offs_obj = copy.deepcopy(np.atleast_2d(offspring.extract_obj()))
        surv_obj = copy.deepcopy(np.atleast_2d(survived.extract_obj()))

        where_is_equal = []
        for i in range(len(survived)):
            n_equal = np.count_nonzero(surv_obj[i] == offs_obj, axis=1)
            if any(n_equal):
                where_is_equal.append(i)

        return copy.deepcopy(survived[where_is_equal])

    def _plot_populations(self, pop, aux, aux_in_pop, pop_in_aux):
        fig, ax = plt.subplots(1, 1, figsize=(9, 7))
        fig.supxlabel('Obj 1', fontsize=14)
        fig.supylabel('Obj 2', fontsize=14)

        # extract objs
        pop_obj = copy.deepcopy(np.atleast_2d(pop.extract_obj()))
        aux_obj = copy.deepcopy(np.atleast_2d(aux.extract_obj()))
        aux_in_pop_obj = copy.deepcopy(np.atleast_2d(aux_in_pop.extract_obj()))
        pop_in_aux_obj = copy.deepcopy(np.atleast_2d(pop_in_aux.extract_obj()))

        # Exact front
        exact_obj = self.problem.pareto_set.extract_obj()
        ax.scatter(exact_obj[:, 0], exact_obj[:, 1], color='k', s=75, label="Exact PF")

        # Primary population & cross-survival from auxiliary population
        ax.scatter(pop_obj[:, 0], pop_obj[:, 1], color='blue', s=50, alpha=0.5, label="Pop")
        try:
            ax.scatter(aux_in_pop_obj[:, 0], aux_in_pop_obj[:, 1], color='w', s=15, edgecolors='blue', label="Aux-in-pop")
        except:
            pass

        # Auxiliary population & cross-survival from primary population
        ax.scatter(aux_obj[:, 0], aux_obj[:, 1], color='red', s=50, alpha=0.5, label="Aux")
        try:
            ax.scatter(pop_in_aux_obj[:, 0], pop_in_aux_obj[:, 1], color='w', s=15, edgecolors='red', label="Pop-in-aux")
        except:
            pass

        plt.legend(loc='best', frameon=False)

        plt.savefig(f'/home/juan/Desktop/ccmo_viz/ccmo_refdir_safr_eps_{self.problem.name}_gen_{self.n_gen-1}.png')
        # plt.show()
