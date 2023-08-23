import copy

import numpy as np

from optimisation.model.population import Population
from optimisation.algorithms.evolutionary_algorithm import EvolutionaryAlgorithm

from optimisation.operators.sampling.random_sampling import RandomSampling
from optimisation.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from optimisation.operators.selection.tournament_selection import TournamentSelection, comp_by_cv_then_random
from optimisation.operators.selection.random_selection import RandomSelection
from optimisation.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from optimisation.operators.mutation.polynomial_mutation import PolynomialMutation
from optimisation.operators.survival.adp_survival import APDSurvival
from optimisation.model.duplicate import DefaultDuplicateElimination
from optimisation.operators.survival.rank_and_crowding_survival import RankAndCrowdingSurvival

from optimisation.util.misc import calc_V

import matplotlib.pyplot as plt
import matplotlib
plt.style.use('seaborn-talk')
np.set_printoptions(suppress=True)
matplotlib.use('TkAgg')
line_colors = ['green', 'blue', 'red', 'orange', 'cyan', 'lawngreen', 'm', 'orangered','sienna', 'gold', 'violet', 'indigo', 'cornflowerblue']


class RVEA(EvolutionaryAlgorithm):
    """
    RVEA algorithm:
    Cheng2016 "A Reference Vector Guided Evolutionary Algorithm for Many-Objective Optimization"
    """
    def __init__(self,
                 ref_dirs=None,
                 n_population=100,
                 sampling=RandomSampling(),
                 # selection=TournamentSelection(comp_func=comp_by_cv_then_random),
                 selection=RandomSelection(),
                 crossover=SimulatedBinaryCrossover(eta=30, prob=1.0),
                 mutation=PolynomialMutation(eta=20, prob=None),
                 eliminate_duplicates=DefaultDuplicateElimination(),
                 **kwargs):

        self.ref_dirs = ref_dirs

        # Have to define here given the need to pass ref_dirs
        if 'survival' in kwargs:
            survival = kwargs['survival']
            del kwargs['survival']
        else:
            survival = APDSurvival(self.ref_dirs, filter_infeasible=True)  # True works for feasible start / False does not work

        if 'alpha' in kwargs:
            self.alpha = kwargs['alpha']
        else:
            self.alpha = 2.0

        if 'adapt_fr' in kwargs:
            self.adapt_fr = kwargs['adapt_fr']
        else:
            self.adapt_fr = 0.1

        super().__init__(n_population=n_population,
                         sampling=sampling,
                         selection=selection,
                         crossover=crossover,
                         mutation=mutation,
                         survival=survival,
                         eliminate_duplicates=eliminate_duplicates,
                         **kwargs)

        # Reference Vectors
        self.ideal, self.nadir = None, None
        self.V = calc_V(self.ref_dirs)
        self.survival.V = copy.deepcopy(self.V)

    def _next(self):

        # Initialise Reference Vectors
        if self.ideal is None:
            self.ideal = np.full(self.problem.n_obj, np.inf)

        # Conduct surrogate model refinement (adaptive sampling)
        if self.surrogate is not None:
            self.surrogate.run(self.problem)

        # Conduct mating using the current population
        self.offspring = self.mating.do(self.problem, self.population, self.n_offspring)
        # print('n_offspring: ', len(self.offspring))

        # Evaluate offspring
        if self.surrogate is not None:
            self.offspring = self.evaluator.do(self.surrogate.obj_func, self.problem, self.offspring, self.max_abs_con_vals)
        else:
            self.offspring = self.evaluator.do(self.problem.obj_func, self.problem, self.offspring, self.max_abs_con_vals)

        # Merge the offspring with the current population
        self.population = Population.merge(self.population, self.offspring)
        old_population = copy.deepcopy(self.population)

        # Conduct survival selection
        self.population = self.survival.do(self.problem, self.population, self.n_population, self.n_gen, self.max_gen)
        # print('n_survived_population: ', len(self.population))

        # Conduct reference vector adaptation
        if self.n_gen % np.ceil(self.max_gen * self.adapt_fr) == 0:
            # print(self.survival.niches)
            # self.plot_pareto(ref_vec=self.survival.V, selected=self.population, population=self.offspring)

            self.adapt()

        # Updating population normalised constraint function values
        if self.problem.n_con > 0:
            # Update maximum constraint values across the population
            self.max_abs_con_vals = self.evaluator.calc_max_abs_cons(self.population, self.problem)
            self.population = self.evaluator.sum_normalised_cons(self.population, self.problem, max_abs_con_vals=self.max_abs_con_vals)

        # Update optimum
        self.opt = self.n_population

    def adapt(self):
        # Re-calculate ideal and nadir points
        obj_arr = self.population.extract_obj()
        self.ideal = np.minimum(obj_arr.min(axis=0), self.ideal)
        self.nadir = obj_arr.max(axis=0)

        # Scale old vectors
        self.V = calc_V(calc_V(self.ref_dirs) * (self.nadir - self.ideal))

        # Pass onto survival class
        self.survival.V = copy.deepcopy(self.V)

    def plot_pareto(self, ref_vec=None, selected=None, population=None):

        # Initialise Plot
        fig, ax = plt.subplots(1, 1, figsize=(9, 7))
        fig.supxlabel('Obj 1', fontsize=14)
        fig.supylabel('Obj 2', fontsize=14)

        # Plot reference vectors
        if ref_vec is not None:
            scaling = 1.0
            origin = np.zeros(len(ref_vec))
            x_vec = scaling * np.vstack((origin, ref_vec[:, 0])).T
            y_vec = scaling * np.vstack((origin, ref_vec[:, 1])).T
            for i in range(len(x_vec)):
                if i == 0:
                    ax.plot(x_vec[i], y_vec[i], color='black', linewidth=2, label='reference vectors')
                else:
                    ax.plot(x_vec[i], y_vec[i], color='black', linewidth=2)

        if population is not None:
            pop_vars = np.atleast_2d(selected.extract_obj())
            # obj_min = np.min(selected_vars, axis=0)
            # obj_max = np.max(selected_vars, axis=0)
            # infill_vars = (selected_vars - obj_min) / (obj_max - obj_min)
            ax.scatter(pop_vars[:, 0], pop_vars[:, 1], color='blue', s=75, label='Population & Offspring')

        if selected is not None:
            selected_vars = np.atleast_2d(selected.extract_obj())
            # obj_min = np.min(selected_vars, axis=0)
            # obj_max = np.max(selected_vars, axis=0)
            # infill_vars = (selected_vars - obj_min) / (obj_max - obj_min)
            ax.scatter(selected_vars[:, 0], selected_vars[:, 1], color='red', s=25, label='APD Selection')

        # ax.set_title(problem + ' ' + str(dim) + 'D, Distance-based')
        # ax.set_aspect('equal')
        # ax.set_xlim((-0.2, 1.2))
        # ax.set_ylim((-0.2, 1.2))
        plt.legend(loc='best', frameon=False)
        plt.show()
        # plt.pause(3)
        # plt.close()
        # plt.savefig('/home/juan/PycharmProjects/optimisation_framework/multi_obj/results/zdt1_mmode_gen_' + str(len(self.surrogate.population)) + '.png')
