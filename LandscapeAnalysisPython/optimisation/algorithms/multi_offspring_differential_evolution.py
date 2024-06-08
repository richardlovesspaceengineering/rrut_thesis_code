import numpy as np
import copy

from optimisation.algorithms.evolutionary_algorithm import EvolutionaryAlgorithm
from optimisation.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from optimisation.operators.mutation.de_mutation import DifferentialEvolutionMutation

from optimisation.operators.selection.random_selection import RandomSelection
from optimisation.operators.crossover.differential_evolution_crossover import DifferentialEvolutionCrossover
from optimisation.operators.crossover.binomial_crossover import BiasedCrossover
from optimisation.operators.crossover.exponential_crossover import ExponentialCrossover

from optimisation.model.population import Population
from optimisation.model.repair import BounceBackBoundsRepair
from optimisation.operators.replacement.improvement_replacement import ImprovementReplacement
from optimisation.operators.survival.rank_and_crowding_survival import RankAndCrowdingSurvival

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-talk')
# matplotlib.use('TkAgg')
np.set_printoptions(suppress=True)


class MultiOffspringDE(EvolutionaryAlgorithm):

    def __init__(self,
                 n_population=100,
                 sampling=LatinHypercubeSampling(),
                 survival=RankAndCrowdingSurvival(),
                 crossover=None,
                 mutation=None,
                 surrogate=None,
                 **kwargs):

        # Initialise Parent Object
        super().__init__(n_population=n_population,
                         sampling=sampling,
                         selection=None,
                         crossover=crossover,
                         mutation=mutation,
                         survival=None,
                         n_offspring=n_population,
                         **kwargs)

        # Initialise parameters
        self.sampling = sampling
        self.survival = survival
        self.n_population = n_population
        self.repair = BounceBackBoundsRepair()
        # self.n_generation = n_generation
        self.init_population = kwargs['init_population']

        # self.surrogate = surrogate

        # Memory for f and CR
        self.f = [0.8, 1.0, 1.0, 0.8, 0.8, 0.9, 0.4, 0.4]
        self.cr = [0.2, 0.1, 0.9, 0.8, 0.6, 0.2, 0.2, 0.9]

    def _initialise(self):

        # Initialise from the evolutionary algorithm class
        super()._initialise()

        # Initialisation MODE Population with LHS and Initial Population
        seed = np.random.randint(low=0, high=10000, size=(1,))
        self.sampling.do(self.n_population, x_lower=self.problem.x_lower, x_upper=self.problem.x_upper, seed=seed)
        x_init = self.sampling.x
        de_population = Population(self.problem, n_individuals=len(x_init))
        de_population.assign_var(self.problem, x_init)
        de_population = Population.merge(de_population, copy.deepcopy(self.init_population))
        de_population = self.evaluator.do(self.problem.obj_func, self.problem, de_population)
        self.population = de_population
        self.pareto_archive = copy.deepcopy(self.population)

        # DE multioffspring mutation
        self.create_offspring = DifferentialEvolutionMutation(F=self.f, Cr=self.cr, problem=self.problem)

    def _next(self):

        # Perform the next iteration
        self._step()

    def _step(self):

        # Create Offspring
        offspring_population = self.create_offspring.do(self.population, self.pareto_archive)

        # Repair out-of-bounds
        offspring_population = self.repair.do(self.problem, offspring_population)

        # Evaluate Offspring 
        offspring_population = self.evaluator.do(self.problem.obj_func, self.problem, offspring_population)

        # Top Ranking Survival
        copy_pop = copy.deepcopy(offspring_population)
        offspring_population = self.survival.do(self.problem, offspring_population, self.n_population, self.n_gen, self.max_gen)
        # offspring_population = self._objective_rank_survival(offspring_population, self.n_population, flag='FR')  # RANKING ATTEMPT
        self.population = copy.deepcopy(offspring_population)

        # Update Archive
        self.pareto_archive = self._update_archive(self.pareto_archive, offspring_population, self.n_population)

        # DEBUG PURPOSES
        # self.plot_pf(final_population=copy_pop, infill_population=offspring_population, plot_pf=False)

    def _update_archive(self, population, offspring, population_size):
        merged_population = Population.merge(population, offspring)
        merged_population = self.survival.do(self.problem, merged_population, population_size,
                                             gen=self.n_gen, max_gen=self.max_gen)
        return merged_population

    def _objective_rank_survival(self, population, n_survive, flag='avg', weights=[0.4, 0.6]):
        """
        Select from flag = ['avg', 'min', 'comb', 'FR']
        """
        # Extract objective array
        population = copy.deepcopy(population)
        obj_array = population.extract_obj()

        # Get ranks of each objective column
        rank_array = np.argsort(obj_array, axis=0).argsort(axis=0)

        # Combined ranks by sum / minimum
        avg_rank = np.sum(rank_array, axis=1)
        min_rank = np.min(rank_array, axis=1)

        # Result n_survive individuals with lowest combined ranks
        # survived_indices = np.argpartition(rank_sum, n_survive)[:n_survive]
        if 'avg' in flag:
            survived_indices = np.argsort(avg_rank)[:n_survive]
        elif 'min' in flag:
            survived_indices = np.argsort(min_rank)[:n_survive]
        elif 'comb' in flag:
            half_n_survive = np.floor(n_survive / 2).astype('int')
            indices1 = np.argsort(avg_rank)[:half_n_survive]
            half_n_survive = int(n_survive - half_n_survive)
            indices2 = np.argsort(min_rank)[:half_n_survive]
            survived_indices = np.hstack((indices1, indices2))

        elif 'FR' in flag:
            # Calculate GD metric
            gd_rank = self._calculate_gd(obj_array)
            # survived_indices = np.argsort(gd_rank)[:n_survive]

            # Fusion Ranking (Li2016d)
            fr_rank = weights[0]*avg_rank + weights[1]*gd_rank
            survived_indices = np.argsort(fr_rank)[:n_survive]
        else:
            raise Exception('no appropriate ranking method provided')

        return population[survived_indices]

    def plot_pf(self, final_population=None, infill_population=None, extreme_population=None, line=None, dividers=None, plot_pf=True):

        # Initialise Plot
        fig, ax = plt.subplots(1, 1, figsize=(9, 7))
        fig.supxlabel('Obj 1', fontsize=14)
        fig.supylabel('Obj 2', fontsize=14)

        if final_population is not None:
            final_vars = final_population.extract_obj()
            # obj_min = np.min(final_vars, axis=0)
            # obj_max = np.max(final_vars, axis=0)
            # final_vars = (final_vars - obj_min) / (obj_max - obj_min)
            ax.scatter(final_vars[:, 0], final_vars[:, 1], color='blue', s=75, label='Non-dominated Front')

        if infill_population is not None:
            infill_vars = np.atleast_2d(infill_population.extract_obj())
            # obj_min = np.min(infill_vars, axis=0)
            # obj_max = np.max(infill_vars, axis=0)
            # infill_vars = (infill_vars - obj_min) / (obj_max - obj_min)
            ax.scatter(infill_vars[:, 0], infill_vars[:, 1], color='red', s=25, label='Knee-point Selection')

        if plot_pf:
            pareto_front = self.problem.variables['x_vars'][0].f_opt
            ax.plot(pareto_front[:, 0], pareto_front[:, 1], '-k')

        # ax.set_title(problem + ' ' + str(dim) + 'D, Distance-based')
        plt.legend(loc='best', frameon=False)
        plt.show()

    @staticmethod
    def _calculate_gd(obj_array):

        gd = np.zeros(len(obj_array))
        # Loop through all individuals
        for idx in range(len(obj_array)):
            obj_diff = obj_array[idx] - obj_array
            gd[idx] = np.sum(np.sum(np.maximum(obj_diff, 0), axis=1))

        return gd