import numpy as np
from copy import copy
from scipy.spatial.distance import cdist

from optimisation.algorithms.evolutionary_algorithm import EvolutionaryAlgorithm
from optimisation.model.individual import Individual
from optimisation.model.population import Population
from optimisation.operators.sampling.random_sampling import RandomSampling
from optimisation.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from optimisation.operators.mutation.polynomial_mutation import PolynomialMutation
from optimisation.operators.decomposition.get import get_decomposition


class MOEAD(EvolutionaryAlgorithm):

    def __init__(self,
                 ref_dirs,
                 n_population=100,
                 n_neighbours=20,
                 decomposition='auto',
                 prob_neighbour_mating=0.9,
                 sampling=RandomSampling(),
                 crossover=SimulatedBinaryCrossover(eta=20, prob=1.0),
                 mutation=PolynomialMutation(eta=20, prob=None),
                 **kwargs):

        self.ref_dirs = ref_dirs
        self.n_neighbours = n_neighbours
        self.prob_neighbour_mating = prob_neighbour_mating
        self.decomposition = decomposition

        if self.ref_dirs.shape[0] < self.n_neighbours:
            print('Setting number of neighbours to population size: %s' % self.ref_dirs.shape[0])
            self.n_neighbours = self.ref_dirs.shape[0]

        super().__init__(n_population=n_population,
                         sampling=sampling,
                         selection=None,
                         crossover=crossover,
                         mutation=mutation,
                         survival=None,
                         **kwargs)

        self.neighbours = np.argsort(cdist(self.ref_dirs, self.ref_dirs), axis=1, kind='quicksort')[:, :self.n_neighbours]

        self.ideal_point = None

    def _initialise(self):

        if type(self.decomposition) == str:

            decomp_str = self.decomposition

            # Use chebychev for <= 2 objectives
            if decomp_str == 'auto':
                if self.problem.n_obj <= 2:
                    decomp_str = 'chebychev'
                else:
                    decomp_str = 'pbi'

            # Decomposition instance
            self.decomposition = get_decomposition(decomp_str)

        else:
            self.decomposition = self.decomposition

        super()._initialise()
        self.ideal_point = np.min(self.population.extract_obj(), axis=0)

    def _next(self):

        # Iterate over population in random order
        for i in np.random.permutation(len(self.population)):

            # Calculate neighbours of the current individual
            current_neighbours = self.neighbours[i, :]

            # Extract parents from neighbours or population
            if np.random.random() < self.prob_neighbour_mating:
                parents = current_neighbours[np.random.permutation(self.n_neighbours)][:self.mating.crossover.n_parents]
            else:
                parents = np.random.permutation(self.n_population)[:self.mating.crossover.n_parents]

            # Conducting mating process
            self.offspring = self.mating.crossover.do(self.problem, self.population, parents[None, :])
            self.offspring = self.mating.mutation.do(self.problem, self.offspring)

            # Select one offspring randomly
            self.offspring = self.offspring[np.random.randint(0, len(self.offspring))]

            # Ensure self.offspring is a Population instance (selecting one offspring extracts an Individual instance)
            if isinstance(self.offspring, Individual):
                temp = copy(self.offspring)
                self.offspring = Population(self.problem, n_individuals=1)
                self.offspring[0] = temp

            # Set variable dicts for each offspring
            for j in range(len(self.offspring)):
                self.offspring[j].set_var(self.problem, self.offspring[j].var)

            # Evaluate offspring
            self.offspring = self.evaluator.do(self.problem.obj_func, self.problem, self.offspring, self.max_abs_con_vals)

            # Update the ideal point
            offspring_obj_array = self.offspring.extract_obj()
            self.ideal_point = np.min(np.vstack([self.ideal_point, offspring_obj_array]), axis=0)

            # Calculate the decomposed values for each neighbour
            temp = self.population[current_neighbours]
            current_neighbour_obj_arry = temp.extract_obj()
            fv = self.decomposition.do(current_neighbour_obj_arry,
                                       weights=self.ref_dirs[current_neighbours, :],
                                       ideal_point=self.ideal_point)
            offspring_fv = self.decomposition.do(offspring_obj_array,
                                                 weights=self.ref_dirs[current_neighbours, :],
                                                 ideal_point=self.ideal_point)

            # Calculate the index in the decomposed space where the offspring fitness is better than the current
            # population fitness
            idxs = np.where(offspring_fv < fv)[0]

            # Replace individuals in population with offspring where required (note offspring is always singular?)
            self.population[current_neighbours[idxs]] = self.offspring

        # Assign rank and crowding to population
        self.population.assign_rank_and_crowding()
