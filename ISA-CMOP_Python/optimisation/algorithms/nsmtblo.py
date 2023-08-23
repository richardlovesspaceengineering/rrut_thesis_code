import numpy as np

from optimisation.model.algorithm import Algorithm

from optimisation.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from optimisation.operators.selection.roulette_wheel_selection import RouletteWheelSelection
from optimisation.operators.selection.tournament_selection import TournamentSelection
from optimisation.operators.survival.rank_and_crowding_survival import RankAndCrowdingSurvival
from optimisation.model.population import Population
from optimisation.model.repair import BasicBoundsRepair
from optimisation.operators.replacement.improvement_replacement import ImprovementReplacement


class NSMTLBO(Algorithm):

    """
    Non-Dominated Sorting Modified Teaching-Learning-Based Optimisation
    https://doi.org/10.1007/s10845-019-01486-9

    Alternative teacher selection for non-dominated individuals
    https://doi.org/10.1016/j.asoc.2017.08.056

    Logarithmic spiral in the teaching phase
    https://doi.org/10.1007/s00521-018-3785-6
    """

    def __init__(self,
                 n_population=100,
                 sampling=LatinHypercubeSampling(),
                 selection=RouletteWheelSelection(),
                 survival=RankAndCrowdingSurvival(),
                 replacement=ImprovementReplacement(),
                 repair=BasicBoundsRepair(),
                 surrogate=None,
                 **kwargs):

        super().__init__(**kwargs)

        # Population parameters
        self.n_population = n_population

        # Generation parameters
        self.max_f_eval = (self.max_gen+1)*self.n_population

        # Population
        self.population = None

        # Surrogate strategy instance
        self.surrogate = surrogate

        # Sampling
        self.sampling = sampling

        # Selection (used for selection of teachers for non-dominated individuals)
        self.selection = selection

        # Survival
        self.survival = survival

        # Replacement
        self.replacement = replacement

        # Repair
        self.repair = repair

        # Probability of mutation (self-learning)
        self.p_mutation = 0.5

        # Flag to apply teaching, learning, self-learning & mutation sequentially on the offspring with evaluation
        # after each operation
        self.apply_operators_sequentially = False

        # Flag to apply a traditional survival method, whereby the population and offspring are merged, and the
        # individuals with the highest fitness survive
        self.apply_survival = True

        # Flag to apply a logarithmic spiral strategy in the teaching phase
        self.apply_spiral = True

        # Optimum position
        self.opt = None

    def _initialise(self):

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
            self.sampling.do(self.n_population, self.problem.x_lower, self.problem.x_upper)

            # Assign sampled design variables to population
            self.population.assign_var(self.problem, self.sampling.x)
            if self.x_init:
                self.population[0].set_var(self.problem, self.problem.x_value)

        # Evaluate initial population
        if self.surrogate is not None:
            self.population = self.evaluator.do(self.surrogate.obj_func, self.problem, self.population)
        else:
            self.population = self.evaluator.do(self.problem.obj_func, self.problem, self.population)

        # Dummy survival call to ensure population is ranked
        if self.survival:
            self.population = self.survival.do(self.problem, self.population, self.n_population, self.n_gen, self.max_gen)

        # Calculate maximum constraint values across the population
        self.max_abs_con_vals = self.evaluator.calc_max_abs_cons(self.population, self.problem)

        # Assign rank and crowding to population
        self.population.assign_rank_and_crowding()

    def _next(self):

        self._step()

        # Dummy survival call to ensure population is ranked
        if self.survival:
            self.population = self.survival.do(self.problem, self.population, self.n_population, self.n_gen, self.max_gen)

        # Assign rank and crowding to population
        self.population.assign_rank_and_crowding()

    def _step(self):

        # Extract current position and velocity of each individual
        position = self.population.extract_var()

        if self.apply_operators_sequentially:

            # Teaching
            _position = self.teaching(position)
            self.conduct_replacement(_position)

            # Learning
            _position = self.learning(self.population.extract_var())
            self.conduct_replacement(_position)

        else:

            if self.n_gen % 2 == 0:
                # Teaching
                _position = self.teaching(position)
            else:
                # Learning
                _position = self.learning(position)

        # Repair positions if they exist outside variable bounds
        _position = self.repair.do(self.problem, _position)

        if not self.apply_operators_sequentially:
            if self.apply_survival:
                self.conduct_survival(_position)
            else:
                self.conduct_replacement(_position)

    def teaching(self, position):

        _position = np.zeros(np.shape(position))

        if self.apply_spiral:
            _position1 = np.zeros(np.shape(position))
            _position2 = np.zeros(np.shape(position))

        pop_rank = self.population.extract_rank()
        non_dominated_idx = [idx for idx, rank in enumerate(pop_rank) if rank == 0]
        if len(non_dominated_idx) == 1:
            non_dominated_idx = [idx for idx, rank in enumerate(pop_rank) if rank <= 1]

        # Extract sub-population of non-dominated individuals
        if len(non_dominated_idx) > 0:
            nd_pop = self.population[non_dominated_idx]
        else:
            nd_pop = self.population[:]

        # Random factors to preserve diversity when calculating variable mean of non-dominated front
        r = np.random.uniform(0.0, 1.0, len(nd_pop))

        # Random variables
        r_3 = np.random.uniform(0.0, 1.0, self.n_population)
        r_4 = np.random.uniform(0.0, 1.0, self.n_population)

        # Teaching factors
        t_f_1 = np.random.uniform(1.0, 2.0, self.n_population)
        t_f_2 = np.random.uniform(1.0, 2.0, self.n_population)

        # Extract position of non-dominated front
        nd_position = nd_pop.extract_var()
        # Extract crowding distance of non-dominated front
        nd_crowding = nd_pop.extract_crowding()

        # Determine non-dominated selection probabilities
        # TODO is the issue here? Check the original article and my matlab version
        # should make sure that more than one teacher is selected
        # possibly change over to hypervolume probability?
        nd_selection_prob = self.calculate_cd_probabilities(nd_pop, nd_crowding)

        # Calculate variable mean of non-dominated front
        weighted_mean_nd_position = np.sum(nd_position*np.tile(r, (self.problem.n_var, 1)).conj().transpose(), axis=0)/np.sum(r, axis=0)

        dist_to_nd = np.zeros((self.n_population, len(nd_pop)))
        for i in range(self.n_population):

            if pop_rank[i] == 0:
                if self.selection:
                    # Determine teacher using roulette wheel selection
                    # nd_teacher_idx = self.selection.do(nd_pop, 1, 1)
                    nd_teacher_idx = self.selection.do(nd_pop, 1, 1, probabilities=nd_selection_prob)[0][0]
                else:
                    # Determine teacher using highest crowding distance across the non-dominated front
                    temp_nd_idx_cd = np.argsort(nd_crowding)
                    if temp_nd_idx_cd[0] == i:
                        nd_teacher_idx = temp_nd_idx_cd[1]
                    else:
                        nd_teacher_idx = temp_nd_idx_cd[0]
            else:
                # Calculate position (in design space) from individual i to each individual in non-dominated front
                for j in range(len(nd_pop)):
                    dist_to_nd[i, j] = np.sqrt(np.sum((position[i, :] - nd_position[j, :])**2.0))

                # Teacher is closest non-dominated individual (in design space)
                non_zero_dist_to_nd_mask = [idx for idx in range(len(nd_pop)) if dist_to_nd[i, idx] > 0.0]
                temp_nd_idx = np.argmin(dist_to_nd[i, :][non_zero_dist_to_nd_mask])
                nd_teacher_idx = non_zero_dist_to_nd_mask[temp_nd_idx]

            # Calculate new position
            if self.apply_spiral:
                _position1[i, :] = position[i, :] + r_3[i] * (nd_position[nd_teacher_idx, :] - t_f_1[i] * position[i, :]) \
                                  + r_4[i] * (weighted_mean_nd_position - t_f_2[i] * position[i, :])
                theta = 2*(1 - self.n_gen/self.max_gen) -1
                _position2[i,:] = abs((nd_position[nd_teacher_idx, :] - position[i, :])) * np.exp(theta) * \
                                  np.cos(2 * np.pi * theta) + nd_position[nd_teacher_idx, :]
                k = np.random.uniform(0.0, 1.0, position[1,:].shape[0])
                _position[i, :] = k * _position1[i, :] + (1-k) * _position2[i, :]

            else:
                _position[i, :] = position[i, :] + r_3[i]*(nd_position[nd_teacher_idx, :] - t_f_1[i]*position[i, :]) \
                                             + r_4[i]*(weighted_mean_nd_position - t_f_2[i]*position[i, :])

        return _position

    def learning(self, position):

        _position = np.zeros(np.shape(position))

        self_learning_mask = np.random.random(self.n_population) < self.p_mutation

        _position[self_learning_mask] = self.self_learning(position[self_learning_mask])
        _position[~self_learning_mask] = self.domination_learning(self.population[~self_learning_mask],
                                                                  position[~self_learning_mask])

        return _position

    def domination_learning(self, pop, position):

        _position = np.zeros(np.shape(position))

        # Random variable
        r = np.random.random(len(position))

        for idx in range(len(position)):

            # Select random individual from the rest of the (passed) population
            l_idx = idx
            while l_idx == idx:
                l_idx = np.random.randint(1, len(position))

            # Calculate new position
            if pop[l_idx].rank < pop[idx].rank:     # The current individual dominates the other individual
                _position[idx, :] = position[idx, :] + r[idx]*(position[idx, :] - position[l_idx, :])
            elif pop[l_idx].rank > pop[idx].rank:   # The other individual dominates the current individual
                _position[idx, :] = position[idx, :] + r[idx]*(position[l_idx, :] - position[idx, :])
            else:
                _r = np.random.uniform(0.0, 1.0)
                if _r < 0.5:
                    _position[idx, :] = position[idx, :] + r[idx]*(position[idx, :] - position[l_idx, :])
                else:
                    _position[idx, :] = position[idx, :] + r[idx]*(position[l_idx, :] - position[idx, :])

        return _position

    def self_learning(self, position):

        _position = np.zeros(np.shape(position))

        # Random variables
        r_5 = np.random.uniform(-1.0, 1.0, len(position))
        k = np.random.randint(0, self.problem.n_var, len(position))

        for idx in range(len(position)):

            # Calculate new position
            _position[idx, k[idx]] = position[idx, k[idx]] + r_5[idx]*(self.problem.x_upper[k[idx]] - self.problem.x_lower[k[idx]])

        return _position

    def conduct_survival(self, _position):

        # Create offspring
        offspring = Population(self.problem, self.n_population)

        # Assign new positions & velocities
        offspring.assign_var(self.problem, _position)

        # Evaluate the population at new positions
        if self.surrogate is not None:
            self.offspring = self.evaluator.do(self.surrogate.obj_func, self.problem, offspring)
        else:
            self.offspring = self.evaluator.do(self.problem.obj_func, self.problem, offspring)

        # Merge the offspring with the current population
        self.population = Population.merge(self.population, offspring)

        # Conduct survival selection
        self.population = self.survival.do(self.problem, self.population, self.n_population, self.n_gen, self.max_gen)

    def conduct_replacement(self, _position):

        # Create offspring
        offspring = Population(self.problem, self.n_population)

        # Assign new positions & velocities
        offspring.assign_var(self.problem, _position)

        # Evaluate the population at new positions
        offspring = self.evaluator.do(self.problem.obj_fun, self.problem, offspring)

        # Evaluate whether a solution has improved
        has_improved = self.replacement.do(self.problem, self.population, offspring, return_indices=True)

        # Replace solutions that have improved
        self.population[has_improved] = offspring[has_improved]

    @staticmethod
    def calculate_cd_probabilities(nd_pop, nd_crowding):

        # Initialise lambda parameter
        cd_lambda = np.ones(len(nd_crowding))

        # Set distance of boundary points to the maximum crowding distance elsewhere on the front
        nd_boundary_mask = nd_crowding == 1.0e14
        nd_crowding[nd_boundary_mask] = np.amax(nd_crowding[~nd_boundary_mask])

        # Check for identically zero crowding distances across non-dominated front
        if nd_crowding.all() == 0.0:
            nd_crowding = np.ones(len(nd_crowding))
        # Check for infinite crowding distances and set to nd_boundary_mask (only happens with singular front)
        if sum(np.isinf(nd_crowding)) > 0:
            nd_infinity_mask = np.isinf(nd_crowding)
            nd_crowding[nd_infinity_mask] = 1.0e5


        # Set lambda values of boundary points
        cd_lambda[nd_boundary_mask] = 1.5

        # Determine u-rank across non-dominated front
        nd_u_rank = np.linspace(1, len(nd_pop), len(nd_pop))[::-1]

        # Calculate probability for roulette wheel selection
        nd_selection_prob = cd_lambda * nd_u_rank * nd_crowding / np.sum(nd_crowding)
        # Normalise probability to ensure valid CDF
        nd_selection_prob /= np.sum(nd_selection_prob)

        return nd_selection_prob

