import numpy as np

from optimisation.model.algorithm import Algorithm

from optimisation.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from optimisation.operators.survival.rank_and_crowding_survival import RankAndCrowdingSurvival
from optimisation.model.population import Population
from optimisation.model.repair import BasicBoundsRepair
from optimisation.operators.replacement.improvement_replacement import ImprovementReplacement


class RTLBO(Algorithm):

    """
    Reformative Teaching-Learning-Based Optimisation
    https://doi.org/10.1007/s00500-020-04918-4
    """

    def __init__(self,
                 n_population=None,
                 sampling=LatinHypercubeSampling(),
                 survival=RankAndCrowdingSurvival(),
                 replacement=ImprovementReplacement(),
                 repair=BasicBoundsRepair(),
                 surrogate=None,
                 **kwargs):

        super().__init__(**kwargs)

        # Population parameters
        if n_population is None:
            n_population = 100
        self.n_population = n_population

        # Generation parameters
        self.max_f_eval = (self.max_gen+1)*self.n_population

        # Population
        self.population = None

        # Surrogate strategy instance
        self.surrogate = surrogate

        # Sampling
        self.sampling = sampling

        # Survival
        self.survival = survival

        # Replacement
        self.replacement = replacement

        # Repair
        self.repair = repair

        # Flag to apply teaching, learning, self-learning & mutation sequentially on the offspring with evaluation
        # after each operation
        self.apply_operators_sequentially = False

        # Flag to apply a traditional survival method, whereby the population and offspring are merged, and the
        # individuals with the highest fitness survive
        self.apply_survival = True

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

        # Assign rank and crowding to population
        self.population.assign_rank_and_crowding()

        # Calculate maximum constraint values across the population
        self.max_abs_con_vals = self.evaluator.calc_max_abs_cons(self.population, self.problem)

    def _next(self):

        # Conduct surrogate model refinement (adaptive sampling)
        if self.surrogate is not None:
            self.surrogate.run(self.problem)

        self._step()

        # Dummy survival call to ensure population is ranked
        if self.survival:
            self.population = self.survival.do(self.problem, self.population, self.n_population, self.n_gen, self.max_gen)

        # Assign rank and crowding to population
        self.population.assign_rank_and_crowding()

    def _step(self):

        # Extract current position and velocity of each individual
        position = self.population.extract_var()

        # Compute global best & worst individuals
        dummy_pop = self.survival.do(self.problem, self.population, self.n_population)
        global_best = dummy_pop[0]
        global_worst = dummy_pop[-1]

        if self.apply_operators_sequentially:

            # Teaching
            _position = self.teaching(position, global_best, global_worst)
            self.conduct_replacement(_position)

            # Learning
            _position = self.learning(self.population, self.population.extract_var())
            self.conduct_replacement(_position)

            # Self-learning
            _position = self.self_learning(self.population.extract_var())
            self.conduct_replacement(_position)

            # Mutation
            _position = self.mutation(self.population.extract_var())
            self.conduct_replacement(_position)

        else:

            if self.n_gen % 4 == 0:
                _position = self.teaching(position, global_best, global_worst)
            elif self.n_gen % 4 == 1:
                # Learning
                _position = self.learning(self.population, position)
            elif self.n_gen % 4 == 2:
                # Self-learning
                _position = self.self_learning(position)
            else:
                # Mutation
                _position = self.mutation(position)

        # Repair positions if they exist outside variable bounds
        _position = self.repair.do(self.problem, _position)

        if not self.apply_operators_sequentially:
            if self.apply_survival:
                self.conduct_survival(_position)
            else:
                self.conduct_replacement(_position)

    def teaching(self, position, global_best, global_worst):

        _position = np.zeros(np.shape(position))

        # Compute mean variable set
        mean_position = np.mean(position, axis=0)

        for idx in range(self.n_population):

            # Random variables
            r = np.random.uniform(0.0, 1.0, 2)

            # Adaptive teaching factors
            t_f_1 = 2.0 / (1.0 + np.exp(-np.sum(np.abs(global_best.var - position[idx, :]))))
            t_f_2 = 2.0 / (1.0 + np.exp(-np.sum(np.abs(position[idx, :] - global_worst.var))))

            # Calculate new position
            _position[idx, :] = position[idx, :] + r[0] * (global_best.var - t_f_1 * mean_position) \
                                                 - r[1] * (global_worst.var - t_f_2 * mean_position)

        return _position

    def learning(self, pop, position):

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

        for idx in range(self.n_population):

            # Random variable
            r = np.random.uniform(0.0, 1.0)

            # Calculate new position
            _position[idx, :] = position[idx, :] + (r - 0.5)*((self.problem.x_upper - self.problem.x_lower)/self.n_population)

        return _position

    def mutation(self, position):

        _position = np.zeros(np.shape(position))

        # Scaling factor
        f = 0.5

        for idx in range(self.n_population):

            # Select three random individuals from the rest of the population
            o_idx = idx
            p_idx = idx
            q_idx = idx
            while o_idx == idx or p_idx == idx or q_idx == idx:
                o_idx = np.random.randint(1, self.n_population)
                p_idx = np.random.randint(1, self.n_population)
                q_idx = np.random.randint(1, self.n_population)

            # Calculate mutated position
            _position[idx, :] = position[o_idx, :] + f*(position[p_idx, :] - position[q_idx])

        return _position

    def conduct_survival(self, _position):

        # Create offspring
        offspring = Population(self.problem, self.n_population)

        # Assign new positions & velocities
        offspring.assign_var(self.problem, _position)

        # Evaluate the population at new positions
        if self.surrogate is not None:
            offspring = self.evaluator.do(self.surrogate.obj_func, self.problem, offspring)
        else:
            offspring = self.evaluator.do(self.problem.obj_func, self.problem, offspring)
        # offspring = self.evaluator.do(self.problem.obj_func, self.problem, offspring)

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
        if self.surrogate is not None:
            offspring = self.evaluator.do(self.surrogate.obj_func, self.problem, offspring)
        else:
            offspring = self.evaluator.do(self.problem.obj_func, self.problem, offspring)
        # offspring = self.evaluator.do(self.problem, offspring)

        # Evaluate whether a solution has improved
        has_improved = self.replacement.do(self.problem, self.population, offspring, return_indices=True)

        # Replace solutions that have improved
        self.population[has_improved] = offspring[has_improved]