import numpy as np
import copy

from optimisation.model.algorithm import Algorithm

from optimisation.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from optimisation.operators.survival.rank_and_crowding_survival import RankAndCrowdingSurvival
from optimisation.operators.survival.population_based_epsilon_survival import PopulationBasedEpsilonSurvival
from optimisation.operators.survival.rank_survival import RankSurvival

from optimisation.operators.selection.tournament_selection import TournamentSelection, binary_tournament

from optimisation.operators.mutation.polynomial_mutation import PolynomialMutation
from optimisation.operators.mutation.no_mutation import NoMutation
from optimisation.operators.mutation.uniform_mutation import UniformMutation
from optimisation.operators.mutation.non_uniform_mutation import NonUniformMutation
from optimisation.model.swarm import Swarm
from optimisation.model.repair import InversePenaltyBoundsRepair, BasicBoundsRepair, BounceBackBoundsRepair
from optimisation.operators.replacement.improvement_replacement import ImprovementReplacement


class OMOPSO(Algorithm):

    def __init__(self,
                 n_population=100,
                 sampling=LatinHypercubeSampling(),
                 survival=RankAndCrowdingSurvival(),    # RankSurvival(),    #
                 leaders_survival=PopulationBasedEpsilonSurvival(),  # RankSurvival(),    #
                 selection=TournamentSelection(comp_func=binary_tournament),
                 w=0.5,     # 0.9,
                 c_1=2.0,
                 c_2=2.0,
                 adaptive=True,     # False,    #
                 initial_velocity=None,     # 'random',     #
                 max_velocity_rate=2.0,         # 0.5,         #
                 **kwargs):

        super().__init__(**kwargs)

        # Population parameters
        self.n_population = n_population

        # Generation parameters
        self.max_f_eval = (self.max_gen+1)*self.n_population

        # Population
        self.population = None

        # leaders archive
        self.leaders = None

        # Sampling
        self.sampling = sampling

        # Survival (used for evaluating global best)
        self.survival = survival
        self.leaders_survival = leaders_survival

        # Mutation
        self.mutation1 = UniformMutation(eta=20, prob=0.1)    # None,     #
        self.mutation2 = NoMutation()
        self.mutation3 = NonUniformMutation(eta=20, prob=0.1)

        # Mutation
        self.selection = selection

        # Adaptive flag
        self.adaptive = adaptive

        # With this set to true, each individual is only replaced by its new position if the solution improves
        self.replace_if_improved = False

        # Swarm parameters
        self.w = w      # The inertia during velocity computation (if adaptive=True, this is the initial value only)
        self.c_1 = c_1  # The cognitive impact (personal best) used during velocity computation (if adaptive=True,
        # this is the initial value only)
        self.c_2 = c_2  # The social impact (global best) used during velocity computation (if adaptive=True,
        # this is the initial value only)

        # Velocity terms
        self.initial_velocity = initial_velocity
        self.max_velocity_rate = max_velocity_rate
        self.v_max = None

        # Personal best and global best
        self.personal_best = None
        self.global_best = None

        # Optimum position
        self.opt = None

    def setup(self, problem, **kwargs):
        super().setup(problem, **kwargs)

        # Compute normalised max velocity
        self.v_max = self.max_velocity_rate*(problem.x_upper - problem.x_lower)

    def _initialise(self):

        # Instantiate population
        self.population = Swarm(self.problem, self.n_population)

        # Population initialisation
        if self.hot_start:
            # Initialise population using hot-start
            self.hot_start_initialisation()
        else:
            # Compute sampling
            self.sampling.do(self.n_population, self.problem.x_lower, self.problem.x_upper)

            # Assign sampled design variables to population
            self.population.assign_var(self.problem, self.sampling.x)
            if self.x_init:
                self.population[0].set_var(self.problem, self.problem.x_value)

        # Evaluate initial population
        self.population = self.evaluator.do(self.problem.obj_func, self.problem, self.population)

        # Calculate maximum constraint values across the population
        self.max_abs_con_vals = self.evaluator.calc_max_abs_cons(self.population, self.problem)

        # Assign initial velocity to population
        if self.initial_velocity == 'random':
            v_initial = np.random.random((self.n_population, self.problem.n_var))*self.v_max[None, :]
        else:
            v_initial = np.zeros((self.n_population, self.problem.n_var))
        self.population.assign_velocity(v_initial)

        # Assign rank and crowding to population
        self.population.assign_rank_and_crowding()

        # Create the archive
        self.leaders = copy.deepcopy(self.population)

    def _next(self):

        self._step()

        if self.adaptive:
            self._adapt()

    def _step(self):

        # Compute swarm local & global optima (and extract positions)
        # Personal best is calculated based on non-dominated sorting
        self.personal_best = self.population.compute_local_optima(self.problem)

        # TODO select leaders based on crowding distance

        # Global best is calculated based on the selected survival method
        self.opt, self.global_best = self.population.compute_global_optimum(self.problem, self.n_population,
                                                                                 survival=self.survival)

        # update the archive of leaders
        # Create offspring
        leader_temp = Swarm(self.problem, len(self.population))

        # Assign new positions & velocities
        leader_position = self.population.extract_var()
        leader_temp.assign_var(self.problem, leader_position)

        # combine the personal best and the previous leaders
        self.leaders = Swarm.merge(self.leaders, leader_temp)

        # Conduct survival selection
        self.leaders = self.leaders_survival.do(self.problem, self.leaders, self.n_population, self.n_gen, self.max_gen)

        # Extract current position and velocity of each individual
        position = self.population.extract_var()
        velocity = self.population.extract_velocity()

        # Calculate inertia of each individual
        inertia = self.w*velocity

        # Calculate random values for directional computations
        # r_1 = np.random.random((self.n_population, self.problem.n_var))
        # r_2 = np.random.random((self.n_population, self.problem.n_var))

        r_1 = np.random.random(self.n_population)
        r_2 = np.random.random(self.n_population)
        # r_2 = np.repeat(r_2[np.newaxis, :], position.shape[0], axis=0).T
        r_2 = np.tile(r_2, (self.problem.n_var, 1)).T

        # need to rank them and use tournament selection based on crowding distance
        _leader_idx = self.selection.do(self.leaders, self.n_population, 1)
        # Calculate cognitive and social influence
        leader_position = self.leaders.extract_var()

        cognitive = copy.deepcopy(position)
        for i in range(self.n_population):
            # cognitive[i,:] = self.c_1 * r_1[i , :]* (leader_position[_leader_idx[i]] - position[i,:])
            cognitive[i, :] = self.c_1 * r_1[i] * (leader_position[_leader_idx[i]] - position[i, :])
        social = self.c_2* r_2* (self.global_best - position)

        # Calculate new velocity
        _velocity = inertia + cognitive + social
        for i in range(self.n_population):
            upper_mask = _velocity[i, :] > self.v_max
            _velocity[i, upper_mask] = self.v_max[upper_mask]

            lower_mask = _velocity[i, :] < -self.v_max
            _velocity[i, lower_mask] = -self.v_max[lower_mask]

        # Calculate new position of each particle
        _position = position + _velocity

        # Modify velocity if position exceeded bounds
        upper_mask = _position > self.problem.x_upper
        lower_mask = _position < self.problem.x_lower
        _velocity[np.logical_or(upper_mask, lower_mask)] *= -1.0

        # Repair positions if they exist outside variable bounds
        # BasicBoundsRepair sets variable to the bound
        _position = BasicBoundsRepair().do(self.problem, _position)
        # _position = BounceBackBoundsRepair().do(self.problem, _position)
        # _position = InversePenaltyBoundsRepair().do(self.problem, _position, parent_array=position)

        if self.replace_if_improved:
            # Create offspring
            offspring = Swarm(self.problem, len(self.population))

            # Assign new positions & velocities
            offspring.assign_var(self.problem, _position)
            offspring.assign_velocity(_velocity)

            # Evaluate the population at new positions
            offspring = self.evaluator.do(self.problem.obj_func, self.problem, offspring)
            offspring.compute_local_optima(self.problem)

            # Evaluate whether a solution has improved
            has_improved = ImprovementReplacement().do(self.problem, self.population, offspring, return_indices=True)

            # Replace solutions that have improved
            self.population[has_improved] = offspring[has_improved]
        else:
            # Assign new positions & velocities
            self.population.assign_var(self.problem, _position)
            self.population.assign_velocity(_velocity)

            # Evaluate the population at new positions
            # TODO turn back on to test if it helps - but is not in the original omopso article
            # self.population = self.evaluator.do(self.problem.obj_func, self.problem, self.population)

        # Mutate the population
        # need 3 mutations
        _population1 = copy.deepcopy(self.population)
        _population2 = copy.deepcopy(self.population)
        _population3 = copy.deepcopy(self.population)

        _population1 = self.mutation1.do(self.problem, _population1)
        _population2 = self.mutation2.do(self.problem, _population2)
        _population3 = self.mutation3.do(self.problem, _population3, current_iteration=self.n_gen,
                                    max_iterations=self.max_gen)

        # Create Mask for mutations
        random_numbers_for_mask = np.random.random(self.n_population)
        do_mutation1 = np.zeros(len(self.population), dtype=bool)
        do_mutation2 = np.zeros(len(self.population), dtype=bool)
        do_mutation3 = np.zeros(len(self.population), dtype=bool)

        for i in range(len(self.population)):
            if random_numbers_for_mask[i] <= 1/3:
                do_mutation1[i] = True
            elif random_numbers_for_mask[i] <= 2/3:
                do_mutation2[i] = True
            else:
                do_mutation3[i] = True

        self.population[do_mutation1] = _population1[do_mutation1]
        self.population[do_mutation2] = _population1[do_mutation2]
        self.population[do_mutation3] = _population1[do_mutation3]

        # Evaluate the population at new positions
        self.population = self.evaluator.do(self.problem.obj_func, self.problem, self.population)

        # Assign rank and crowding to population
        self.population.assign_rank_and_crowding()

    def _adapt(self):

        # Here using the platypus technique of randomising each iteration
        self.w = np.random.uniform(0.1, 0.5)
        self.c_1 = np.random.uniform(1.5, 2.0)
        self.c_2 = np.random.uniform(1.5, 2.0)

