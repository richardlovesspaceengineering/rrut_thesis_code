import numpy as np
import copy
import random

from optimisation.model.algorithm import Algorithm

from optimisation.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from optimisation.operators.survival.rank_and_crowding_survival import RankAndCrowdingSurvival
from optimisation.operators.survival.population_based_epsilon_survival import PopulationBasedEpsilonSurvival
from optimisation.operators.survival.two_ranking_survival import TwoRankingSurvival
from optimisation.operators.survival.rank_survival import RankSurvival
from optimisation.operators.survival.rank_and_hypervolume_survival import RankAndHypervolumeSurvival

from optimisation.operators.selection.tournament_selection import TournamentSelection, binary_tournament

from optimisation.operators.mutation.polynomial_mutation import PolynomialMutation
from optimisation.operators.mutation.no_mutation import NoMutation
from optimisation.operators.mutation.uniform_mutation import UniformMutation
from optimisation.operators.mutation.non_uniform_mutation import NonUniformMutation
from optimisation.model.swarm import Swarm
from optimisation.model.repair import InversePenaltyBoundsRepair, BasicBoundsRepair, BounceBackBoundsRepair
from optimisation.operators.replacement.improvement_replacement import ImprovementReplacement


class MDEPSO(Algorithm):

    def __init__(self,
                 n_population=100,
                 sampling=LatinHypercubeSampling(),
                 # survival=RankAndCrowdingSurvival(),    # RankSurvival(),    #
                 # survival=PopulationBasedEpsilonSurvival(),
                 # survival=TwoRankingSurvival(),
                 survival=RankAndHypervolumeSurvival(),
                 mutation=PolynomialMutation(eta=20, prob=0.01),
                 # mutation=NonUniformMutation(eta=20, prob=0.01),
                 selection=TournamentSelection(comp_func=binary_tournament),
                 w=0.7298,     # 0.9,
                 c_1=2.0,
                 c_2=2.0,
                 f_set=None,
                 cr_set=None,
                 adaptive=True,     # False,    #
                 initial_velocity='random',     # None,     #
                 max_velocity_rate=0.5,         # 0.5,         #
                 **kwargs):

        super().__init__(**kwargs)

        # Population parameters
        self.n_population = n_population

        # Generation parameters
        # self.max_gen = 2*self.max_gen # need to fix this. Due to copy.deepcopy of full_population
        self.max_f_eval = (self.max_gen+1)*self.n_population

        # Population
        self.population = None

        # leaders archive
        self.leaders = None

        # Sampling
        self.sampling = sampling

        # Survival (used for evaluating global best)
        self.survival = survival

        # Mutation
        self.mutation = mutation
        # Mutation
        # self.mutation1 = UniformMutation(eta=20, prob=None)
        # # self.mutation1 = PolynomialMutation(eta=20, prob=0.01)      # UniformMutation(eta=20, prob=None)
        # self.mutation2 = NoMutation()
        # self.mutation3 = NonUniformMutation(eta=20, prob=None)

        # Selection
        self.selection = selection

        # Adaptive flag
        self.adaptive = adaptive

        # Mutation flag
        self.apply_mutation = True

        # Flag for second order PSO mechanism
        self.use_second_order = True # see Ma2009aa.pdf for second order description

        # Swarm parameters
        self.w = w      # The inertia during velocity computation (if adaptive=True, this is the initial value only)
        self.c_1 = c_1  # The cognitive impact (personal best) used during velocity computation (if adaptive=True,
        # this is the initial value only)
        self.c_2 = c_2  # The social impact (global best) used during velocity computation (if adaptive=True,
        # this is the initial value only)

        # DE parameters for the leaders
        # TODO check the f_set and cr_set values from C2ODE and compare with Wickramasinghe2008
        if f_set is None:
            # self.f_set = 2*[0.2, 0.3, 0.45]
            self.f_set = [0.3, 0.6, 0.9]
        else:
            self.f_set = f_set
        if cr_set is None:
            # self.cr_set = 2*[0.05, 0.1, 0.25]
            self.cr_set = [0.1, 0.2, 0.5]
        else:
            self.cr_set = cr_set

        # Archive for second order system
        if self.use_second_order:
            self.archive = None

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

        # Create the archive for the leaders (personal best)
        self.leaders = copy.deepcopy(self.population)

        # Create the archive for second order
        if self.use_second_order:
            self.archive = copy.deepcopy(self.population)

    def _next(self):

        self._step()

        if self.adaptive:
            if not self.use_second_order:
                self._adapt()
            else:
                self._adapt_second_order()

    def _step(self):

        # create archive first
        if self.use_second_order:
            if np.mod(self.n_gen, 2) == 0:
                self.archive = copy.deepcopy(self.population)

        # Compute swarm local & global optima (and extract positions)
        # Personal best is calculated based on non-dominated sorting
        self.personal_best = self.population.compute_local_optima(self.problem)

        # Global best is calculated based on the selected survival method
        self.opt, self.global_best = self.population.compute_global_optimum(self.problem, self.n_population,
                                                                                 survival=self.survival)

        # Assign new positions & velocities
        # leader_position = self.population.extract_var()
        leader_position = self.personal_best

        # select leaders via a DE mechanism
        for i in range(self.n_population):
            # select 3 random indices
            index1, index2, index3 = self._select_random_indices(self.n_population,i)
            f, cr = self._generate_f_and_cr()
            for var in range(self.problem.n_var):
                rand = np.random.random(1)
                j_rand = np.random.randint(0,self.problem.n_var)
                if rand < cr or var == j_rand:
                    leader_position[i, var] = leader_position[index1, var] + \
                                             f*(leader_position[index2, var] - leader_position[index3, var])

        # Extract current position and velocity of each individual
        position = self.population.extract_var()
        velocity = self.population.extract_velocity()

        # Calculate inertia of each individual
        inertia = velocity

        if self.use_second_order:
            r_1 = np.random.random((self.n_population, self.problem.n_var))
            r_2 = np.random.random((self.n_population, self.problem.n_var))

            r_1 *= self.c_1
            r_2 *= self.c_2
            # chi1 = (2. * np.sqrt(r_1) - 1) / r_1 * self.max_gen / (max(self.n_gen, 10))
            # chi2 = (2. * np.sqrt(r_2) - 1) / r_2 * self.max_gen / (max(self.n_gen, 10))
            chi1 = (2. * np.sqrt(r_1) - 1) / r_1
            chi2 = (2. * np.sqrt(r_2) - 1) / r_2
            archive_position = self.archive.extract_var()

            cognitive = r_1 * (self.personal_best - (1+chi1) * position) + chi1 * archive_position
            social = r_2 * (leader_position - (1+chi2) * position) + chi2 * archive_position

            _velocity = self.w * (inertia + cognitive + social)
        else:

            # Calculate random values for directional computations
            r_1 = np.random.random(self.n_population)
            r_2 = np.random.random(self.n_population)

            r_1 = np.tile(r_1, (self.problem.n_var, 1)).T
            r_2 = np.tile(r_2, (self.problem.n_var, 1)).T

            cognitive = self.c_1*r_1*(self.personal_best - position)
            social = self.c_2*r_2*(leader_position - position)

            # Calculate new velocity
            _velocity = self.w*(inertia + cognitive + social)

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
        _position = BasicBoundsRepair().do(self.problem, _position)

        # Create offspring
        offspring = copy.deepcopy(self.population)

        # Assign new positions & velocities
        offspring.assign_var(self.problem, _position)
        offspring.assign_velocity(_velocity)

        # Evaluate the population at new positions
        offspring = self.evaluator.do(self.problem.obj_func, self.problem, offspring)


        # self.population = copy.deepcopy(full_population)


        # Mutate the population
        if self.apply_mutation:
            # # need 3 mutations
            # _population1 = copy.deepcopy(offspring)
            # _population2 = copy.deepcopy(offspring)
            # _population3 = copy.deepcopy(offspring)
            #
            # _population1 = self.mutation1.do(self.problem, _population1)
            # _population2 = self.mutation2.do(self.problem, _population2)
            # _population3 = self.mutation3.do(self.problem, _population3, current_iteration=self.n_gen,
            #                             max_iterations=self.max_gen)
            #
            # # Create Mask for mutations
            # random_numbers_for_mask = np.random.random(self.n_population)
            # do_mutation1 = np.zeros(len(self.population), dtype=bool)
            # do_mutation2 = np.zeros(len(self.population), dtype=bool)
            # do_mutation3 = np.zeros(len(self.population), dtype=bool)
            #
            # for i in range(len(self.population)):
            #     if random_numbers_for_mask[i] <= 1/3:
            #         do_mutation1[i] = True
            #     elif random_numbers_for_mask[i] <= 2/3:
            #         do_mutation2[i] = True
            #     else:
            #         do_mutation3[i] = True
            #
            # offspring[do_mutation1] = _population1[do_mutation1]
            # offspring[do_mutation2] = _population1[do_mutation2]
            # offspring[do_mutation3] = _population1[do_mutation3]
            offspring = self.mutation.do(self.problem, offspring, current_iteration=self.n_gen,max_iterations=self.max_gen)

        # Evaluate the population at new positions
        offspring = self.evaluator.do(self.problem.obj_func, self.problem, offspring)

        # Merge the offspring and main population
        full_population = Swarm.merge(self.population, offspring)

        # Apply survival
        full_population = self.survival.do(self.problem, full_population, self.n_population,
                                           gen=self.n_gen, max_gen=self.max_gen)

        _position = full_population.extract_var()
        _velocity = full_population.extract_velocity()
        self.population.assign_var(self.problem,_position)
        self.population.assign_velocity(_velocity)


        # Assign rank and crowding to population
        self.population.assign_rank_and_crowding()

        # Add a survival method
        # TODO add survival method? see if another survival method will work better.
        #  Need to generate offspring for that first! otherwise can not be recombined

    def _adapt(self):

        # Here using the platypus technique of randomising each iteration
        # self.w = np.random.uniform(0.1, 0.5)
        # self.c_1 = np.random.uniform(1.5, 2.5)
        # self.c_2 = np.random.uniform(1.5, 2.5)
        self.c_1 = np.random.uniform(0, 2.1)
        self.c_2 = np.random.uniform(0, 2.1)

    def _adapt_second_order(self):

        # Here using the platypus technique of randomising each iteration
        self.w = np.random.uniform(0.1, 0.5)
        self.c_1 = np.random.uniform(1.5, 2.5)
        self.c_2 = np.random.uniform(1.5, 2.5)

    def _generate_f_and_cr(self):
        rand = np.random.random(1)
        if rand <= 1/3:
            f_value = self.f_set[0]
        elif rand <= 2/3:
            f_value = self.f_set[1]
        else:
            f_value = self.f_set[2]

        rand = np.random.random(1)
        if rand <= 1/3:
            cr_value = self.cr_set[0]
        elif rand <= 2/3:
            cr_value = self.cr_set[1]
        else:
            cr_value = self.cr_set[2]
        return f_value, cr_value

    def _select_random_indices(self, population_size, current_index):
        index_list = list(range(population_size))
        index_list.pop(current_index)
        selected_indices = random.sample(index_list, 3)
        return selected_indices[0], selected_indices[1], selected_indices[2]

