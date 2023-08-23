import numpy as np

from optimisation.model.algorithm import Algorithm

from optimisation.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from optimisation.operators.survival.rank_and_crowding_survival import RankAndCrowdingSurvival
from optimisation.operators.survival.rank_survival import RankSurvival
from optimisation.operators.mutation.polynomial_mutation import PolynomialMutation
from optimisation.model.swarm import Swarm
from optimisation.model.repair import InversePenaltyBoundsRepair, BasicBoundsRepair, BounceBackBoundsRepair
from optimisation.operators.replacement.improvement_replacement import ImprovementReplacement


class PSO(Algorithm):

    def __init__(self,
                 n_population=100,
                 sampling=LatinHypercubeSampling(),
                 survival=RankAndCrowdingSurvival(),    # RankSurvival(),    #
                 mutation=PolynomialMutation(eta=20, prob=None),    # None,     #
                 w=0.9,     # 0.9,
                 c_1=0.5,
                 c_2=0.3,
                 adaptive=True,     # False,    #
                 initial_velocity='random',     # None,     #
                 max_velocity_rate=0.2,         # 0.5,         #
                 **kwargs):

        super().__init__(**kwargs)

        # Population parameters
        self.n_population = n_population

        # Generation parameters
        self.max_f_eval = (self.max_gen+1)*self.n_population
        if mutation:
            self.max_f_eval += self.max_f_eval

        # Population
        self.population = None

        # Sampling
        self.sampling = sampling

        # Survival (used for evaluating global best)
        self.survival = survival

        # Mutation
        self.mutation = mutation

        # Adaptive flag
        self.adaptive = adaptive

        # With this set to true, each individual is only replaced by its new position if the solution improves
        self.replace_if_improved = True

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

        # Local and global optimal positions
        self.local_optimal_x = None
        self.global_optimum_x = None

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

    def _next(self):

        self._step()

        if self.adaptive:
            self._adapt()

    def _step(self):

        # Compute swarm local & global optima (and extract positions)
        self.local_optimal_x = self.population.compute_local_optima(self.problem)
        self.opt, self.global_optimum_x = self.population.compute_global_optimum(self.problem, self.n_population,
                                                                                 survival=self.survival)

        # Extract current position and velocity of each individual
        position = self.population.extract_var()
        velocity = self.population.extract_velocity()

        # Calculate inertia of each individual
        inertia = self.w*velocity

        # Calculate random values for directional computations
        r_1 = np.random.random((self.n_population, self.problem.n_var))
        r_2 = np.random.random((self.n_population, self.problem.n_var))

        # Calculate cognitive and social influence
        cognitive = self.c_1*r_1*(self.local_optimal_x - position)
        social = self.c_2*r_2*(self.global_optimum_x - position)

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
        _velocity[np.logical_or(upper_mask, lower_mask)] *= -0.5

        # Repair positions if they exist outside variable bounds
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
            self.population = self.evaluator.do(self.problem.obj_func, self.problem, self.population)

        # Mutate the population
        if self.mutation:
            self.population = self.mutation.do(self.problem, self.population)

            # Evaluate the population at new positions
            self.population = self.evaluator.do(self.problem.obj_func, self.problem, self.population)

        # Assign rank and crowding to population
        self.population.assign_rank_and_crowding()

    def _adapt(self):

        # Here using the platypus technique of randomising each iteration
        self.w = np.random.uniform(0.1, 0.5)
        self.c_1 = np.random.uniform(1.5, 2.0)
        self.c_2 = np.random.uniform(1.5, 2.0)

