import numpy as np
from optimisation.operators.sampling.RandomWalk import RandomWalk
import matplotlib.pyplot as plt
from optimisation.model.population import Population
from features.feature_helpers import generate_bounds_from_problem
from pymoo.problems import get_problem
import pygmo
from features.Analysis import Analysis, MultipleAnalysis


class AdaptiveWalk(RandomWalk):
    def __init__(self, dim, max_steps, step_size_pct, neighbourhood_size, problem):
        self.dim = dim

        # Use actual problem bounds as opposed to unit hypercube.
        self.bounds = generate_bounds_from_problem(problem)
        self.num_steps = max_steps
        self.step_size_pct = step_size_pct
        self.neighbourhood_size = neighbourhood_size

        self.problem_instance = problem

        self.initialise_step_sizes()

    def do_adaptive_phc_walk_for_starting_point(
        self, starting_point, constrained_ranks, seed=None
    ):
        """
        Simulate a Pareto Hill Climber - walk will move in direction that ensures improvement. If no further improvement is possible, the walk is concluded.

        The walk starts in a user-specified starting zone.
        """

        # In this case, num_steps defines the maximum number of adaptive walk steps
        walk = np.empty((self.num_steps, self.dim))  # array to store the walk
        walk[:] = np.nan
        np.random.seed(seed)

        # Start adaptive walk.
        walk[0, :] = starting_point

        # Continue adaptive walk.
        improving_solutions_exist = True
        step_counter = 0
        while improving_solutions_exist and step_counter <= self.num_steps:
            # Generate neighbours for this step.
            potential_next_steps = self.generate_neighbours_for_step(
                walk[step_counter, :], self.neighbourhood_size
            )

            # Put all together in one large matrix.
            step_and_neighbours = np.vstack(
                (walk[step_counter, :], potential_next_steps)
            )

            # Evaluate the problem at this step.
            pop_first_step = Population(
                self.problem_instance, n_individuals=step_and_neighbours.shape[0]
            )
            pop_first_step.evaluate(step_and_neighbours, eval_fronts=True)

            # The first solution which has a rank lower than the current solution (located at top of matrix) is the next step of our walk.
            if not constrained_ranks:
                pop_first_step.eval_unconstrained_rank()
                ranks = pop_first_step.extract_uncons_rank()
            else:
                ranks = pop_first_step.extract_rank()

            # Take first dominating solution.
            try:
                next_step = step_and_neighbours[ranks < ranks[0]][0, :]
                walk[step_counter + 1, :] = next_step
                step_counter += 1
            except:
                improving_solutions_exist = False

        # Trim any remaining rows in the walk array.
        return walk[~np.isnan(walk).any(axis=1)]
