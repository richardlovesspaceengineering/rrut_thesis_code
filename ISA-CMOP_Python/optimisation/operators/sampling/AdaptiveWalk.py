import numpy as np
from optimisation.operators.sampling.RandomWalk import RandomWalk
import matplotlib.pyplot as plt
from optimisation.model.population import Population
from pymoo.problems import get_problem
from features.Analysis import Analysis, MultipleAnalysis


class AdaptiveWalk(RandomWalk):
    def __init__(self, dim, max_steps, step_size_pct, neighbourhood_size, problem):
        self.dim = dim

        # Use actual problem bounds as opposed to unit hypercube.
        self.bounds = Analysis.generate_bounds_from_problem(problem)
        self.num_steps = max_steps
        self.step_size_pct = step_size_pct
        self.neighbourhood_size = neighbourhood_size

        self.problem_instance = problem

        self.initialise_step_sizes()

    def do_adaptive_phc_walk_for_starting_point(
        self,
        starting_point,
        constrained_ranks,
        num_processes=1,
        return_pop=False,
        seed=None,
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

        # Create single-step population. Guaranteed that first step works.
        pop_walk = Population(self.problem_instance, n_individuals=1)
        pop_walk.evaluate(np.atleast_2d(walk[0, :]), eval_fronts=False, num_processes=1)

        # Continue adaptive walk.
        improving_solutions_exist = True
        step_counter = 0
        while improving_solutions_exist and step_counter <= self.num_steps:
            # Generate neighbours for this step.
            potential_next_steps = self.generate_neighbours_for_step(
                walk[step_counter, :], self.neighbourhood_size
            )

            # Evaluate potential next steps.
            pop_potential_next = Population(
                self.problem_instance, n_individuals=potential_next_steps.shape[0]
            )
            pop_potential_next.evaluate(
                potential_next_steps, eval_fronts=False, num_processes=num_processes
            )

            # Remove any NaNs before moving on.
            pop_potential_next_clean, _ = pop_potential_next.remove_nan_inf_rows()

            # If there are only NaNs around
            if len(pop_potential_next_clean) == 0:
                print("All nearby solutions are NaN. AW has ended.")
                improving_solutions_exist = False
                break

            # Concatenate populations and eval fronts.
            pop_first_step = Population.merge(
                pop_walk.get_single_pop(step_counter),
                pop_potential_next_clean,
            )
            pop_first_step.evaluate_fronts()

            # The first solution which has a rank lower than the current solution (located at top of matrix) is the next step of our walk.
            if not constrained_ranks:
                pop_first_step.eval_unconstrained_rank()
                ranks = pop_first_step.extract_uncons_rank()
            else:
                ranks = pop_first_step.extract_rank()

            # Take first dominating solution.
            try:
                mask = ranks < ranks[0]
                index_of_true = np.where(mask)[0][0]
                pop_best = pop_first_step.get_single_pop(index_of_true)
                pop_walk = Population.merge(pop_walk, pop_best)
                walk[step_counter + 1, :] = pop_best.extract_var()
                step_counter += 1
            except:
                improving_solutions_exist = False

        # Trim any remaining rows in the walk array.
        if return_pop:
            return pop_walk.extract_var(), pop_walk
        else:
            return pop_walk.extract_var()
