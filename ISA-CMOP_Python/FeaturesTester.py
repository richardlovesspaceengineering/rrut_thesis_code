from ProblemEvaluator import ProblemEvaluator
import numpy as np
from pymoo.problems import get_problem
import matplotlib.pyplot as plt
from optimisation.model.population import Population

import copy

from optimisation.operators.sampling.AdaptiveWalk import AdaptiveWalk


from features.globalfeatures import *

from features.randomwalkfeatures import *


def plot_adaptive_walk(problem):
    # Create a 2x2 grid of subplots
    fig, ax_obj = plt.subplots(5, 5, figsize=(10, 10))
    # fig, ax_var = plt.subplots(5, 5, figsize=(10, 10))
    for i in range(5):
        for j in range(5):
            # Define bounds for each subplot
            aw = AdaptiveWalk(problem, 1000, 0.02, 21)

            # Starting zone binary array.
            starting_zone = np.array([1 for _ in range(10)])

            # Simulate adaptive walk.
            walk = aw.do_adaptive_walk(
                starting_zone=starting_zone, constrained_ranks=True, seed=1
            )

            # Evaluate population and plot objectives.
            pop = Population(problem, n_individuals=walk.shape[0])
            pop.evaluate(walk, eval_fronts=False)
            obj = pop.extract_obj()
            ax_obj[i, j].plot(obj[:, 0], obj[:, 1], "b-")
            ax_obj[i, j].plot(obj[0, 0], obj[0, 1], "ko")
            ax_obj[i, j].plot(obj[-1, 0], obj[-1, 1], "rx")

            # Plot decision variables.
            # var = pop.extract_var()
            # ax_var[i, j].plot(var[:, 0], var[:, 1], "g-")
            # ax_var[i, j].plot(var[0, 0], var[0, 1], "ko")
            # ax_var[i, j].plot(var[-1, 0], var[-1, 1], "rx")

            # Plot CV

    # Adjust subplot spacing
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_transformed_objective_space(
    pop_walk,
    pop_neighbours,
    pop_walk_normalised,
    pop_neighbours_normalised,
    PF_normalised,
    scale_offset,
):
    """
    Side by side plots of objective space showing non-normalised and normalised values respectively.
    """

    # Compare with two side-by-side plots.
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))  # Adjust figsize as needed

    # Non-normalised
    # Walk
    walk_obj = pop_walk.extract_obj()
    ax[0].plot(walk_obj[:, 0], walk_obj[:, 1], "b-")

    # Neighbours
    for neighbour in pop_neighbours:
        neig_obj = neighbour.extract_obj()
        ax[0].plot(neig_obj[:, 0], neig_obj[:, 1], "go", markersize=2)

    # PF
    pf = pop_walk.extract_pf()
    ax[0].plot(pf[:, 0], pf[:, 1], "r-")

    ax[0].set_aspect("equal")

    # Set the title for ax[0]
    ax[0].set_title("Objectives")

    # Normalised
    # Walk
    walk_obj = pop_walk_normalised.extract_obj()
    ax[1].plot(walk_obj[:, 0], walk_obj[:, 1], "b-")

    # Neighbours
    for neighbour in pop_neighbours_normalised:
        neig_obj = neighbour.extract_obj()
        ax[1].plot(neig_obj[:, 0], neig_obj[:, 1], "go", markersize=2)

    # PF
    pf = PF_normalised
    ax[1].plot(pf[:, 0], pf[:, 1], "r-")

    # Set equal aspect ratio and limits
    ax[1].set_aspect("equal")
    ax[1].set_xlim(0, 1.1)
    ax[1].set_ylim(0, 1.1)
    ax[1].axhline(1, color="gray", linestyle="--")
    ax[1].axvline(1, color="gray", linestyle="--")

    # Set the title for ax[1] with the scale offset
    ax[1].set_title(f"Normalised using Scale Factor {scale_offset}")

    plt.show()


if __name__ == "__main__":
    # Flags for which feature set we want to test.
    sample_global = False
    sample_rw = True
    sample_aw = False

    # Example problem.
    num_samples = 2
    n_var = 10
    problem_name = "MW11"
    problem = get_problem(problem_name, n_var)
    instance_string = f"{problem_name}_d{n_var}"

    # Generate evaluator which can generate RW samples and LHS samples.
    evaluator = ProblemEvaluator(problem, instance_string, "debug", None)
    pre_sampler = evaluator.create_pre_sampler(num_samples)

    if sample_global:
        ### GLOBAL FEATURES.

        # Generate distributed samples and evaluate populations on these samples.
        sample = pre_sampler.read_global_sample(1)
        pop_global = evaluator.generate_global_population(
            problem, sample, eval_fronts=False
        )

        test_ic_features = True
        if test_ic_features:
            (H_max, eps_s, m0, eps05) = compute_ic_features(pop_global)

    if sample_rw:
        test_obj_normalisation = False
        test_neighbourhood_dist_features = False
        test_neighbourhood_hv_features = False
        test_neighbourhood_violation_features = False
        test_scr = False

        ### RW FEATURES
        walk, neighbours = pre_sampler.read_walk_neighbours(1, 3)
        pop_walk, pop_neighbours_list = evaluator.generate_rw_neighbours_populations(
            problem, walk, neighbours, eval_fronts=False
        )
        # if test_obj_normalisation:
        #     # Compute normalised objectives.
        #     scale_offset = 1
        #     (
        #         pop_walk_normalised,
        #         pop_neighbours_normalised,
        #         PF_normalised,
        #     ) = normalise_objective_space(
        #         pop_walk,
        #         pop_neighbours,
        #         pop_walk.extract_pf(),
        #         scale_offset=scale_offset,
        #         region_of_interest=False,
        #     )

        #     plot_transformed_objective_space(
        #         pop_walk,
        #         pop_neighbours,
        #         pop_walk_normalised,
        #         pop_neighbours_normalised,
        #         PF_normalised,
        #         scale_offset,
        # )

        # Neighbourhood distance features
        if test_neighbourhood_dist_features:
            (
                dist_x_avg,
                dist_x_r1,
                dist_f_avg,
                dist_f_r1,
                dist_c_avg,
                dist_c_r1,
                dist_f_c_avg,
                dist_f_c_r1,
                dist_f_dist_x_avg,
                dist_f_dist_x_r1,
                dist_c_dist_x_avg,
                dist_c_dist_x_r1,
                dist_f_c_dist_x_avg,
                dist_f_c_dist_x_r1,
            ) = compute_neighbourhood_distance_features(pop_walk, pop_neighbours)

        # Neighbourhood HV features.
        if test_neighbourhood_hv_features:
            (
                hv_ss_avg,
                hv_ss_r1,
                nhv_avg,
                nhv_r1,
                hvd_avg,
                hvd_r1,
                bhv_avg,
                bhv_r1,
            ) = compute_neighbourhood_hv_features(pop_walk, pop_neighbours)

        # Neighbourhood violation features.
        if test_neighbourhood_violation_features:
            (
                nrfbx,
                nncv_avg,
                nncv_r1,
                ncv_avg,
                ncv_r1,
                bncv_avg,
                bncv_r1,
            ) = compute_neighbourhood_violation_features(pop_walk, pop_neighbours)

        # Solver crash ratio.
        if test_scr:
            walk_with_issue = copy.deepcopy(walks_neighbours_list[0][0])

            # Use different step to the walk issue.
            neighbourhood_with_issue = copy.deepcopy(walks_neighbours_list[0][1])

            # Put an infeasible value in the walk (step 1) and in the neighbours (step 2) for MW11.
            walk_with_issue[0, :] = 5 * np.ones((1, walk_with_issue.shape[1]))
            neighbourhood_with_issue[1][0, :] = 5 * np.ones(
                (1, walk_with_issue.shape[1])
            )

            walks_neighbours_list_new = [(walk_with_issue, neighbourhood_with_issue)]

            pop_walk_neighbourhood = evaluator.evaluate_populations_for_rw_features(
                problem, walks_neighbours_list_new
            )[0]

            pop_walk = pop_walk_neighbourhood[0]
            pop_neighbours = pop_walk_neighbourhood[1]

            # Preprocess nans and infinities, compute solver crash ratio and update attributes.
            (
                pop_new,
                pop_neighbours_new,
                pop_neighbours_checked,
            ) = preprocess_nans_on_walks(pop_walk, pop_neighbours)
            scr = compute_solver_crash_ratio(pop_walk, pop_new)
            ncr = compute_neighbourhood_crash_ratio(
                pop_neighbours_new, pop_neighbours_checked
            )

        test_ic_features = True
        if test_ic_features:
            (H_max, eps_s, m0, eps05) = compute_ic_features(pop_walk, sample_type="rw")

    if sample_aw:
        plot_aw = True

        if plot_aw:
            plot_adaptive_walk(problem)
