import os
import numpy as np

from optimisation.model.sampling import Sampling
from optimisation.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling

from lib import config


class CrossoverPrescreenedSampling(Sampling):
    def __init__(self, problem, n_sample_multiplier=3, n_resampling_attempts=1):
        super().__init__()

        self.problem = problem
        self.n_sample_multiplier = n_sample_multiplier
        self.n_resampling_attempts = n_resampling_attempts
        self.lhs_sampling = LatinHypercubeSampling(criterion='maximin', iterations=1000)

    def _do(self, dim, n_samples, seed=None):
        n_feasible = 0
        x_lhs = None
        for i in range(self.n_resampling_attempts):
            # Generate LHS designs  TODO: Additional AoA var for max_lift_obj (x_lower[:-1], x_upper[:-1])
            self.lhs_sampling.do(self.n_sample_multiplier * n_samples, self.problem.x_lower, self.problem.x_upper, seed)
            x_lhs = self.lhs_sampling.x

            # Evaluate crossover constraint on LHS designs
            feasible = np.zeros(len(x_lhs), dtype=bool)
            for idx in range(len(x_lhs)):
                config.design.shape_variables = x_lhs[idx]
                temp = config.design.airfoil
                temp.generate_section(config.design.shape_variables, config.design.n_pts, delta_z_te=0.0025)
                temp.calc_thickness_and_camber()
                temp.calculate_curvature()

                # Crossover check
                crossover_viol = np.min(temp.thickness) > 0.0

                # Max thickness check
                max_thickness_viol = np.amax(temp.thickness) < config.design.max_thickness_margin * \
                                     config.design.max_thickness_constraint

                # Leading edge radius check
                le_edge_radius_viol = temp.leading_edge_radius > config.design.leading_edge_radius_constraint

                # Combined geometric constraints
                feasible[idx] = crossover_viol and max_thickness_viol and le_edge_radius_viol

            # Determine number of crossover-feasible lhs individuals
            n_feasible = np.count_nonzero(feasible)

            print('iter, n_feas: ', i, n_feasible)

            if n_feasible >= n_samples:
                break

        # Assign the required feasible n_samples
        # n_feasible = self.n_sample_multiplier * n_samples
        rand_selection = np.random.choice(np.arange(n_feasible), n_samples)
        x_selected = x_lhs[feasible][rand_selection]

        # Re-scale back to [0, 1] bounds for super class method .do()
        self.x = (x_selected - self.problem.x_lower) / (self.problem.x_upper - self.problem.x_lower)
