import numpy as np
from features.randomwalkfeatures import *
from scipy.stats import yeojohnson
from features.Analysis import Analysis, MultipleAnalysis


class RandomWalkAnalysis(Analysis):
    """
    Calculate all features generated from random walk samples.

    Populations is a list of populations that represents a walk, each entry is a solution and its neighbours.
    """

    def eval_features(self, pop_walk, pop_neighbours_list):
        """
        Evaluate features along the random walk and save to class.
        """

        # Preprocess nans and infinities, compute solver crash ratio and update attributes.
        pop_new, pop_neighbours_new, pop_neighbours_checked = preprocess_nans_on_walks(
            pop_walk, pop_neighbours_list
        )
        self.features["scr"] = compute_solver_crash_ratio(pop_walk, pop_new)
        (
            self.features["ncr_avg"],
            self.features["ncr_r1"],
        ) = compute_neighbourhood_crash_ratio(
            pop_neighbours_new, pop_neighbours_checked
        )

        # Update populations
        pop_walk = pop_new
        pop_neighbours_list = pop_neighbours_checked

        # Evaluate neighbourhood distance features
        (
            self.features["dist_x_avg"],
            self.features["dist_x_r1"],
            self.features["dist_f_avg"],
            self.features["dist_f_r1"],
            self.features["dist_c_avg"],
            self.features["dist_c_r1"],
            self.features["dist_f_c_avg"],
            self.features["dist_f_c_r1"],
            self.features["dist_f_dist_x_avg"],
            self.features["dist_f_dist_x_r1"],
            self.features["dist_c_dist_x_avg"],
            self.features["dist_c_dist_x_r1"],
            self.features["dist_f_c_dist_x_avg"],
            self.features["dist_f_c_dist_x_r1"],
        ) = compute_neighbourhood_distance_features(
            pop_walk,
            pop_neighbours_list,
            self.normalisation_values,
            norm_method="95th",
        )

        # Evaluate unconstrained neighbourhood HV features
        (
            self.features["uhv_ss_avg"],
            self.features["uhv_ss_r1"],
            self.features["nuhv_avg"],
            self.features["nuhv_r1"],
            self.features["uhvd_avg"],
            self.features["uhvd_r1"],
            _,
            _,
        ) = compute_neighbourhood_hv_features(
            pop_walk,
            pop_neighbours_list,
            self.normalisation_values,
            norm_method="95th",
        )

        # Evaluate constrained neighbourhood HV features.
        (
            pop_walk_feas,
            _,
            pop_neighbours_feas,
        ) = extract_feasible_steps_neighbours(pop_walk, pop_neighbours_list)
        (
            self.features["hv_ss_avg"],
            self.features["hv_ss_r1"],
            self.features["nhv_avg"],
            self.features["nhv_r1"],
            self.features["hvd_avg"],
            self.features["hvd_r1"],
            self.features["bhv_avg"],
            self.features["bhv_r1"],
        ) = compute_neighbourhood_hv_features(
            pop_walk_feas,
            pop_neighbours_feas,
            self.normalisation_values,
            norm_method="95th",
        )

        # Evaluate neighbourhood violation features
        (
            self.features["nrfbx"],
            self.features["nncv_avg"],
            self.features["nncv_r1"],
            self.features["ncv_avg"],
            self.features["ncv_r1"],
            self.features["bncv_avg"],
            self.features["bncv_r1"],
        ) = compute_neighbourhood_violation_features(
            pop_walk,
            pop_neighbours_list,
            self.normalisation_values,
            norm_method="95th",
        )

        # Evaluate neighbourhood domination features. Note that no normalisation is needed for these.
        (
            self.features["sup_avg"],
            self.features["sup_r1"],
            self.features["inf_avg"],
            self.features["inf_r1"],
            self.features["inc_avg"],
            self.features["inc_r1"],
            self.features["lnd_avg"],
            self.features["lnd_r1"],
            self.features["nfronts_avg"],
            self.features["nfronts_r1"],
        ) = compute_neighbourhood_dominance_features(pop_walk, pop_neighbours_list)
