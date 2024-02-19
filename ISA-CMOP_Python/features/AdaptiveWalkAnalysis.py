import numpy as np
from features.RandomWalkAnalysis import RandomWalkAnalysis


class AdaptiveWalkAnalysis(RandomWalkAnalysis):
    def eval_features(self, pop_walk, pop_neighbours_list):
        # Preprocess nans and infinities, compute solver crash ratio and update attributes.
        pop_new, pop_neighbours_new, pop_neighbours_checked = (
            RandomWalkAnalysis.preprocess_nans_on_walks(pop_walk, pop_neighbours_list)
        )

        # Update populations
        pop_walk = pop_new
        pop_neighbours_list = pop_neighbours_checked

        # Evaluate neighbourhood domination features. Note that no normalisation is needed for these.
        (
            self.features["sup_avg"],
            _,
            self.features["inf_avg"],
            _,
            self.features["inc_avg"],
            _,
            self.features["lnd_avg"],
            _,
            _,
            _,
        ) = super().compute_neighbourhood_dominance_features(
            pop_walk, pop_neighbours_list
        )

        # Evaluate unconstrained neighbourhood HV features
        (
            self.features["uhv_ss_avg"],
            _,
            self.features["nuhv_avg"],
            _,
            self.features["uhvd_avg"],
            _,
            _,
            _,
        ) = super().compute_neighbourhood_hv_features(
            pop_walk,
            pop_neighbours_list,
            norm_method="95th",
        )

        # Evaluate constrained neighbourhood HV features

        (
            pop_walk_feas,
            _,
            pop_neighbours_feas,
        ) = super().extract_feasible_steps_neighbours(pop_walk, pop_neighbours_list)
        (
            self.features["hv_ss_avg"],
            _,
            self.features["nhv_avg"],
            _,
            self.features["hvd_avg"],
            _,
            self.features["bhv_avg"],
            _,
        ) = compute_neighbourhood_hv_features(
            pop_walk_feas,
            pop_neighbours_feas,
            self.normalisation_values,
            norm_method="95th",
        )

        # Compute average length of adaptive walks.
        self.features["length_avg"] = len(
            pop_walk
        )  # TODO: decide if we need to normalise this.
