import numpy as np
from features.RandomWalkAnalysis import RandomWalkAnalysis


class AdaptiveWalkAnalysis(RandomWalkAnalysis):
    def eval_features(self):
        # Preprocess nans and infinities, compute solver crash ratio and update attributes.
        pop_new, pop_neighbours_new, pop_neighbours_checked = (
            self.preprocess_nans_on_walks()
        )

        # Update populations
        self.pop_walk = pop_new
        self.pop_neighbours_list = pop_neighbours_checked

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
        ) = super().compute_neighbourhood_dominance_features()

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
        ) = super().compute_uncons_neighbourhood_hv_features(norm_method="95th")

        # Evaluate constrained neighbourhood HV features
        (
            self.features["hv_ss_avg"],
            _,
            self.features["nhv_avg"],
            _,
            self.features["hvd_avg"],
            _,
            self.features["bhv_avg"],
            _,
        ) = self.compute_cons_neighbourhood_hv_features(norm_method="95th")

        # Compute average length of adaptive walks.
        self.features["length_avg"] = len(
            self.pop_walk
        )  # TODO: decide if we need to normalise this.
