import numpy as np
from features.RandomWalkAnalysis import RandomWalkAnalysis, MultipleRandomWalkAnalysis
from features.randomwalkfeatures import (
    compute_neighbourhood_dominance_features,
    compute_neighbourhood_hv_features,
)


class AdaptiveWalkAnalysis(RandomWalkAnalysis):
    def eval_features(self):
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
        ) = compute_neighbourhood_dominance_features(self.pop, self.pop_neighbours_list)

        # Evaluate neighbourhood HV features
        (
            self.features["hv_single_soln_avg"],
            _,
            self.features["nhv_avg"],
            _,
            self.features["hvd_avg"],
            _,
            self.features["bhv_avg"],
            _,
        ) = compute_neighbourhood_hv_features(
            self.pop,
            self.pop_neighbours_list,
            self.normalisation_values,
            norm_method="95th",
        )

        # Compute average length of adaptive walks.
        self.features["length_avg"] = len(
            self.pop
        )  # TODO: decide if we need to normalise this.


class MultipleAdaptiveWalkAnalysis(MultipleRandomWalkAnalysis):
    def __init__(self, pops_walks, pops_neighbours_list, normalisation_values):
        super().__init__(
            pops_walks,
            pops_neighbours_list,
            normalisation_values,
            single_analysis_class=AdaptiveWalkAnalysis,
        )
