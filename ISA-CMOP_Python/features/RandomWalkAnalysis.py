import numpy as np
from features.randomwalkfeatures import *
from scipy.stats import yeojohnson
from features.Analysis import Analysis, MultipleAnalysis


class RandomWalkAnalysis(Analysis):
    """
    Calculate all features generated from random walk samples.

    Populations is a list of populations that represents a walk, each entry is a solution and its neighbours.
    """

    def __init__(self, pop_walk, pop_neighbours_list):
        """
        Populations must already be evaluated.
        """
        super().__init__(pop_walk)
        self.pop_neighbours_list = pop_neighbours_list

    def eval_features(self):
        """
        Evaluate features along the random walk and save to class.
        """

        # Preprocess nans and infinities, compute solver crash ratio and update attributes.
        pop_new, pop_neighbours_new, pop_neighbours_checked = preprocess_nans(
            self.pop, self.pop_neighbours_list
        )
        self.scr = compute_solver_crash_ratio(self.pop, pop_new)
        self.ncr_avg, self.ncr_r1 = compute_neighbourhood_crash_ratio(
            pop_neighbours_new, pop_neighbours_checked
        )

        # Update populations
        self.pop = pop_new
        self.pop_neighbours_list = pop_neighbours_checked

        # Evaluate neighbourhood distance features. Note that these assignments will also append names to feature names list
        (
            self.dist_x_avg,
            self.dist_x_r1,
            self.dist_f_avg,
            self.dist_f_r1,
            self.dist_c_avg,
            self.dist_c_r1,
            self.dist_f_c_avg,
            self.dist_f_c_r1,
            self.dist_f_dist_x_avg,
            self.dist_f_dist_x_r1,
            self.dist_c_dist_x_avg,
            self.dist_c_dist_x_r1,
            self.dist_f_c_dist_x_avg,
            self.dist_f_c_dist_x_r1,
        ) = compute_neighbourhood_distance_features(self.pop, self.pop_neighbours_list)

        # Evaluate neighbourhood HV features.
        (
            self.hv_single_soln_avg,
            self.hv_single_soln_r1,
            self.nhv_avg,
            self.nhv_r1,
            self.hvd_avg,
            self.hvd_r1,
            self.bhv_avg,
            self.bhv_r1,
        ) = compute_neighbourhood_hv_features(self.pop, self.pop_neighbours_list)

        # Evaluate neighbourhood violation features.
        (
            self.nrfbx,
            self.nncv_avg,
            self.nncv_r1,
            self.ncv_avg,
            self.ncv_r1,
            self.bncv_avg,
            self.bncv_r1,
        ) = compute_neighbourhood_violation_features(self.pop, self.pop_neighbours_list)

        # Evaluate neighbourhood domination features.
        (
            self.sup_avg,
            self.sup_r1,
            self.inf_avg,
            self.inf_r1,
            self.inc_avg,
            self.inc_r1,
            self.lnd_avg,
            self.lnd_r1,
            self.nfronts_avg,
            self.nfronts_r1,
        ) = compute_neighbourhood_dominance_features(self.pop, self.pop_neighbours_list)


class MultipleRandomWalkAnalysis(MultipleAnalysis):
    """
    Aggregate RW features across populations/walks.
    """

    def __init__(self, pops_walks, pops_neighbours_list):
        self.pops = pops_walks
        self.analyses = []

        if len(self.pops) != 0:
            for ctr, pop in enumerate(pops_walks):
                self.analyses.append(RandomWalkAnalysis(pop, pops_neighbours_list[ctr]))

    @staticmethod
    def concatenate_multiple_analyses(multiple_analyses):
        """
        Concatenate feature arrays from multiple MultipleRandomWalkAnalysis objects into one.
        """

        # Create a new MultipleAnalysis object with the combined populations
        combined_analysis = MultipleRandomWalkAnalysis([], [])

        # Extract the feature names too.
        combined_analysis.feature_names = multiple_analyses[0].feature_names

        # Iterate through feature names
        for feature_name in combined_analysis.feature_names:
            feature_arrays = [
                getattr(ma, f"{feature_name}_array") for ma in multiple_analyses
            ]
            combined_array = np.concatenate(feature_arrays)

            # Save as attribute array in the combined_analysis object
            setattr(combined_analysis, f"{feature_name}_array", combined_array)

        return combined_analysis
