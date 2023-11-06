import numpy as np
from features.randomwalkfeatures import (
    compute_solver_crash_ratio,
    compute_neighbourhood_crash_ratio,
    preprocess_nans,
    compute_neighbourhood_distance_features,
    compute_neighbourhood_hv_features,
    compute_neighbourhood_violation_features,
    compute_neighbourhood_dominance_features
    )
from scipy.stats import yeojohnson
from features.Analysis import Analysis, MultipleAnalysis


class RandomWalkAnalysis(Analysis):
    """
    Calculate all features generated from random walk samples.

    Populations is a list of populations that represents a walk, each entry is a solution and its neighbours.
    """
    
    # Define feature names as a static attribute at the class level
    feature_names = [
        "scr",
        "ncr_avg",
        "ncr_r1",
        "dist_x_avg",
        "dist_x_r1",
        "dist_f_avg",
        "dist_f_r1",
        "dist_c_avg",
        "dist_c_r1",
        "dist_f_c_avg",
        "dist_f_c_r1",
        "dist_f_dist_x_avg",
        "dist_f_dist_x_r1",
        "dist_c_dist_x_avg",
        "dist_c_dist_x_r1",
        "dist_f_c_dist_x_avg",
        "dist_f_c_dist_x_r1",
        "hv_single_soln_avg",
        "hv_single_soln_r1",
        "nhv_avg",
        "nhv_r1",
        "hvd_avg",
        "hvd_r1",
        "bhv_avg",
        "bhv_r1",
        "nrfbx",
        "nncv_avg",
        "nncv_r1",
        "ncv_avg",
        "ncv_r1",
        "bncv_avg",
        "bncv_r1",
        "sup_avg",
        "sup_r1",
        "inf_avg",
        "inf_r1",
        "inc_avg",
        "inc_r1",
        "lnd_avg",
        "lnd_r1",
        "nfronts_avg",
        "nfronts_r1"
    ]



    def __init__(self, pop_walk, pop_neighbours_list):
        """
        Populations must already be evaluated.
        """
        super().__init__(pop_walk)
        self.pop_neighbours_list = pop_neighbours_list
        
        # Initialise values.
        self.initialize_features()

    def eval_features(self):
        """
        Evaluate features along the random walk and save to class.
        """
        
        # Preprocess nans and infinities, compute solver crash ratio and update attributes.
        pop_new, pop_neighbours_new, pop_neighbours_checked = preprocess_nans(self.pop, self.pop_neighbours_list)
        scr = compute_solver_crash_ratio(self.pop,pop_new)
        ncr_avg, ncr_r1 = compute_neighbourhood_crash_ratio(pop_neighbours_new, pop_neighbours_checked)
        self.pop = pop_new
        self.pop_neighbours_list = pop_neighbours_checked
        
        # Evaluate neighbourhood distance features.
        dist_x_avg, dist_x_r1, dist_f_avg, dist_f_r1, dist_c_avg, dist_c_r1, dist_f_c_avg, dist_f_c_r1, dist_f_dist_x_avg, dist_f_dist_x_r1, dist_c_dist_x_avg, dist_c_dist_x_r1, dist_f_c_dist_x_avg, dist_f_c_dist_x_r1 = compute_neighbourhood_distance_features(self.pop, self.pop_neighbours_list)
        
        # Evaluate neighbourhood HV features.
        hv_single_soln_avg, hv_single_soln_r1, nhv_avg, nhv_r1, hvd_avg, hvd_r1, bhv_avg, bhv_r1 = compute_neighbourhood_hv_features(self.pop, self.pop_neighbours_list)
        
        # Evaluate neighbourhood violation features.
        nrfbx, nncv_avg, nncv_r1, ncv_avg, ncv_r1, bncv_avg, bncv_r1 = compute_neighbourhood_violation_features(self.pop, self.pop_neighbours_list)
        
        # Evaluate neighbourhood domination features.
        sup_avg, sup_r1, inf_avg, inf_r1, inc_avg, inc_r1, lnd_avg, lnd_r1, nfronts_avg, nfronts_r1 = compute_neighbourhood_dominance_features(self.pop, self.pop_neighbours_list)
        
        # Create a dictionary to store feature names and values
        feature_dict = {
            "scr": scr,
            "ncr_avg": ncr_avg,
            "ncr_r1": ncr_r1,
            "dist_x_avg": dist_x_avg,
            "dist_x_r1": dist_x_r1,
            "dist_f_avg": dist_f_avg,
            "dist_f_r1": dist_f_r1,
            "dist_c_avg": dist_c_avg,
            "dist_c_r1": dist_c_r1,
            "dist_f_c_avg": dist_f_c_avg,
            "dist_f_c_r1": dist_f_c_r1,
            "dist_f_dist_x_avg": dist_f_dist_x_avg,
            "dist_f_dist_x_r1": dist_f_dist_x_r1,
            "dist_c_dist_x_avg": dist_c_dist_x_avg,
            "dist_c_dist_x_r1": dist_c_dist_x_r1,
            "dist_f_c_dist_x_avg": dist_f_c_dist_x_avg,
            "dist_f_c_dist_x_r1": dist_f_c_dist_x_r1,
            "hv_single_soln_avg": hv_single_soln_avg,
            "hv_single_soln_r1": hv_single_soln_r1,
            "nhv_avg": nhv_avg,
            "nhv_r1": nhv_r1,
            "hvd_avg": hvd_avg,
            "hvd_r1": hvd_r1,
            "bhv_avg": bhv_avg,
            "bhv_r1": bhv_r1,
            "nrfbx": nrfbx,
            "nncv_avg": nncv_avg,
            "nncv_r1": nncv_r1,
            "ncv_avg": ncv_avg,
            "ncv_r1": ncv_r1,
            "bncv_avg": bncv_avg,
            "bncv_r1": bncv_r1,
            "sup_avg": sup_avg,
            "sup_r1": sup_r1,
            "inf_avg": inf_avg,
            "inf_r1": inf_r1,
            "inc_avg": inc_avg,
            "inc_r1": inc_r1,
            "lnd_avg": lnd_avg,
            "lnd_r1": lnd_r1,
            "nfronts_avg": nfronts_avg,
            "nfronts_r1": nfronts_r1
        }

        # Set the class attributes
        for feature_name, feature_value in feature_dict.items():
            setattr(self, feature_name, feature_value)


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
                
        # Initialise feature arrays.
        self.feature_names = RandomWalkAnalysis.feature_names

        # Initialise arrays to store these values.
        super().initialize_arrays()

    @staticmethod
    def concatenate_multiple_analyses(multiple_analyses):
        """
        Concatenate feature arrays from multiple MultipleRandomWalkAnalysis objects into one.
        """

        # Create a new MultipleAnalysis object with the combined populations
        combined_analysis = MultipleRandomWalkAnalysis([], [])
        
        # Iterate through feature names
        for feature_name in multiple_analyses[0].feature_names:
            feature_arrays = [getattr(ma, f"{feature_name}_array") for ma in multiple_analyses]
            combined_array = np.concatenate(feature_arrays)
            
            # Save as attribute array in the combined_analysis object
            setattr(combined_analysis, f"{feature_name}_array", combined_array)

        return combined_analysis