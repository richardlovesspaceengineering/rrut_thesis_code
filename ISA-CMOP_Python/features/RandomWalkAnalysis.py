import numpy as np
from features.randomwalkfeatures import (
    compute_neighbourhood_distance_features,
    compute_neighbourhood_hv_features
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
        
        # Evaluate neighbourhood distance features.
        dist_x_avg, dist_x_r1, dist_f_avg, dist_f_r1, dist_c_avg, dist_c_r1, dist_f_c_avg, dist_f_c_r1, dist_f_dist_x_avg, dist_f_dist_x_r1, dist_c_dist_x_avg, dist_c_dist_x_r1, dist_f_c_dist_x_avg, dist_f_c_dist_x_r1 = compute_neighbourhood_distance_features(self.pop, self.pop_neighbours_list)
        
        # Evaluate neighbourhood HV features.
        hv_single_soln_avg, hv_single_soln_r1, nhv_avg, nhv_r1, hvd_avg, hvd_r1, bhv_avg, bhv_r1 = compute_neighbourhood_hv_features(self.pop, self.pop_neighbours_list)
        
        # Create a dictionary to store feature names and values
        feature_dict = {
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
            "bhv_r1": bhv_r1
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
        # Combine the 

        # Create a new MultipleAnalysis object with the combined populations
        combined_analysis = MultipleRandomWalkAnalysis([], [])
        
        # Iterate through feature names
        for feature_name in multiple_analyses[0].feature_names:
            feature_arrays = [getattr(ma, f"{feature_name}_array") for ma in multiple_analyses]
            combined_array = np.concatenate(feature_arrays)
            
            # Save as attribute array in the combined_analysis object
            setattr(combined_analysis, f"{feature_name}_array", combined_array)

        return combined_analysis