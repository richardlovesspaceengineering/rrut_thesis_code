import numpy as np
from features.randomwalkfeatures import compute_neighbourhood_features
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
        
        # Save as computed values for this walk.
        self.feature_names = [
            "dist_x_avg",
            "dist_f_avg",
            "dist_c_avg",
            "dist_f_dist_x_avg",
            "dist_c_dist_x_avg"
        ]
        
        # Initialise values.
        self.initialize_features()

    def eval_features(self):
        """
        Evaluate features along the random walk and save to class.
        """
        
        # Evaluate neighbourhood features.
        dist_x_avg, dist_f_avg, dist_c_avg, dist_f_dist_x_avg, dist_c_dist_x_avg = compute_neighbourhood_features(self.pop, self.pop_neighbours_list, self.pareto_front)
        
        # Set the class attributes
        self.dist_x_avg = dist_x_avg
        self.dist_f_avg = dist_f_avg
        self.dist_c_avg = dist_c_avg
        self.dist_f_dist_x_avg = dist_f_dist_x_avg
        self.dist_c_dist_x_avg = dist_c_dist_x_avg


class MultipleRandomWalkAnalysis(MultipleAnalysis):
    """
    Aggregate RW features across populations/walks.
    """

    def __init__(self, pops_walks, pops_neighbours_list):
        self.pops = pops_walks
        self.analyses = []
        for ctr, pop in enumerate(pops_walks):
            self.analyses.append(RandomWalkAnalysis(pop, pops_neighbours_list[ctr]))
        self.feature_names = self.analyses[0].feature_names

        # Initialise arrays to store these values.
        super().initialize_arrays()
