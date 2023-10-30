import numpy as np
from features.randomwalkfeatures import compute_neighbourhood_features
from scipy.stats import yeojohnson
from features.Analysis import Analysis, MultipleAnalysis


class RandomWalkAnalysis(Analysis):
    """
    Calculate all features generated from random walk samples.

    Populations is a list of populations that represents a walk, each entry is a solution and its neighbours.
    """

    def __init__(self, pop_walk, pop_neighbour):
        """
        Populations must already be evaluated.
        """
        super().__init__(pop_walk)
        self.pop_neighbour = pop_neighbour

    def eval_features(self):
        """
        Evaluate features along the random walk.
        """
        
        # Evaluate neighbourhood features.
        dist_f_dist_x_avg_rws, dist_c_dist_x_avg_rws, bhv_avg_rws = compute_neighbourhood_features(self.pop, self.pop_neighbour, self.pareto_front)
        
        # Save as computed values for this walk.
        self.bhv_avg_rws = bhv_avg_rws
        self.dist_c_dist_x_avg_rws = dist_c_dist_x_avg_rws
        self.dist_f_dist_x_avg_rws = dist_f_dist_x_avg_rws


class MultipleRandomWalkAnalysis(MultipleAnalysis):
    """
    Aggregate RW features across populations/walks.
    """

    def __init__(self, pops):
        super().__init__(pops, RandomWalkAnalysis)

        self.feature_names = [
            "dist_c_dist_x_avg_rws",
            "dist_f_dist_x_avg_rws",
            "bhv_avg_rws",
        ]

        # Initialise values
        super().initialize_arrays()
