import numpy as np
from features.randomwalkfeatures import randomwalkfeatures
from scipy.stats import yeojohnson
from features.Analysis import Analysis, MultipleAnalysis


class RandomWalkAnalysis(Analysis):
    """
    Calculate all features generated from random walk samples.

    Populations is a list of populations that represents a walk, each entry is a solution and its neighbours.
    """

    def __init__(self, pop):
        """
        Populations must already be evaluated.
        """
        super().__init__(pop)

    def eval_features(self):
        dist_f_dist_x_avg_rws, dist_c_dist_x_avg_rws, bhv_avg_rws = randomwalkfeatures(
            self.pop, self.pareto_front, Instances=None
        )
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
