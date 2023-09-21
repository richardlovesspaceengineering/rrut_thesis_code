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
        dist_f_dist_x_avg, dist_c_dist_x_avg, bhv = randomwalkfeatures(
            self.pop, self.pareto_front, Instances=None
        )
        self.bhv = bhv
        self.dist_c_dist_x_avg = dist_c_dist_x_avg
        self.dist_f_dist_x_avg = dist_f_dist_x_avg


class MultipleRandomWalkAnalysis(MultipleAnalysis):
    """
    Aggregate RW features across populations/walks.
    """

    def __init__(self, pops):
        super().__init__(pops, RandomWalkAnalysis)

        # Initialising feature arrays.
        self.dist_f_dist_x_avg_rws_array = np.empty(len(pops))
        self.dist_c_dist_x_avg_rws_array = np.empty(len(pops))
        self.bhv_avg_rws_array = np.empty(len(pops))

        # Initialising features.
        self.dist_f_dist_x_avg_rws_array = np.nan
        self.dist_c_dist_x_avg_rws_array = np.nan
        self.bhv_avg_rws_array = np.nan

    def generate_feature_arrays(self):
        """
        Collate features into an array. Must be run after eval_features_for_all_populations.
        """
        self.dist_f_dist_x_avg_rws_array = self.generate_array_for_attribute(
            "dist_f_dist_x_avg"
        )
        self.dist_c_dist_x_avg_rws_array = self.generate_array_for_attribute(
            "dist_c_dist_x_avg"
        )
        self.bhv_avg_rws_array = self.generate_array_for_attribute("bhv")

    def aggregate_features(self, YJ_transform=True):
        """
        Aggregate features for all populations.
        """

        attribute_names = [
            "dist_c_dist_x_avg_rws",
            "dist_f_dist_x_avg_rws",
            "bhv_avg_rws",
        ]

        self.aggregate_features_from_names_list(attribute_names)
