import numpy as np
from features.cv_distr import cv_distr
from features.cv_mdl import cv_mdl
from features.rank_mdl import rank_mdl
from features.dist_corr import dist_corr
from features.f_corr import f_corr
from features.f_decdist import f_decdist
from features.f_skew import f_skew
from features.fvc import fvc
from features.randomwalkfeatures import randomwalkfeatures


class RandomWalkAnalysis:
    """
    Calculate all features generated from random walk samples.

    Populations is a list of populations that represents a walk, each entry is a solution and its neighbours.
    """

    def __init__(self, pop):
        """
        Populations must already be evaluated.
        """
        self.pop = pop
        self.pareto_front = pop[0].pareto_front

    def eval_rw_features(self):
        dist_f_dist_x_avg, dist_c_dist_x_avg, bhv = randomwalkfeatures(
            self.pop, self.pareto_front, Instances=None
        )
        self.bhv = bhv
        self.dist_c_dist_x_avg = dist_c_dist_x_avg
        self.dist_f_dist_x_avg = dist_f_dist_x_avg


class MultipleRandomWalkAnalysis(np.ndarray):
    """
    Aggregate RW features across populations/walks.
    """

    def __new__(cls, pops):
        obj = (
            super(RandomWalkAnalysis, cls).__new__(cls, len(pops), dtype=cls).view(cls)
        )
        for i in range(len(pops)):
            obj[i] = RandomWalkAnalysis(pops[i])

        return obj

    def eval_features_for_all_populations(self):
        """
        Evaluate features for all populations.
        """
        return

    def extract_dist_f_dist_x_avg_array(self):
        return

    def extract_dist_c_dist_x_avg_array(self):
        return

    def extract_bhv_array(self):
        return

    def aggregate_features(self):
        """
        Aggregate features for all populations. Must be run after eval_features_for_all_populations.
        """
        dist_x_avg = np.zeros(len(self.analyses))
        dist_f_avg = np.zeros(len(self.analyses))
        dist_c_avg = np.zeros(len(self.analyses))
        bhv = np.zeros(len(self.analyses))

        # for i, pop in enumerate(self.analyses):
