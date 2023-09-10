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

    def __init__(self, pops):
        """
        Populations must already be evaluated.
        """
        self.pops = pops

    def eval_rw_features(self):
        self.bhv_avg_rws = self.get_bhv_avg_rws()
        self.dist_c_dist_x_avg_rws = self.get_dist_c_dist_x_avg_rws()
        self.dist_f_dist_x_avg_rws = self.get_dist_f_dist_x_avg_rws()
        self.cv_mdl_r2 = self.get_cv_mdl_r2()

    def get_bhv_avg_rws(self):
        return

    def get_dist_c_dist_x_avg_rws(self):
        return

    def get_dist_f_dist_x_avg_rws(self):
        return
