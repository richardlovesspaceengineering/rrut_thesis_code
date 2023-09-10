from cases.MW_setup import MW3
import numpy as np
from optimisation.model.individual import Individual
from features.cv_distr import cv_distr
from features.cv_mdl import cv_mdl
from features.dist_corr import dist_corr
from features.f_corr import f_corr
from features.f_decdist import f_decdist
from features.f_skew import f_skew


class LandscapeFeatures:
    def __init__(self, pop):
        """
        Population must already be evaluated.
        """
        self.pop = pop

    def compute_landscape_features(self):
        self.corr_cf = self.get_corr_cf()
        self.f_mdl_r2 = self.get_f_mdl_r2()
        self.dist_c_corr = self.get_dist_c_corr()
        self.min_cv = self.get_min_cv()
        self.bhv_avg_rws = self.get_bhv_avg_rws()
        self.skew_rnge = self.get_skew_rnge()
        self.piz_ob_min = self.get_piz_ob_min()
        self.get_ps_dist_iqr_mean = self.get_ps_dist_iqr_mean()
        self.dist_c_dist_x_avg_rws = self.get_dist_c_dist_x_avg_rws()
        self.cpo_upo_n = self.get_cpo_upo_n()
        self.cv_range_coeff = self.get_cv_range_coeff()
        self.corr_obj = self.get_corr_obj()
        self.dist_f_dist_x_avg_rws = self.get_dist_f_dist_x_avg_rws()
        self.cv_mdl_r2 = self.get_cv_mdl_r2()

    def get_corr_cf(self):
        return

    def get_f_mdl_r2(self):
        return

    def get_dist_c_corr(self):
        return

    def get_min_cv(self):
        return

    def get_bhv_avg_rws(self):
        return

    def get_skew_rnge(self):
        return

    def get_piz_ob_min(self):
        return

    def get_ps_dist_iqr_mean(self):
        return

    def get_dist_c_dist_x_avg_rws(self):
        return

    def get_cpo_upo_n(self):
        return

    def get_cv_range_coeff(self):
        return

    def get_corr_obj(self):
        return

    def get_dist_f_dist_x_avg_rws(self):
        return

    def get_cv_mdl_r2(self):
        return
