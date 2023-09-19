import numpy as np
from optimisation.model.individual import Individual
from features.cv_distr import cv_distr
from features.cv_mdl import cv_mdl
from features.rank_mdl import rank_mdl
from features.dist_corr import dist_corr
from features.f_corr import f_corr
from features.f_decdist import f_decdist
from features.f_skew import f_skew
from features.fvc import fvc
from features.PiIZ import PiIZ
from optimisation.util.non_dominated_sorting import NonDominatedSorting


class FitnessAnalysis:
    """
    Calculate all features generated from a random sample.

    """

    def __init__(self, pop):
        """
        Population must already be evaluated.
        """
        self.pop = pop

    def eval_fitness_features(self):
        # TODO: check each of these work properly.
        self.corr_cf = self.get_corr_cf()
        self.f_mdl_r2 = self.get_f_mdl_r2()
        self.dist_c_corr = self.get_dist_c_corr()
        self.min_cv = self.get_min_cv()
        self.skew_rnge = self.get_skew_rnge()
        self.piz_ob_min = self.get_piz_ob_min()
        self.ps_dist_iqr_mean = self.get_ps_dist_iqr_mean()
        self.cpo_upo_n = self.get_cpo_upo_n()
        self.cv_range_coeff = self.get_cv_range_coeff()
        self.corr_obj = self.get_corr_obj()
        self.cv_mdl_r2 = self.get_cv_mdl_r2()

    def get_corr_cf(self):
        return fvc(self.pop)[1]

    def get_f_mdl_r2(self):
        return rank_mdl(self.pop)[0]

    def get_dist_c_corr(self):
        return dist_corr(self.pop, self.pop.extract_nondominated())

    def get_min_cv(self):
        return cv_distr(self.pop)[2]

    def get_skew_rnge(self):
        return f_skew(self.pop)[-1]

    def get_piz_ob_min(self):
        return np.min(PiIZ(self.pop)[0])

    def get_ps_dist_iqr_mean(self):
        return f_decdist(self.pop, 1, 1)[-1]

    def get_cpo_upo_n(self):
        return

    def get_cv_range_coeff(self):
        return

    def get_corr_obj(self):
        return

    def get_cv_mdl_r2(self):
        return


class MultipleFitnessAnalysis(np.ndarray):
    """
    Aggregate global features across populations/walks.
    """

    def __new__(cls, pops):
        obj = (
            super(MultipleFitnessAnalysis, cls)
            .__new__(cls, len(pops), dtype=cls)
            .view(cls)
        )
        for i in range(len(pops)):
            obj[i] = FitnessAnalysis(pops[i])

        return obj

    def eval_features_for_all_populations(self):
        """
        Evaluate features for all populations.
        """

        for i in range(len(self)):
            self[i].eval_fitness_features()

    def generate_array_for_attribute(self, attribute_name):
        attribute_array = []
        for i in range(len(self)):
            attribute_array.append(getattr(self[i], attribute_name))
        return np.asarray(attribute_array)

    def aggregate_features(self):
        """
        Aggregate features for all populations. Must be run after eval_features_for_all_populations.
        """
        self.f_mdl_r2 = np.mean(self.generate_array_for_attribute("f_mdl_r2"))
        self.dist_c_corr = np.mean(self.generate_array_for_attribute("dist_c_corr"))
        self.min_cv = np.mean(self.generate_array_for_attribute("min_cv"))
        self.skew_rnge = np.mean(self.generate_array_for_attribute("skew_rnge"))
        self.piz_ob_min = np.mean(self.generate_array_for_attribute("piz_ob_min"))
        self.ps_dist_iqr_mean = np.mean(
            self.generate_array_for_attribute("get_ps_dist_iqr_mean")
        )
        self.cpo_upo_n = np.mean(self.generate_array_for_attribute("cpo_upo_n"))
        self.cv_range_coeff = np.mean(
            self.generate_array_for_attribute("cv_range_coeff")
        )
        self.corr_obj = np.mean(self.generate_array_for_attribute("corr_obj"))
        self.cv_mdl_r2 = np.mean(self.generate_array_for_attribute("cv_mdl_r2"))
