import numpy as np
from features.globalfeatures import (
    cv_distr,
    cv_mdl,
    rank_mdl,
    dist_corr,
    f_corr,
    f_decdist,
    f_skew,
    fvc,
    PiIZ,
)

from scipy.stats import yeojohnson
from features.Analysis import Analysis, MultipleAnalysis


class GlobalAnalysis(Analysis):
    """
    Calculate all features generated from a random sample.
    """

    # Define feature names as a static attribute at the class level
    feature_names = [
        "fsr",
        "corr_cf",
        "f_mdl_r2",
        "dist_c_corr",
        "min_cv",
        "skew_rnge",
        "piz_ob_min",
        "ps_dist_iqr_mean",
        "cpo_upo_n",
        "cv_range_coeff",
        "corr_obj",
        "cv_mdl_r2",
    ]

    def __init__(self, pop):
        """
        Population must already be evaluated.
        """
        self.pop = pop

    def eval_features(self):
        self.fsr = self.get_fsr()
        self.corr_cf = self.get_corr_cf()
        self.f_mdl_r2 = self.get_f_mdl_r2()
        self.dist_c_corr = self.get_dist_c_corr()
        self.min_cv = self.get_min_cv()
        self.skew_rnge = self.get_skew_rnge()
        self.piz_ob_min = self.get_piz_ob_min()
        self.ps_dist_iqr_mean = self.get_ps_dist_iqr_mean()
        self.cpo_upo_n = self.get_cpo_upo_n()
        self.corr_obj = self.get_corr_obj()

        # Fit linear model then save related features of interest.
        self.eval_cv_mdl()
        self.cv_range_coeff = self.get_cv_range_coeff()
        self.cv_mdl_r2 = self.get_cv_mdl_r2()

    def get_fsr(self):
        feasible = self.pop.extract_feasible()
        return len(feasible) / len(self.pop)

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
        nondominated_cons = self.pop.extract_nondominated(constrained=True)
        nondominated_uncons = self.pop.extract_nondominated(constrained=False)
        return len(nondominated_cons) / len(nondominated_uncons)

    def eval_cv_mdl(self):
        """
        Ensures we only need to fit the linear model once per population.
        """
        mdl_r2, range_coeff = cv_mdl(self.pop)
        self.cv_mdl_params = [mdl_r2, range_coeff]

    def get_cv_range_coeff(self):
        """
        Only works after eval_cv_mdl is run.
        """
        return self.cv_mdl_params[1]

    def get_corr_obj(self):
        return f_corr(self.pop)

    def get_cv_mdl_r2(self):
        """
        Only works after eval_cv_mdl is run.
        """
        return self.cv_mdl_params[0]


class MultipleGlobalAnalysis(MultipleAnalysis):
    """
    Aggregate global features across populations.
    """

    def __init__(self, pops):
        super().__init__(pops, GlobalAnalysis)

        super().initialize_arrays()
