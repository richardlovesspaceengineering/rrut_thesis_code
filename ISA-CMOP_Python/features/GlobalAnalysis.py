import numpy as np
from features.globalfeatures import *

from scipy.stats import yeojohnson
from features.Analysis import Analysis, MultipleAnalysis


class GlobalAnalysis(Analysis):
    """
    Calculate all features generated from a random sample.
    """

    def eval_features(self):
        # Remove any samples if they contain infs or nans.
        new_pop, _ = self.pop.remove_nan_inf_rows("global", re_evaluate=True)

        # Global scr. "glob" will be appended to the name in the results file.
        self.features["scr"] = compute_solver_crash_ratio(self.pop, new_pop)

        # Now work with the trimmed population from now on.
        self.pop = new_pop

        # Feasibility
        self.features["fsr"] = compute_fsr(self.pop)

        # Correlation of objectives.
        (
            self.features["corr_obj_min"],
            self.features["corr_obj_max"],
            self.features["corr_obj_range"],
        ) = corr_obj(self.pop)

        # Skewness of objective values.
        (
            self.features["skew_avg"],
            self.features["skew_min"],
            self.features["skew_max"],
            self.features["skew_rnge"],
        ) = obj_skew(self.pop)

        # Kurtosis of objective values.
        (
            self.features["kurt_avg"],
            self.features["kurt_min"],
            self.features["kurt_max"],
            self.features["kurt_rnge"],
        ) = obj_kurt(self.pop)

        # Distribution of unconstrained ranks.
        (
            self.features["mean_uc_rk"],
            self.features["std_uc_rk"],
            self.features["min_uc_rk"],
            self.features["max_uc_rk"],
            self.features["skew_uc_rk"],
            self.features["kurt_uc_rk"],
        ) = uc_rk_distr(self.pop)

        # Distribution of CV (normalised).
        (
            self.features["mean_cv"],
            self.features["std_cv"],
            self.features["min_cv"],
            self.features["max_cv"],
            self.features["skew_cv"],
            self.features["kurt_cv"],
        ) = cv_distr(self.pop, self.normalisation_values, norm_method="95th")

        # Proportion of solutions in ideal zone per objectives and overall proportion of solutions in ideal zone.
        (
            self.features["piz_ob_min"],
            self.features["piz_ob_max"],
            self.features["piz_ob_f"],
        ) = PiIZ(self.pop)

        # Pareto set and front properties (normalised).
        (
            self.features["PS_dist_max"],
            self.features["PS_dist_mean"],
            self.features["PS_dist_iqr"],
            self.features["PF_dist_max"],
            self.features["PF_dist_mean"],
            self.features["PF_dist_iqr"],
        ) = compute_ps_pf_distances(
            self.pop, self.normalisation_values, norm_method="95th"
        )

        # Get PF-UPF relationship features.
        (
            self.features["po_n"],
            self.features["cpo_upo_n"],
            self.features["cover_cpo_upo_n"],
        ) = compute_PF_UPF_features(self.pop)

        # Extract violation-distance correlation.
        self.features["dist_c_corr"] = dist_corr(
            self.pop, self.pop.extract_nondominated()
        )

        # Correlations of objectives with cv, unconstrained ranks and then cv with ranks.
        (
            self.features["corr_obj_cv_min"],
            self.features["corr_obj_cv_max"],
            self.features["corr_obj_uc_rk_min"],
            self.features["corr_obj_uc_rk_max"],
            self.features["corr_cv_ranks"],
        ) = compute_ranks_cv_corr(self.pop)

        # Decision variables-unconstrained ranks model properties.
        (
            self.features["rk_uc_mdl_r2"],
            self.features["rk_uc_range_coeff"],
        ) = rk_uc_var_mdl(self.pop)

        # Decision variables-CV model properties.
        (
            self.features["cv_mdl_r2"],
            self.features["cv_range_coeff"],
        ) = rk_uc_var_mdl(self.pop)


class MultipleGlobalAnalysis(MultipleAnalysis):
    """
    Aggregate global features across populations.
    """

    def __init__(self, pops, normalisation_values):
        super().__init__(pops, normalisation_values, GlobalAnalysis)
