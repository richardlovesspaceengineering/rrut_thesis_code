import numpy as np
from features.globalfeatures import *

from scipy.stats import yeojohnson
from features.Analysis import Analysis, MultipleAnalysis


class GlobalAnalysis(Analysis):
    """
    Calculate all features generated from a random sample.
    """

    def eval_features(self, pop_global):
        # Remove any samples if they contain infs or nans.
        new_pop, _ = pop_global.remove_nan_inf_rows("global", re_evaluate=True)

        # Global scr. "glob" will be appended to the name in the results file.
        self.features["scr"] = compute_solver_crash_ratio(pop_global, new_pop)

        # Now work with the trimmed population from now on.
        pop_global = new_pop

        # Feasibility
        self.features["fsr"] = compute_fsr(pop_global)

        # Correlation of objectives.
        (
            self.features["corr_obj_min"],
            self.features["corr_obj_max"],
            self.features["corr_obj_range"],
        ) = corr_obj(pop_global)

        # Skewness of objective values.
        (
            self.features["skew_avg"],
            self.features["skew_min"],
            self.features["skew_max"],
            self.features["skew_rnge"],
        ) = obj_skew(pop_global)

        # Kurtosis of objective values.
        (
            self.features["kurt_avg"],
            self.features["kurt_min"],
            self.features["kurt_max"],
            self.features["kurt_rnge"],
        ) = obj_kurt(pop_global)

        # Distribution of unconstrained ranks.
        (
            self.features["mean_uc_rk"],
            self.features["std_uc_rk"],
            self.features["min_uc_rk"],
            self.features["max_uc_rk"],
            self.features["skew_uc_rk"],
            self.features["kurt_uc_rk"],
        ) = uc_rk_distr(pop_global)

        # Distribution of CV (normalised).
        (
            self.features["mean_cv"],
            self.features["std_cv"],
            self.features["min_cv"],
            self.features["max_cv"],
            self.features["skew_cv"],
            self.features["kurt_cv"],
        ) = cv_distr(pop_global, self.normalisation_values, norm_method="95th")

        # Proportion of solutions in ideal zone per objectives and overall proportion of solutions in ideal zone.
        (
            self.features["piz_ob_min"],
            self.features["piz_ob_max"],
            self.features["piz_ob_f"],
        ) = PiIZ(pop_global)

        # Pareto set and front properties (normalised).
        (
            self.features["PFd"],
            self.features["PFCV"],
            self.features["PS_dist_max"],
            self.features["PS_dist_mean"],
            self.features["PS_dist_iqr"],
            self.features["PF_dist_max"],
            self.features["PF_dist_mean"],
            self.features["PF_dist_iqr"],
        ) = compute_ps_pf_distances(
            pop_global, self.normalisation_values, norm_method="95th"
        )

        # Get PF-UPF relationship features.
        (
            self.features["hv_est"],
            self.features["uhv_est"],
            self.features["hv_uhv_n"],
            self.features["GD_cpo_upo"],
            self.features["upo_n"],
            self.features["po_n"],
            self.features["cpo_upo_n"],
            self.features["cover_cpo_upo_n"],
        ) = compute_PF_UPF_features(
            pop_global, self.normalisation_values, norm_method="95th"
        )

        # Extract violation-distance correlation.
        self.features["dist_c_corr"] = dist_corr(
            pop_global, pop_global.extract_nondominated()
        )

        # Correlations of objectives with cv, unconstrained ranks and then cv with ranks.
        (
            self.features["corr_obj_cv_min"],
            self.features["corr_obj_cv_max"],
            self.features["corr_obj_uc_rk_min"],
            self.features["corr_obj_uc_rk_max"],
            self.features["corr_cv_ranks"],
        ) = compute_ranks_cv_corr(pop_global)

        # Decision variables-unconstrained ranks model properties.
        (
            self.features["rk_uc_mdl_r2"],
            self.features["rk_uc_range_coeff"],
        ) = rk_uc_var_mdl(pop_global)

        # Decision variables-CV model properties.
        (
            self.features["cv_mdl_r2"],
            self.features["cv_range_coeff"],
        ) = rk_uc_var_mdl(pop_global)

        # Information content features.
        (
            self.features["H_max"],
            self.features["eps_s"],
            self.features["m0"],
            self.features["eps05"],
        ) = compute_ic_features(pop_global, sample_type="global")
