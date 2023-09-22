import numpy as np
from scipy.stats import yeojohnson
import pandas as pd


class LandscapeAnalysis:
    """
    Collates all features for one sample.
    """

    def __init__(self, fitnessanalysis, randomwalkanalysis):
        """
        Give instances of MultipleFitnessAnalysis and MultipleRandomWalkAnalysis here.
        """
        self.fitnessanalysis = fitnessanalysis
        self.randomwalkanalysis = randomwalkanalysis
        projection_matrix = [
            0.2559,
            0.1348,
            -0.2469,
            -0.1649,
            -0.0257,
            -0.2703,
            0.2938,
            -0.2278,
            -0.2148,
            -0.1338,
            -0.1935,
            -0.2210,
            -0.1651,
            0.2998,
            -0.2150,
            0.3137,
            0.3067,
            0.1382,
            0.0709,
            0.3047,
            0.2032,
            -0.0515,
            0.1436,
            0.2869,
            0.1940,
            0.1154,
            -0.0508,
            -0.2466,
        ]

        # Reshape the data into a 2x14 array
        self.projection_matrix = np.transpose(
            np.array(projection_matrix).reshape(14, 2)
        )

        # Initialise features.
        self.feature_names = [
            "feasibility_ratio",
            "corr_cf",
            "f_mdl_r2",
            "dist_c_corr",
            "min_cv",
            "bhv_avg_rws",
            "skew_rnge",
            "piz_ob_min",
            "ps_dist_iqr_mean",
            "dist_c_dist_x_avg_rws",
            "cpo_upo_n",
            "cv_range_coeff",
            "corr_obj",
            "dist_f_dist_x_avg_rws",
            "cv_mdl_r2",
        ]

        # TODO: initialize feature arrays.

    def extract_feature_arrays(self):
        """
        Save feature arrays into this instance.
        """
        # For self.fitnessanalysis attributes
        self.feasibility_ratio_array = self.fitnessanalysis.feasibility_ratio_array
        self.corr_cf_array = self.fitnessanalysis.corr_cf_array
        self.f_mdl_r2_array = self.fitnessanalysis.f_mdl_r2_array
        self.dist_c_corr_array = self.fitnessanalysis.dist_c_corr_array
        self.min_cv_array = self.fitnessanalysis.min_cv_array
        self.skew_rnge_array = self.fitnessanalysis.skew_rnge_array
        self.piz_ob_min_array = self.fitnessanalysis.piz_ob_min_array
        self.ps_dist_iqr_mean_array = self.fitnessanalysis.ps_dist_iqr_mean_array
        self.cpo_upo_n_array = self.fitnessanalysis.cpo_upo_n_array
        self.cv_range_coeff_array = self.fitnessanalysis.cv_range_coeff_array
        self.corr_obj_array = self.fitnessanalysis.corr_obj_array
        self.cv_mdl_r2_array = self.fitnessanalysis.cv_mdl_r2_array

        # For self.randomwalkanalysis attributes
        self.bhv_avg_rws_array = self.randomwalkanalysis.bhv_avg_rws_array
        self.dist_c_dist_x_avg_rws_array = (
            self.randomwalkanalysis.dist_c_dist_x_avg_rws_array
        )
        self.dist_f_dist_x_avg_rws_array = (
            self.randomwalkanalysis.dist_f_dist_x_avg_rws_array
        )

    def apply_YJ_transform(self, array):
        return yeojohnson(array)[0]

    def aggregate_array_for_feature(self, array, YJ_transform):
        if YJ_transform:
            return np.mean(self.apply_YJ_transform(array))
        else:
            return np.mean(array)

    def aggregate_features(self, YJ_transform):
        """
        Aggregate feature for all populations. Must be called after extract_feature_arrays.
        """
        for feature_name in self.feature_names:
            setattr(
                self,
                feature_name,
                self.aggregate_array_for_feature(
                    getattr(self, f"{feature_name}_array"), YJ_transform
                ),
            )

    def extract_features_vector(self):
        """
        Combine aggregated features into a single vector, ready to be projected by the projection matrix in Eq. (13) of Alsouly.

        Must be called after extract_feature_scalars.

        Returns a column vector ready for vector operations.
        """
        self.features_vector = np.array(
            [
                self.corr_cf,
                self.f_mdl_r2,
                self.dist_c_corr,
                self.min_cv,
                self.bhv_avg_rws,
                self.skew_rnge,
                self.piz_ob_min,
                self.ps_dist_iqr_mean,
                self.dist_c_dist_x_avg_rws,
                self.cpo_upo_n,
                self.cv_range_coeff,
                self.corr_obj,
                self.dist_f_dist_x_avg_rws,
                self.cv_mdl_r2,
            ],
            ndmin=2,
        ).reshape((-1, 1))

    def map_features_to_instance_space(self):
        """
        Run after combine_features.
        """
        self.instance_space = self.projection_matrix @ self.features_vector

    def make_aggregated_feature_table(self):
        """
        Create a 1-row table of all the features to allow comparison.
        """
        dat = pd.DataFrame(columns=self.feature_names)
        for feature_name in self.feature_names:
            dat[feature_name] = getattr(self, f"{feature_name}")
        return dat

    def make_unaggregated_feature_table(self, feature_names):
        """
        Create an n-samples-row table of all the features to allow comparison.
        """
        dat = pd.DataFrame()
        for feature_name in feature_names:
            dat[feature_name] = getattr(self, f"{feature_name}_array")
        return dat

    def make_unaggregated_global_feature_table(self):
        return self.make_unaggregated_feature_table(self.fitnessanalysis.feature_names)

    def make_unaggregated_rw_feature_table(self):
        return self.make_unaggregated_feature_table(
            self.randomwalkanalysis.feature_names
        )
