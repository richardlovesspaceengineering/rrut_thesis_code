import numpy as np


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

    def combine_features(self):
        """
        Combine features into a single vector, ready to be projected by the projection matrix in Eq. (13) of Alsouly.

        Returns a column vector ready for vector operations.
        """
        self.features = np.array(
            [
                self.fitnessanalysis.corr_cf,
                self.fitnessanalysis.f_mdl_r2,
                self.fitnessanalysis.dist_c_corr,
                self.fitnessanalysis.min_cv,
                self.randomwalkanalysis.bhv_avg_rws,
                self.fitnessanalysis.skew_rnge,
                self.fitnessanalysis.piz_ob_min,
                self.fitnessanalysis.ps_dist_iqr_mean,
                self.randomwalkanalysis.dist_c_dist_x_avg_rws,
                self.fitnessanalysis.cpo_upo_n,
                self.fitnessanalysis.cv_range_coeff,
                self.fitnessanalysis.corr_obj,
                self.randomwalkanalysis.dist_f_dist_x_avg_rws,
                self.fitnessanalysis.cv_mdl_r2,
            ],
            ndmin=2,
        ).reshape((-1, 1))

    def map_features_to_instance_space(self):
        """
        Run after combine_features.
        """
        self.instance_space = self.projection_matrix @ self.features
