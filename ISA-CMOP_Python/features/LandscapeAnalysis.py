import numpy as np
from scipy.stats import yeojohnson
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math


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
            "fsr",
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

        self.initialize_arrays_and_scalars()

    def initialize_arrays_and_scalars(self):
        for feature in self.feature_names:
            if feature in self.fitnessanalysis.feature_names:
                array_length = len(self.fitnessanalysis.pops)
            else:
                array_length = len(self.randomwalkanalysis.pops)

            # Initialising feature arrays.
            setattr(
                self,
                (f"{feature}_array"),
                np.empty(array_length),
            )

        # Initialising feature values.
        for feature in self.feature_names:
            setattr(
                self,
                (f"{feature}"),
                np.nan,
            )

    def extract_feature_arrays(self):
        """
        Save feature arrays into this instance.
        """
        # For self.fitnessanalysis attributes
        self.fsr_array = self.fitnessanalysis.fsr_array
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

    def plot_feature_histograms(self, num_bins=20):
        """
        Plot histograms for each feature array.
        """
        # Calculate the number of rows and columns for a close-to-square grid
        num_features = len(self.feature_names)
        num_cols = int(math.ceil(math.sqrt(num_features)))
        num_rows = int(math.ceil(num_features / num_cols))

        # Create a subplot grid based on the number of features
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))

        # Flatten the axes array for easier iteration
        axes = axes.ravel()

        for i, feature_name in enumerate(self.feature_names):
            # Select the current axis
            ax = axes[i]

            # Calculate the bin width based on the data range and the desired number of bins
            feature_array = getattr(self, f"{feature_name}_array")
            data_range = feature_array.max() - feature_array.min()
            bin_width = data_range / num_bins

            # Plot a histogram for the feature array with the specified number of bins
            sns.histplot(
                getattr(self, f"{feature_name}_array"),
                ax=ax,
                bins=num_bins,
                kde=False,
                stat="probability",  # Normalize to proportions
            )
            ax.set_xlabel(feature_name)
            ax.set_ylabel("")

        # Set a global y-axis label
        fig.text(0.02, 0.5, "Proportion", va="center", rotation="vertical", fontsize=14)

        # Remove any empty subplots (if the number of features is not a perfect square)
        for i in range(num_features, num_rows * num_cols):
            fig.delaxes(axes[i])

        # Adjust layout and display the plot
        plt.tight_layout()
        plt.show()

    def plot_feature_violin_box_plots(self):
        """
        Plot violin plots with overlaid box plots for each feature array.
        """
        # Calculate the number of rows and columns for a close-to-square grid
        num_features = len(self.feature_names)
        num_cols = int(math.ceil(math.sqrt(num_features)))
        num_rows = int(math.ceil(num_features / num_cols))

        # Create a subplot grid based on the number of features
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))

        # Flatten the axes array for easier iteration
        axes = axes.ravel()

        for i, feature_name in enumerate(self.feature_names):
            # Select the current axis
            ax = axes[i]

            # Plot a violin plot for the feature array
            sns.violinplot(
                x=getattr(self, f"{feature_name}_array"),
                ax=ax,
                inner="box",  # Overlay box plots
            )
            ax.set_xlabel(feature_name)

        # Remove any empty subplots (if the number of features is not a perfect square)
        for i in range(num_features, num_rows * num_cols):
            fig.delaxes(axes[i])

        # Adjust layout and display the plot
        plt.tight_layout()
        plt.show()

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

    def make_aggregated_feature_table(self, instance_name):
        """
        Create a 1-row table of all the features to allow comparison.
        """
        dat = pd.DataFrame()

        # Add problem name and number of dimensions.
        dat["Name"] = [instance_name]
        dat["D"] = [instance_name.split("d")[1]]

        # Add all features.
        for feature_name in self.feature_names:
            dat[feature_name] = [getattr(self, f"{feature_name}")]
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

    def extract_experimental_results(self, csv_name="data/raw_features_alsouly.csv"):
        problem = self.fitnessanalysis.pops[0][0].problem
        problem_name = problem.problem_name
        exp_dat = pd.read_csv(csv_name)

        # Extract relevant problem data
        exp_dat = exp_dat.loc[
            (exp_dat["Instances"] == problem_name)
            & (exp_dat["feature_D"] == problem.dim)
        ]

        # Slice to columns that we need
        cols = [
            col
            for col in exp_dat.columns
            if any(feature_name in col for feature_name in self.feature_names)
        ]
        exp_dat = exp_dat[["Instances"] + cols]

        # Rename columns to match own naming.
        new_cols1 = {
            item: item[len("feature_") :] if item.startswith("feature_") else item
            for item in exp_dat.columns
        }

        exp_dat = exp_dat.rename(columns=new_cols1)

        # Rename columns to match own naming.
        new_cols2 = {
            item: item[len("pop_") :] if item.startswith("pop_") else item
            for item in exp_dat.columns
        }

        exp_dat = exp_dat.rename(columns=new_cols2)
        exp_dat["Instances"] = problem_name + "_experimental"

        return exp_dat
