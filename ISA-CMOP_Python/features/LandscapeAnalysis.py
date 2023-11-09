import numpy as np
from scipy.stats import yeojohnson
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from features.GlobalAnalysis import GlobalAnalysis
from features.RandomWalkAnalysis import RandomWalkAnalysis

import os
from datetime import datetime


class LandscapeAnalysis:
    """
    Collates all features for one sample.
    """

    def __init__(self, globalanalysis, randomwalkanalysis):
        """
        Give instances of MultipleGlobalAnalysis and MultipleRandomWalkAnalysis here.
        """
        self.globalanalysis = globalanalysis
        self.randomwalkanalysis = randomwalkanalysis

        # Initialise features.
        self.feature_names = (
            self.globalanalysis.feature_names + self.randomwalkanalysis.feature_names
        )

        self.initialize_arrays_and_scalars()

    def initialize_arrays_and_scalars(self):
        for feature in self.feature_names:
            if feature in self.globalanalysis.feature_names:
                array_length = len(self.globalanalysis.pops)
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
        for feature_name in self.feature_names:
            if feature_name in self.globalanalysis.feature_names:
                setattr(
                    self,
                    f"{feature_name}_array",
                    getattr(self.globalanalysis, f"{feature_name}_array"),
                )
            elif feature_name in self.randomwalkanalysis.feature_names:
                setattr(
                    self,
                    f"{feature_name}_array",
                    getattr(self.randomwalkanalysis, f"{feature_name}_array"),
                )
            else:
                # Handle cases where feature_name is not found in either set of feature names
                pass

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

    def compute_statistic_for_feature(self, array, stat="mean"):
        if stat == "mean":
            return np.mean(array)
        elif stat == "median":
            return np.median(array)
        elif stat == "min":
            return np.min(array)
        elif stat == "max":
            return np.max(array)
        elif stat == "std":
            return np.std(array)
        else:
            raise ValueError(
                "Invalid statistic choice. Use 'mean', 'median', 'min', 'max', or 'std'."
            )

    def aggregate_features(self):
        """
        Aggregate feature for all populations. Must be called after extract_feature_arrays.
        """

        # Initialise list of aggregated feature names.
        self.aggregated_feature_names = []

        for feature_name in self.feature_names:
            if feature_name in ["nrfbx"]:
                statistic = ["mean", "min", "max", "median"]
            else:
                statistic = ["mean", "std"]

            if isinstance(statistic, list):
                # Compute and set multiple statistics

                for stat in statistic:
                    # Replace feature name with more descriptive names.
                    new_name = f"{feature_name}_{stat}"

                    setattr(
                        self,
                        new_name,
                        self.compute_statistic_for_feature(
                            getattr(self, f"{feature_name}_array"), stat
                        ),
                    )
                    self.aggregated_feature_names.append(new_name)

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
        for feature_name in self.aggregated_feature_names:
            dat[feature_name] = [getattr(self, f"{feature_name}")]
        return dat

    def make_unaggregated_feature_tables(self):
        """
        Create an n-samples-row table of all the features to allow comparison.
        """
        global_dat = pd.DataFrame()
        rw_dat = pd.DataFrame()
        for feature_name in self.feature_names:
            if feature_name in self.globalanalysis.feature_names:
                global_dat[feature_name] = getattr(self, f"{feature_name}_array")
            elif feature_name in self.randomwalkanalysis.feature_names:
                rw_dat[feature_name] = getattr(self, f"{feature_name}_array")
        return global_dat, rw_dat

    def export_unaggregated_feature_table(self, dat, instance_name, sampling_method):
        """
        Write Pandas dataframe of raw features results to a csv file.
        """
        # Create a folder if it doesn't exist
        results_folder = "instance_results"
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        # Get the current date and time
        current_time = datetime.now().strftime("%b%d_%H%M")

        # Create the file path
        file_path = os.path.join(
            results_folder,
            f"{instance_name}_{sampling_method}_features_{current_time}.csv",
        )

        # Save the DataFrame to a CSV file
        dat.to_csv(file_path, index=False)

        return dat

    def make_unaggregated_global_feature_table(self):
        return self.make_unaggregated_feature_table(self.globalanalysis.feature_names)

    def make_unaggregated_rw_feature_table(self):
        return self.make_unaggregated_feature_table(
            self.randomwalkanalysis.feature_names
        )

    def extract_experimental_results(self, csv_name="data/raw_features_alsouly.csv"):
        problem = self.globalanalysis.pops[0][0].problem
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
