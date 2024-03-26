import numpy as np
from scipy.stats import yeojohnson
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from features.GlobalAnalysis import GlobalAnalysis
from features.RandomWalkAnalysis import RandomWalkAnalysis


class LandscapeAnalysis:
    """
    Collates all features for one sample.
    """

    def __init__(self, globalanalysis, randomwalkanalysis, adaptivewalkanalysis):
        """
        Give instances of MultipleGlobalAnalysis and MultipleRandomWalkAnalysis here.
        """
        self.analyses = {
            "glob": globalanalysis,
            "rw": randomwalkanalysis,
            "aw": adaptivewalkanalysis,
        }
        self.feature_arrays = {}
        self.aggregated_features = {}
        self.combine_all_feature_dicts()

    def append_to_features_dict(
        self, existing_features_dict, new_features_dict, method_suffix
    ):
        for feature_name, feature_value in new_features_dict.items():
            existing_features_dict[feature_name + "_" + method_suffix] = feature_value

        return existing_features_dict

    def combine_all_feature_dicts(self):
        new_dict = self.feature_arrays
        for suffix, a in self.analyses.items():
            if a:
                new_dict = self.append_to_features_dict(
                    new_dict, a.feature_arrays, suffix
                )

        # Save
        self.feature_arrays = new_dict

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

    def compute_statistic_for_feature(self, array, feature_name, stat="mean"):
        # Remove NaN values from the array and count them
        clean_array = np.array(array)[~np.isnan(array)]
        nan_count = np.count_nonzero(np.isnan(array))

        # Print the message about NaN removal
        if nan_count > 0:
            print(
                f"Removed {nan_count} NaN entries for feature '{feature_name}' when performing final aggregation."
            )

        # Compute the requested statistic on the array without NaNs
        if stat == "mean":
            return np.mean(clean_array)
        elif stat == "median":
            return np.median(clean_array)
        elif stat == "min":
            return np.min(clean_array)
        elif stat == "max":
            return np.max(clean_array)
        elif stat == "std":
            return np.std(clean_array)
        else:
            raise ValueError(
                "Invalid statistic choice. Use 'mean', 'median', 'min', 'max', or 'std'."
            )

    def aggregate_features(self):
        """
        Aggregate feature for all populations. Must be called after extract_feature_arrays.
        """

        for feature_name in self.feature_arrays:
            if feature_name in ["nrfbx"]:
                statistics = ["mean", "min", "max", "median"]
            else:
                statistics = ["mean", "std"]

            if isinstance(statistics, list):
                # Compute and set multiple statistics
                for stat in statistics:
                    # Replace feature name with more descriptive names.
                    new_name = f"{feature_name}_{stat}"

                    # Compute the statistic for the feature array
                    aggregated_value = self.compute_statistic_for_feature(
                        self.feature_arrays[feature_name],
                        feature_name,
                        stat,
                    )

                    # Store the aggregated value in the dictionary
                    self.aggregated_features[new_name] = aggregated_value

    def map_features_to_instance_space(self):
        pass

    def make_aggregated_feature_table(self, instance_name):
        """
        Create a 1-row table of all the features to allow comparison.
        """
        # Create a DataFrame with all features
        dat = pd.DataFrame(self.aggregated_features, index=[0])

        # Add problem name and number of dimensions
        dat.insert(0, "D", instance_name.split("d")[1])
        dat.insert(0, "Name", instance_name)

        return dat

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
