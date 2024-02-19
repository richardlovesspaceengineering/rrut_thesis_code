import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def remove_sd_cols(df, suffix="_std"):
    df = df[[col for col in df.columns if not col.endswith(suffix)]]
    return df


def get_df_from_filepath(filepath, give_sd):

    # Check if the files exist
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} does not exist.")

    features_df = pd.read_csv(filepath)

    if not give_sd:
        features_df = remove_sd_cols(features_df)

    return features_df


class FeaturesDashboard:
    def __init__(self, features_path, algo_perf_path):
        """
        Initialize the FeaturesDashboard with paths to the directories containing features.csv and algo_performance.csv.
        :param features_path: Path to the folder containing the features.csv file.
        :param algo_perf_path: Path to the folder containing the algo_performance.csv file.
        """
        self.features_path = features_path
        self.algo_perf_path = algo_perf_path
        self.overall_df = self.get_overall_df()

    def get_landscape_features_df(self, give_sd=True):
        features_filepath = os.path.join(self.features_path, "features.csv")
        return get_df_from_filepath(features_filepath, give_sd=give_sd)

    def get_algo_performance_df(self, give_sd=True):
        algo_perf_file_path = os.path.join(self.algo_perf_path, "algo_performance.csv")
        return get_df_from_filepath(algo_perf_file_path, give_sd=give_sd)

    def get_overall_df(self, give_sd=True):
        """
        Reads the features.csv and algo_performance.csv files from their respective directories
        and joins them into a single DataFrame, resolving any 'D' column duplication.
        :return: A pandas DataFrame containing the joined data with a single 'D' column.
        """

        # Read the CSV files into DataFrames
        features_df = self.get_landscape_features_df(give_sd=give_sd)
        algo_perf_df = self.get_algo_performance_df(give_sd=give_sd)

        # Join the DataFrames, specifying suffixes for overlapping column names other than the join key
        overall_df = pd.merge(
            features_df, algo_perf_df, on="Name", how="inner", suffixes=("", "_drop")
        )

        # Drop the redundant 'D' column from the right DataFrame (algo_performance.csv)
        # and any other unwanted duplicate columns that were suffixed with '_drop'
        overall_df.drop(
            [col for col in overall_df.columns if "drop" in col], axis=1, inplace=True
        )

        return overall_df

    def get_problem_features_df(self, problem_name, dim, analysis_type):
        """
        Reads a specific features.csv file based on the problem name, dimension, and analysis type.
        :param problem_name: The name of the problem.
        :param dim: The dimension of the problem.
        :param analysis_type: The type of analysis (e.g., "summary", "detailed").
        :return: A pandas DataFrame containing the data from the specific features.csv file.
        """
        # Constructing the file path based on problem name, dimension, and analysis type
        file_name = f"{problem_name}_d{dim}_{analysis_type}_features.csv"
        file_path = os.path.join(
            self.features_path, f"{problem_name}_d{dim}", file_name
        )

        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        return df

    def get_problem_algo_df(self, problem_name, dim):
        algo_perf_file_name = f"{problem_name}_d{dim}_algo.csv"
        algo_perf_file_path = os.path.join(
            self.algo_perf_path, f"{problem_name}_d{dim}", algo_perf_file_name
        )

        # Check if the files exist
        if not os.path.exists(algo_perf_file_path):
            raise FileNotFoundError(
                f"The algorithm performance file {algo_perf_file_path} does not exist."
            )
        # Read the CSV files into DataFrames
        algo_perf_df = pd.read_csv(algo_perf_file_path)

        return algo_perf_df

    @staticmethod
    def get_features_for_analysis_type(df, analysis_type):
        """
        Filters a DataFrame to return only columns related to a specified analysis type,
        but always includes 'D' and 'Name' columns.
        Raises an error if an invalid analysis type is given.
        :param df: The input DataFrame to filter.
        :param analysis_type: The type of analysis to filter columns by (must be "rw", "glob", or "aw").
        :return: A pandas DataFrame containing only the columns that include the analysis_type in their names,
                 plus 'D' and 'Name' columns.
        """
        # Validate analysis_type
        valid_analysis_types = ["rw", "glob", "aw"]
        if analysis_type not in valid_analysis_types:
            raise ValueError(
                f"Invalid analysis type '{analysis_type}'. Valid types are: {', '.join(valid_analysis_types)}"
            )

        # Construct the pattern and filter columns based on the analysis_type
        analysis_type_pattern = f"_{analysis_type}_"
        filtered_columns = [col for col in df.columns if analysis_type_pattern in col]

        # Ensure 'D' and 'Name' columns are included
        essential_columns = ["D", "Name"]
        for col in essential_columns:
            if col in df.columns and col not in filtered_columns:
                filtered_columns.insert(0, col)  # Prepend to keep order if needed

        df_filtered = df[filtered_columns]

        return df_filtered

    @staticmethod
    def get_features_for_suite(df, suite_name):
        # Filter rows based on the suite_name in the 'Name' column

        valid_suites = ["MW", "CTP", "DASCMOP", "DCDTLZ", "CDTLZ"]
        if suite_name not in valid_suites:
            raise ValueError(
                f"Invalid suite '{suite_name}'. Valid types are: {', '.join(valid_suites)}"
            )

        df_filtered = df[df["Name"].str.contains(suite_name, na=False)]
        return df_filtered

    def plot_feature_across_suites(self, feature_name, suite_names):
        """
        Generates a 1xN grid of violin plots for a specified feature across different benchmark suites.
        :param feature_name: The name of the feature to plot. Can be a landscape feature or algo performance.
        :param suite_names: A list of benchmark suite names.
        """
        plt.figure(figsize=(10, 6))

        feature_name = feature_name + "_mean"

        for i, suite_name in enumerate(suite_names, start=1):
            df_filtered = self.get_features_for_suite(self.overall_df, suite_name)
            if feature_name in df_filtered.columns:
                plt.subplot(1, len(suite_names), i)
                sns.violinplot(y=df_filtered[feature_name])
                plt.title(suite_name)
                plt.ylabel(
                    feature_name if i == 1 else ""
                )  # Only add y-label to the first plot
                plt.xlabel("")

        plt.tight_layout()
        plt.show()

    def plot_problem_features(self, problem_name, dim, analysis_type, features=None):
        """
        Creates a grid of violin plots for specified features for a specific problem instance.
        :param problem_name: Name of the problem.
        :param dim: Dimension of the problem.
        :param analysis_type: Type of analysis (e.g., "summary", "detailed").
        :param features: Optional list of features to plot. If None, plots all columns.
        """

        df = self.get_problem_features_df(problem_name, dim, analysis_type)

        # If no specific features are provided, use all numeric columns
        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()

        n_features = len(features)

        # Calculate grid size for plotting
        grid_size = int(np.ceil(np.sqrt(n_features)))

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        fig.suptitle(
            f"Violin Plots for {problem_name} (Dimension: {dim}, Analysis: {analysis_type})",
            fontsize=16,
        )

        # Flatten axes array for easy indexing
        axes = axes.flatten()

        for i, feature in enumerate(features):
            if i < n_features:
                sns.violinplot(y=df[feature], ax=axes[i])
                axes[i].set_title(feature)
            else:
                # Hide unused subplots
                axes[i].axis("off")

        plt.tight_layout(
            rect=[0, 0.03, 1, 0.95]
        )  # Adjust layout to make room for the main title
        plt.show()

    def plot_problem_algo_performance(self, problem_name, dim, algorithms=None):
        """
        Creates a row of violin plots for algorithm performance metrics for a specific problem instance.
        :param problem_name: Name of the problem.
        :param dim: Dimension of the problem.
        :param algorithms: Optional list of algorithms to plot. If None, plots all algorithms.
        """
        df = self.get_problem_algo_df(problem_name, dim)

        # If no specific algorithms are provided, plot for all available in the DataFrame
        if algorithms is None:
            algorithms = df.columns.tolist()
        else:
            # Filter only the columns that match the specified algorithms
            df = df[algorithms]

        n_algorithms = len(algorithms)

        # Create a 1xn grid of plots
        fig, axes = plt.subplots(1, n_algorithms, figsize=(5 * n_algorithms, 5))
        fig.suptitle(
            f"Algorithm Performance for {problem_name} (Dimension: {dim})", fontsize=16
        )

        # In case there's only one algorithm, ensure axes is iterable
        if n_algorithms == 1:
            axes = [axes]

        for i, algorithm in enumerate(algorithms):
            sns.violinplot(y=df[algorithm], ax=axes[i])
            axes[i].set_title(algorithm)

        plt.tight_layout(
            rect=[0, 0.03, 1, 0.95]
        )  # Adjust layout to make room for the main title
        plt.show()
