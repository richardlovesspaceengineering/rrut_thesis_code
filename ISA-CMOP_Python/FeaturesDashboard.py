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
    def __init__(self, features_path_list, new_save_path):
        """
        Initialize the FeaturesDashboard with paths to the directories containing features.csv and algo_performance.csv.
        :param features_path_list: List of paths to the folder containing the features.csv file. Assuming these are stored in instance_results.
        """
        self.features_path_list = features_path_list
        self.new_save_path = new_save_path
        self.features_df = self.get_landscape_features_df(give_sd=True)

    def get_landscape_features_df(self, give_sd):
        """
        Collates features.csv files from each directory in features_path_list, adding an additional column to indicate the source directory.
        """
        # Initialize an empty list to store dataframes
        df_list = []

        # Loop through each path in the features path list
        for folder_path in self.features_path_list:
            # Construct the filepath to the features.csv file
            features_filepath = os.path.join(
                "instance_results", folder_path, "features.csv"
            )

            temp_df = get_df_from_filepath(features_filepath, give_sd)

            # Extract the folder name from the folder path to use as the Date identifier
            folder_name = os.path.basename(folder_path)

            # Add the folder name as a new column in the dataframe
            temp_df["Date"] = folder_name

            # Ensure 'Date' column is the second column
            cols = temp_df.columns.tolist()
            # Then, move 'Date' to the second position (index 1) assuming the first column is something you want to keep at the start
            cols.insert(1, cols.pop(cols.index("Date")))
            # Reindex the dataframe with the new column order
            temp_df = temp_df[cols]

            # Append the dataframe to the list
            df_list.append(temp_df)

        # Concatenate all dataframes in the list
        if df_list:
            overall_df = pd.concat(df_list, ignore_index=True)
        else:
            raise FileNotFoundError(
                "No features.csv files found in the provided directories."
            )

        return overall_df

    def save_features_collated_csv(self):
        """
        Saves the collated features dataframe as features_collated.csv in the specified new_save_path.
        """
        # Ensure the save path directory exists
        os.makedirs(self.new_save_path, exist_ok=True)

        # Construct the full path to the output csv file
        output_filepath = os.path.join(self.new_save_path, "features_collated.csv")

        # Save the dataframe to csv
        self.features_df.to_csv(output_filepath, index=False)

        print(f"Features collated csv saved to {output_filepath}")

    def get_problem_features_df(self, problem_name, dim, analysis_type, path):
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
