import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shutil
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE
from scipy.stats import zscore
import simpsom as sps


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
    def __init__(self, results_dict, new_save_path):
        """
        Initialize the FeaturesDashboard with a dictionary mapping problem names to lists of result storage folders.
        :param results_dict: Dictionary where keys are problem names and values are lists of storage folder names.
        :param new_save_path: Path to the directory where new files will be saved.
        """
        self.results_dict = results_dict
        self.new_save_path = new_save_path
        # Initialize features_path_list as an empty list
        self.features_path_list = []
        # Base directory where instance results are stored
        self.base_results_dir = "instance_results"

        self.suites_problems_dict = {
            "MW": [f"MW{x}" for x in range(1, 15, 1)],
            "CTP": [f"CTP{x}" for x in range(1, 9, 1)],
            "CF": [f"CF{x}" for x in range(1, 11, 1)],
            "DASCMOP": [f"DASCMOP{x}" for x in range(1, 10, 1)],
            "LIRCMOP": [f"LIRCMOP{x}" for x in range(1, 15, 1)],
            "RWMOP": [f"RWMOP{x}" for x in range(1, 51, 1)],
            "DCDTLZ": [
                "DC1DTLZ1",
                "DC1DTLZ3",
                "DC2DTLZ1",
                "DC2DTLZ3",
                "DC3DTLZ1",
                "DC3DTLZ3",
            ],
            "CDTLZ": [
                "C1DTLZ1",
                "C1DTLZ3",
                "C2DTLZ2",
                "DC2DTLZ3",
                "C3DTLZ1",
                "C3DTLZ4",
            ],
            "MODAct": [
                "CS1",
                "CS2",
                "CS3",
                "CS4",
                "CT1",
                "CT2",
                "CT3",
                "CT4",
                "CTSE1",
                "CTSE2",
                "CTSE3",
                "CTSE4",
                "CTSEI1",
                "CTSEI2",
                "CTSEI3",
                "CTSEI4",
            ],
        }

        self.analytical_problems = [
            f
            for f in self.suites_problems_dict.keys()
            if f not in ["MODAct", "RWMOP", "XA"]
        ]

        # Process the results_dict to populate features_path_list
        missing_folders = []
        for folders in results_dict.values():
            if folders is not None:
                for folder in folders:
                    full_path = os.path.join(self.base_results_dir, folder)
                    # Check if the directory exists before appending
                    if os.path.isdir(full_path):
                        if full_path not in self.features_path_list:
                            self.features_path_list.append(full_path)
                    else:

                        # Ensures warning only gets printed once.
                        if folder not in missing_folders:
                            missing_folders.append(folder)
                            print(f"Warning: The directory {full_path} does not exist.")

        # Collate all results into one folder.
        self.wipe_features_directory()
        self.copy_directory_contents()
        self.features_df = self.get_landscape_features_df(give_sd=False)

        # Models/results objects.
        self.pca = None
        self.features_to_ignore = []

    @staticmethod
    def get_numerical_data_from_df(df, give_sd=False):

        # Check if features_to_ignore is provided, if not, initialize it as an empty list

        # Filter out the specific columns you don't want to keep
        filtered_columns = [
            col for col in df.columns if col not in ["D", "Suite", "Date", "Name"]
        ]
        if not give_sd:
            filtered_columns = [
                col for col in filtered_columns if not col.endswith("_std")
            ]

        return df[filtered_columns]

    @staticmethod
    def custom_drop_na(df):

        # Calculate the initial length of the dataframe
        initial_length = len(df)

        # Find rows with NaNs and store their 'Name'
        rows_with_nans = df[df.isnull().any(axis=1)]["Name"]

        # Drop rows with NaNs
        df_clean = df.dropna()

        # Calculate the number of dropped rows
        num_dropped_rows = initial_length - len(df_clean)

        # Display the names of the dropped rows, number of dropped rows, and initial length
        print("Dropped instances:")
        print(rows_with_nans.to_list())
        print(f"Initial length of the dataframe: {initial_length}")
        print(f"Total number of dropped instances: {num_dropped_rows}")

        return df_clean

    def replace_nan_with_value(self, column, value):
        """
        Replace all NaN values in a specific column of self.features_df with a user-specified value.

        :param column: The column in which to replace NaN values.
        :param value: The value to replace NaNs with.
        """
        if column not in self.features_df.columns:
            print(f"Column '{column}' does not exist in the DataFrame.")
            return

        if self.features_df[column].isna().sum() == 0:
            print(f"There are no NaN values in column '{column}'.")
        else:
            self.features_df[column].fillna(value, inplace=True)
            print(f"NaN values in column '{column}' have been replaced with '{value}'.")

    def compare_results_dict_to_df(self):
        """
        This method checks if the keys of the dictionary exist in the specified column of the dataframe.

        :param keys_dict: Dictionary whose keys will be checked.
        :param dataframe: DataFrame where the column is located.
        :param column_name: The name of the column to search for the keys.
        :return: A dictionary with the keys and a boolean value indicating if the key was found in the column.
        """

        # Initialize a result dictionary
        result = {}

        # Iterate through the dictionary keys
        for instance in self.results_dict.keys():
            # Check if the key is in the self.features_df column
            found = instance in self.features_df["Name"].values
            result[instance] = found

            # If the key is not found, print it
            if not found:
                print(f"Missing data for {instance}")

        return result

    def plot_missingness(self, show_only_nans=False, show_ignored_features=True):
        # Create a DataFrame indicating where NaNs are located (True for NaN, False for non-NaN)

        if not show_ignored_features:
            df = self.ignore_specified_features(self.features_df)
        else:
            df = self.features_df

        missingness = FeaturesDashboard.get_numerical_data_from_df(df).isnull()

        row_has_nan = missingness.any(axis=1)

        # Filter columns to show only those that contain at least one NaN value if the flag is set
        if show_only_nans:
            missingness = missingness.loc[row_has_nan, missingness.any(axis=0)]
            ytick_lab = self.features_df.loc[row_has_nan, "Name"].values
        else:
            ytick_lab = self.features_df["Name"].values

        # Plotting the heatmap
        plt.figure(figsize=(36, 24))
        ax = sns.heatmap(
            missingness,
            cbar=False,
            yticklabels=ytick_lab,
            cmap="coolwarm",
        )
        if show_only_nans:
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=24, rotation=90)
        plt.title("Missingness Plot")
        plt.xlabel("Features")
        plt.ylabel("Observations")
        plt.show()

    def get_feature_names(self, shortened=True):

        df = FeaturesDashboard.get_numerical_data_from_df(self.features_df)

        # Remove "_mean" from the remaining column names
        if shortened:
            names = [col.replace("_mean", "") for col in df.columns]
        else:
            names = [col for col in df.columns]

        return names

    def get_suite_for_problem(self, problem_name):
        """
        Determine the suite for a given problem name based on the suites_problems_dict,
        considering only the part of the problem name before an underscore.
        """
        problem_prefix = problem_name.split("_")[
            0
        ]  # Get the part before the first underscore
        for suite, problems in self.suites_problems_dict.items():
            # Check if the problem prefix matches any problem in the suite
            if any(problem.startswith(problem_prefix) for problem in problems):
                return suite
        return "Unknown"  # or another placeholder for problems not found in any suite

    def get_landscape_features_df(self, give_sd):
        """
        Collates features_{timestamp}.csv files that have been copied to self.new_save_path,
        adding additional columns to indicate the source timestamp and the corresponding suite.
        """
        df_list = []

        for filename in os.listdir(self.new_save_path):
            if (
                filename.startswith("features")
                and filename.endswith(".csv")
                and "collated" not in filename
            ):
                file_path = os.path.join(self.new_save_path, filename)
                temp_df = get_df_from_filepath(file_path, give_sd)

                timestamp = filename.replace("features_", "").replace(".csv", "")
                temp_df["Date"] = timestamp

                # Add suite column based on problem name
                temp_df["Suite"] = temp_df["Name"].apply(self.get_suite_for_problem)

                # Filter rows based on 'Name' and 'Date'. For each problem, we only want to keep results from the dates given in results_dict.
                temp_df = temp_df[
                    temp_df.apply(
                        lambda row: row["Name"] in self.results_dict
                        and row["Date"] in self.results_dict[row["Name"]],
                        axis=1,
                    )
                ]

                if not temp_df.empty:
                    cols = temp_df.columns.tolist()
                    cols.insert(
                        1, cols.pop(cols.index("Date"))
                    )  # Ensure 'Date' is second
                    cols.insert(
                        2, cols.pop(cols.index("Suite"))
                    )  # Ensure 'Suite' is third
                    temp_df = temp_df[cols]

                    df_list.append(temp_df)

        if df_list:
            overall_df = pd.concat(df_list, ignore_index=True)
        else:
            raise FileNotFoundError(
                f"No features_{timestamp}.csv files found in the new save path."
            )

        return overall_df

    def replace_outliers_with_nan(self):
        """
        Replaces values in the numerical columns of df grouped by the "Suite" column with np.nan
        if their orders of magnitude are outside the 1.75 * IQR range of the orders of magnitude
        of the rest of the points in the grouped column, and logs the column name, Suite, Name value, and the outlier value.
        """
        df = self.get_landscape_features_df(give_sd=False)
        outlier_log = []  # Initialize a list to store the log information

        # Iterate through each group determined by 'Suite'
        for suite, group_df in df.groupby("Suite"):
            for column in group_df.select_dtypes(include=[np.number]).columns:

                if column.endswith("_std"):  # Skip columns ending with "_std"
                    continue

                # Calculate the orders of magnitude for the column, excluding zero to avoid log(0)
                valid_values = group_df[group_df[column] != 0][column].abs()
                if not valid_values.empty:
                    orders_of_magnitude = np.log10(valid_values)

                    if orders_of_magnitude.max() < 2:
                        continue

                    # Calculate IQR for the orders of magnitude
                    Q1, Q3 = np.percentile(orders_of_magnitude, [25, 75])
                    IQR = Q3 - Q1

                    # Define the bounds based on the IQR
                    upper_bound = Q3 + 3 * IQR

                    # Iterate over each row in the group to check for outliers and log necessary information
                    for index, row in group_df.iterrows():
                        value = row[column]
                        if value != 0:
                            order_of_magnitude = np.log10(abs(value))
                            if order_of_magnitude > upper_bound:
                                # Log the column name, the Suite value, the value in the "Name" column, and the outlier value
                                outlier_log.append((column, suite, row["Name"], value))
                                # Replace the outlier with np.nan in the original dataframe
                                df.at[index, column] = np.nan

        # Optionally, print the log information
        for log_entry in outlier_log:
            print(
                f"Column: {log_entry[0]}, Suite: {log_entry[1]}, Name: {log_entry[2]}, Outlier Value: {log_entry[3]}"
            )

        # Save back
        self.features_df = df

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

    def copy_directory_contents(self):
        """
        Copies the contents of directories from self.features_path_list into self.new_save_path.
        Each item (file or directory) will have a timestamp derived from its source directory appended to its name to ensure uniqueness.
        """
        # Ensure the new_save_path exists
        os.makedirs(self.new_save_path, exist_ok=True)

        for src_dir in self.features_path_list:
            if os.path.exists(src_dir):
                unique_identifier = os.path.basename(
                    src_dir
                )  # Extract a unique identifier from the directory name

                self._copy_items(src_dir, self.new_save_path, unique_identifier)

            else:
                print(f"Warning: {src_dir} is not a directory or does not exist.")

    def _copy_items(self, src, dst, unique_identifier):
        """
        Recursively copies items from src to dst, appending unique_identifier to each name.
        """
        for item in os.listdir(src):
            src_item = os.path.join(src, item)

            if os.path.isdir(src_item):

                # For directories, append the unique identifier to the directory name and create it in the destination
                new_dir_name = f"{item}_{unique_identifier}"
                new_dir_path = os.path.join(dst, new_dir_name)
                os.makedirs(new_dir_path, exist_ok=True)
                # Recursively copy the directory contents
                self._copy_items(src_item, new_dir_path, unique_identifier)
            else:

                # For files, append the unique identifier to the file name before copying
                file_base, file_extension = os.path.splitext(item)

                new_filename = f"{file_base}_{unique_identifier}{file_extension}"

                dst_item = os.path.join(dst, new_filename)
                shutil.copy2(src_item, dst_item)
                print(f"Copied {src_item} to {dst_item}")

    def wipe_features_directory(self):
        """
        Wipes the self.new_save_path directory clean by removing all its contents.
        """
        # Check if the directory exists
        if os.path.exists(self.new_save_path) and os.path.isdir(self.new_save_path):
            # Iterate over all the files and directories within self.new_save_path
            for filename in os.listdir(self.new_save_path):
                file_path = os.path.join(self.new_save_path, filename)
                try:
                    # If it's a file, remove it directly
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    # If it's a directory, remove the entire directory tree
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
        else:
            print(
                f"The directory {self.new_save_path} does not exist or is not a directory."
            )

    def get_problem_features_samples_df(
        self, problem_name, dim, analysis_type, timestamp=None
    ):
        """
        Reads a specific features.csv file based on the problem name, dimension, analysis type, and an optional timestamp.
        :param problem_name: The name of the problem.
        :param dim: The dimension of the problem.
        :param analysis_type: The type of analysis (e.g., "summary", "detailed").
        :param timestamp: Optional. Specifies the exact result folder if there are multiple possibilities and a timestamp is required.
        :return: A pandas DataFrame containing the data from the specific features.csv file.
        """
        problem_key = f"{problem_name}_d{dim}"
        folders = self.results_dict.get(problem_key)

        # Check for multiple folders and no timestamp specified
        if folders and len(folders) > 1 and timestamp is None:
            raise ValueError(
                f"Multiple result directories exist for {problem_key}. Please specify the result directory using a timestamp from: {folders}."
            )

        if timestamp:
            # Use the specified timestamp to find the folder
            folder_name = f"{problem_name}_d{dim}_{timestamp}"
        else:
            # Use the first (or only) folder path available, if only one exists
            folder_name = f"{problem_name}_d{dim}_{folders[0]}" if folders else None

        if not folder_name:
            raise ValueError(
                f"No results directory specified or found for {problem_key}."
            )

        folder_path = os.path.join(self.new_save_path, folder_name)

        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"The directory {folder_path} does not exist.")

        # Constructing the file path
        if timestamp:
            # Use the specified timestamp to find the folder
            file_name = (
                f"{problem_name}_d{dim}_{analysis_type}_features_{timestamp}.csv"
            )
        else:
            # Use the first (or only) folder path available, if only one exists
            file_name = (
                f"{problem_name}_d{dim}_{analysis_type}_features_{folders[0]}.csv"
                if folders
                else None
            )

        file_path = os.path.join(folder_path, file_name)

        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        return df

    def append_to_features_to_ignore(self, features_to_ignore):

        # Check if features_to_ignore is neither None nor an empty list
        if features_to_ignore:
            # Convert to list if it's a single element
            if not isinstance(features_to_ignore, list):
                features_to_ignore = [features_to_ignore]

            # Append each feature only if it's not already in the list
            for feature in features_to_ignore:
                if feature not in self.features_to_ignore:
                    self.features_to_ignore.append(feature)

    def ignore_specified_features(self, df):
        if self.features_to_ignore:
            return df[[col for col in df.columns if col not in self.features_to_ignore]]
        else:
            print("No features to ignore have been set as attributes of the instance.")
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
        essential_columns = ["Suite", "D", "Name"]
        for col in essential_columns:
            if col in df.columns and col not in filtered_columns:
                filtered_columns.insert(0, col)  # Prepend to keep order if needed

        df_filtered = df[filtered_columns]

        return df_filtered

    def compute_global_maxmin_for_plot(self, df_filtered, feature_name_with_mean):
        """
        Determines the global y-axis limits across all selected suites and dimensions for the given feature.
        :param df_filtered: The DataFrame that has already been filtered by suites and dimensions.
        :param feature_name_with_mean: The name of the feature (including '_mean' suffix) for which to compute the limits.
        :return: Tuple containing the global minimum and maximum values with a buffer applied.
        """

        if feature_name_with_mean not in df_filtered.columns:
            raise ValueError(
                f"The feature {feature_name_with_mean} does not exist in the DataFrame."
            )

        # Determine the global y-axis limits across the filtered DataFrame
        global_min = df_filtered[feature_name_with_mean].min()
        global_max = df_filtered[feature_name_with_mean].max()

        # Add a buffer to the global y-axis limits
        buffer_percent = 0.05  # For example, 5% buffer
        buffer = (global_max - global_min) * buffer_percent
        global_min -= buffer
        global_max += buffer

        return global_min, global_max

    def plot_feature_across_suites(
        self, feature_name, suite_names=None, dims=None, show_plot=True
    ):
        """
        Generates a 1xN grid of violin plots for a specified feature across different suites, with points overlaying the violin plots.
        Each violin plot represents a different suite, with distinct colors used for each dimension.
        :param feature_name: The name of the feature to plot. Can be a landscape feature or algo performance.
        :param dims: A list of dimensions to plot.
        :param suite_names: Optional. A list of benchmark suite names to filter the features by.
        """
        plt.figure(figsize=(15, 6))
        feature_name_with_mean = feature_name + "_mean"

        # Define colors for different suites.
        dim_colors = {
            5: "#1f77b4",
            10: "#ff7f0e",
            20: "#2ca02c",
            30: "#d62728",
        }
        marker_type = "o"

        if not suite_names:
            suite_names = self.suites_problems_dict.keys()

        if not dims:
            dims = [5, 10, 20, 30]  # Default dimensions if none provided

        # Use the filtering method to get the filtered DataFrame based on suite names and dimensions
        df_filtered = self.filter_df_by_suite_and_dim(
            self.features_df, suite_names=suite_names, dims=dims
        )

        global_min, global_max = self.compute_global_maxmin_for_plot(
            df_filtered, feature_name_with_mean
        )

        for i, suite_name in enumerate(suite_names, start=1):
            ax = plt.subplot(1, len(suite_names), i)
            df_suite_filtered = df_filtered[df_filtered["Suite"] == suite_name]

            # Plot a single violin plot for the combined data
            if not df_suite_filtered.empty:
                sns.violinplot(
                    y=df_suite_filtered[feature_name_with_mean],
                    color="lightgrey",
                )

                # Overlay sample size.
                num_points = df_suite_filtered.shape[0]
                ax.text(
                    0.95,
                    0.95,
                    f"n={num_points}",  # Position inside the subplot, top right corner
                    verticalalignment="top",
                    horizontalalignment="right",
                    transform=ax.transAxes,  # Coordinate system relative to the axes
                    color="black",
                    fontsize=10,
                )

                # Overlay points for each suite.
                for dim in dims:
                    df_suite_dim_filtered = df_suite_filtered[
                        df_suite_filtered["D"] == dim
                    ]
                    if not df_suite_dim_filtered.empty:
                        sns.stripplot(
                            y=df_suite_dim_filtered[feature_name_with_mean],
                            color=dim_colors.get(
                                dim, "gray"
                            ),  # Default to gray if suite not in colors
                            marker=marker_type,
                            label=f"{dim}D",
                            size=5,
                            alpha=0.5,
                            jitter=True,
                        )

            plt.title(f"{suite_name}")
            plt.ylabel(feature_name_with_mean if i == 1 else "")
            plt.xlabel("")

            if i == 1:
                plt.legend()
            else:
                ax.legend().remove()

            ax.set_ylim(global_min, global_max)

        plt.suptitle(feature_name_with_mean)
        plt.tight_layout()
        if show_plot:
            plt.show()

    def get_correlation_matrix(
        self,
        analysis_type=None,
        ignore_features=True,
        dims=None,
        suite_names=None,
        method="pearson",
        min_corr_magnitude=0,
    ):

        if ignore_features:
            df = self.ignore_specified_features(self.features_df)
        else:
            df = self.features_df

        df = self.custom_drop_na(
            self.filter_df_by_suite_and_dim(
                df,
                suite_names=suite_names,
                dims=dims,
            )
        )

        if analysis_type:
            df = FeaturesDashboard.get_features_for_analysis_type(df, analysis_type)

        if method not in ["pearson", "spearman"]:
            raise ValueError("Method must be either 'pearson' or 'spearman'")

        numeric_df = FeaturesDashboard.get_numerical_data_from_df(df)

        # Calculate the correlation matrix
        correlation_matrix = numeric_df.corr(method=method)

        # Remove the word mean from each string.
        correlation_matrix.columns = correlation_matrix.columns.str.replace("_mean", "")
        correlation_matrix.index = correlation_matrix.index.str.replace("_mean", "")

        return correlation_matrix[correlation_matrix.abs() > min_corr_magnitude]

    def get_top_correlation_pairs(
        self,
        min_corr_magnitude,
        filtered_err_pct=None,
        ignore_features=True,
        analysis_type=None,
        dims=None,
        suite_names=None,
        method="pearson",
    ):
        """
        Returns the top n correlation pairs from the correlation matrix, sorted by absolute correlation value.

        :param n: Number of top correlation pairs to return.
        """
        correlation_matrix = self.get_correlation_matrix(
            analysis_type,
            ignore_features,
            dims,
            suite_names,
            method,
            min_corr_magnitude,
        )

        au_corr = correlation_matrix.unstack()
        # Create a set to hold pairs to drop (diagonal and lower triangular)
        pairs_to_drop = set()
        cols = correlation_matrix.columns
        for i in range(correlation_matrix.shape[1]):
            for j in range(i + 1):
                pairs_to_drop.add((cols[i], cols[j]))
        # Drop redundant pairs and sort
        au_corr = (
            au_corr.drop(labels=pairs_to_drop).sort_values(ascending=False).dropna()
        )

        # Convert to DataFrame for a nicer display
        au_corr_df = au_corr.reset_index()
        au_corr_df.columns = ["Variable 1", "Variable 2", "Overall"]

        # Get a list of all suites if not provided
        if not suite_names:
            suite_names = self.suites_problems_dict.keys()

        # Iterate over each suite and calculate correlation for top pairs
        for suite in suite_names:
            suite_corr_matrix = self.get_correlation_matrix(
                analysis_type,
                ignore_features,
                dims,
                [suite],  # Pass the current suite name
                method,
                min_corr_magnitude=0,
            )
            # Add a column for each suite's correlation
            au_corr_df[suite] = au_corr_df.apply(
                lambda row: suite_corr_matrix.loc[row["Variable 1"], row["Variable 2"]],
                axis=1,
            )

        # We only want to consider correlations that are consistent across the suites.
        if filtered_err_pct:
            # Define a filter function for rows
            def filter_row(row):
                overall_corr = row["Overall"]
                # Compare suite-specific correlation with overall correlation
                return all(
                    abs(overall_corr - row[suite]) / abs(overall_corr) * 100
                    <= filtered_err_pct
                    for suite in suite_names
                )

            # Apply the filter function to each row
            au_corr_df = au_corr_df[au_corr_df.apply(filter_row, axis=1)]

        return au_corr_df

    def plot_top_correlation_heatmap(
        self, min_corr_magnitude, show_only_where_suites=False, filtered_err_pct=None
    ):
        """
        Plots a heatmap where rows are Variable 1/Variable 2 pairs and columns are 'Overall' and each suite.

        :param au_corr_df: A pandas DataFrame containing the correlation data,
                        with Variable 1, Variable 2, Overall Correlation, and Suite Correlations.
        """

        if not show_only_where_suites:
            au_corr_df = self.get_top_correlation_pairs(
                min_corr_magnitude=min_corr_magnitude,
                ignore_features=True,
                filtered_err_pct=filtered_err_pct,
            )
        else:
            au_corr_df = self.get_filtered_correlation_pairs

        # Creating a new column that combines Variable 1 and Variable 2 for labeling purposes
        au_corr_df["Variable Pair"] = au_corr_df[["Variable 1", "Variable 2"]].apply(
            lambda x: " / ".join(x), axis=1
        )

        # Setting this new column as the index
        au_corr_df = au_corr_df.set_index("Variable Pair")

        # Dropping the original Variable 1 and Variable 2 columns
        au_corr_df = au_corr_df.drop(["Variable 1", "Variable 2"], axis=1)

        # Transpose the dataframe for better visualization, if necessary
        # au_corr_df = au_corr_df.T

        # Set the size of the plot
        if filtered_err_pct:
            plt.figure(figsize=(8, 6))
        elif min_corr_magnitude >= 0.95:
            plt.figure(figsize=(18, 12))
        else:
            plt.figure(figsize=(36, 24))

        # Plot the heatmap
        ax = sns.heatmap(
            au_corr_df,
            annot=False,  # Annotate the cells with correlation values
            cmap="coolwarm",  # Color map to use for the heatmap
            vmin=-1,  # Setting the scale minimum to -1
            vmax=1,  # Setting the scale maximum to 1
            square=True,  # Ensuring each cell is square-shaped
            linewidths=0.5,
        )  # Setting the width of the lines that will divide each cell

        # Adjust the plot
        plt.xticks(
            rotation=45, ha="right", fontsize=10
        )  # Rotate the x-axis labels for better readability
        plt.yticks(
            rotation=0, fontsize=10
        )  # Ensure the y-axis labels are horizontal for readability
        plt.tight_layout()  # Adjust the layout to make sure there's no overlap

        # Show the plot
        plt.show()

    def count_feature_occurrences_in_top_corr(self, au_corr_df):
        """
        Count how many times each feature name appears in the 'Variable 1' and 'Variable 2' columns
        and return a dictionary with feature names as keys and counts as values.

        :param au_corr_df: A pandas DataFrame containing the correlation data,
                        with 'Variable 1' and 'Variable 2' columns.
        :return: A dictionary with feature names as keys and their counts as values.
        """
        # Combine counts from 'Variable 1' and 'Variable 2'
        var1_counts = au_corr_df["Variable 1"].value_counts()
        var2_counts = au_corr_df["Variable 2"].value_counts()

        # Combine the counts from both columns
        combined_counts = (
            var1_counts.add(var2_counts, fill_value=0).astype(int).to_dict()
        )

        return combined_counts

    def plot_correlation_heatmap(
        self,
        analysis_type=None,
        ignore_features=True,
        dims=None,
        suite_names=None,
        method="pearson",
        min_corr_magnitude=0,
        show_plot=True,
    ):
        """
        Generates a heatmap of the correlation matrix for the numeric columns of the provided DataFrame.
        :param df: The DataFrame for which the correlation matrix heatmap is to be generated.
        """

        correlation_matrix = self.get_correlation_matrix(
            analysis_type,
            ignore_features,
            dims,
            suite_names,
            method,
            min_corr_magnitude,
        )

        # Generate a heatmap
        plt.figure(figsize=(40, 32))  # You can adjust the size as needed
        ax = sns.heatmap(
            correlation_matrix,
            annot=False,
            cmap="coolwarm",
            square=True,
            vmin=-1,
            vmax=1,
        )
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=12, va="center")
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, ha="center")

        if not dims:
            dims = "All"
        if not suite_names:
            suite_names = "All"
        if not analysis_type:
            analysis_type = "All"

        ax.set_title(
            f"Correlation heatmap for dimensions: {dims}, suites: {suite_names}, analysis type: {analysis_type}"
        )
        if show_plot:
            plt.show()

    def plot_feature_across_dims(
        self, feature_name, dims=None, suite_names=None, show_plot=True
    ):
        """
        Generates a 1xN grid of violin plots for a specified feature across different dimensions, with points overlaying the violin plots.
        Each violin plot represents a different dimension, with distinct colors used for each dimension.
        :param feature_name: The name of the feature to plot. Can be a landscape feature or algo performance.
        :param dims: A list of dimensions to plot.
        :param suite_names: Optional. A list of benchmark suite names to filter the features by.
        """
        plt.figure(figsize=(15, 6))
        feature_name_with_mean = feature_name + "_mean"

        # Define colors for different suites.
        suite_colors = {
            "MW": "#1f77b4",
            "CTP": "#ff7f0e",
            "DASCMOP": "#2ca02c",
            "DCDTLZ": "#d62728",
            "CDTLZ": "#e62728",
        }
        marker_type = "o"

        if not suite_names:
            suite_names = self.suites_problems_dict.keys()

        if not dims:
            dims = [5, 10, 20, 30]  # Default dimensions if none provided

        # Use the filtering method to get the filtered DataFrame based on suite names and dimensions
        df = self.features_df
        df_filtered = self.filter_df_by_suite_and_dim(
            df, suite_names=suite_names, dims=dims
        )

        global_min, global_max = self.compute_global_maxmin_for_plot(
            df_filtered, feature_name_with_mean
        )

        for i, dim in enumerate(dims, start=1):
            ax = plt.subplot(1, len(dims), i)
            df_dim_filtered = df_filtered[df_filtered["D"] == dim]

            # Plot a single violin plot for the combined data
            if not df_dim_filtered.empty:
                sns.violinplot(
                    y=df_dim_filtered[feature_name_with_mean],
                    color="lightgrey",
                )
                # Overlay sample size.
                num_points = df_dim_filtered.shape[0]
                ax.text(
                    0.95,
                    0.95,
                    f"n={num_points}",  # Position inside the subplot, top right corner
                    verticalalignment="top",
                    horizontalalignment="right",
                    transform=ax.transAxes,  # Coordinate system relative to the axes
                    color="black",
                    fontsize=10,
                )

                # Overlay points for each suite.
                for suite_name in suite_names:
                    df_suite_dim_filtered = df_dim_filtered[
                        df_dim_filtered["Suite"] == suite_name
                    ]
                    if not df_suite_dim_filtered.empty:
                        sns.stripplot(
                            y=df_suite_dim_filtered[feature_name_with_mean],
                            color=suite_colors.get(
                                suite_name, "gray"
                            ),  # Default to gray if suite not in colors
                            marker=marker_type,
                            label=suite_name,
                            size=5,
                            alpha=0.5,
                            jitter=True,
                        )

            plt.title(f"{dim}D")
            plt.ylabel(feature_name_with_mean if i == 1 else "")
            plt.xlabel("")

            if i == 1:
                plt.legend(title="Suite")
            else:
                ax.legend().remove()

            ax.set_ylim(global_min, global_max)

        plt.tight_layout()
        if show_plot:
            plt.show()

    def plot_multiple_features_across_suites(
        self, feature_names, suite_names=None, dims=None
    ):
        """
        Generates a series of violin plots for multiple features across different suites, saving each plot to a separate page in a PDF file.
        :param feature_names: A list of feature names to plot.
        :param suite_names: Optional. A list of benchmark suite names to filter the features by.
        :param dims: Optional. A list of dimensions to plot.
        """
        pdf_path = os.path.join(self.new_save_path, "feature_plots_by_suite.pdf")
        with PdfPages(pdf_path) as pdf:
            for feature_name in feature_names:
                self.plot_feature_across_suites(
                    feature_name, suite_names, dims, show_plot=False
                )
                pdf.savefig()  # saves the current figure into a pdf page
                plt.close()  # close the figure to prevent it from being displayed

    def generate_features_results_pdf_across_suites(self):
        df_cols = self.get_landscape_features_df(give_sd=False).columns

        # Remove _mean from each.
        feature_names = [
            col[:-5] for col in df_cols if col not in ["Name", "Date", "Suite", "D"]
        ]
        self.plot_multiple_features_across_suites(feature_names)

    @staticmethod
    def filter_df_by_suite_and_dim(df, suite_names=None, dims=None):
        """
        Filters the features DataFrame based on specified suite names and dimensions.
        :param suite_names: List of suite names to filter by. If None, no suite-based filtering is applied.
        :param dims: List of dimensions to filter by. If None, no dimension-based filtering is applied.
        :return: A filtered DataFrame based on the specified suite names and dimensions.
        """
        # Start with the full DataFrame
        df_filtered = df.copy()

        # Filter by suite if suite_names is specified
        if suite_names:
            df_filtered = df_filtered[df_filtered["Suite"].isin(suite_names)]

        # Filter by dimension if dims is specified
        if dims:
            df_filtered = df_filtered[df_filtered["D"].isin(dims)]

        return df_filtered

    def plot_features_comparison(
        self, feature_x, feature_y, color_by="dimension", suite_names=None, dims=None
    ):
        """
        Plots two features against each other, with point colors corresponding to either dimension or suite.
        Allows specifying suite names and dimensions for slicing the data before plotting.
        """

        feature_x = feature_x + "_mean"
        feature_y = feature_y + "_mean"

        if not suite_names:
            suite_names = self.suites_problems_dict.keys()

        if not dims:
            dims = [5, 10, 20, 30]

        filtered_df = self.filter_df_by_suite_and_dim(
            self.features_df, suite_names, dims
        )

        if color_by not in ["dimension", "suite"]:
            raise ValueError("color_by must be either 'dimension' or 'suite'.")

        # Setup color mapping
        if color_by == "dimension":
            color_by = "D"
            unique_values = filtered_df["D"].unique()
            palette = sns.color_palette("hsv", len(unique_values))
            color_map = dict(zip(unique_values, palette))
            hue_order = sorted(unique_values)
        else:  # color_by == 'suite'
            color_by = "Suite"
            unique_values = filtered_df["Suite"].unique()
            palette = sns.color_palette("coolwarm", len(unique_values))
            color_map = dict(zip(unique_values, palette))
            hue_order = sorted(unique_values)

        # Plotting
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=filtered_df,
            x=feature_x,
            y=feature_y,
            hue=color_by,
            palette=color_map,
            hue_order=hue_order,
            s=50,
            alpha=0.7,
            edgecolor="none",
        )
        plt.title(f"{feature_y} vs. {feature_x}")
        plt.xlabel(feature_x)
        plt.ylabel(feature_y)
        plt.legend(
            title=color_by.capitalize(), bbox_to_anchor=(1.05, 1), loc="upper left"
        )
        plt.tight_layout()
        plt.show()

    def plot_problem_features(
        self, problem_name, dim, analysis_type, features=None, path=None
    ):
        """
        Creates a grid of violin plots for specified features for a specific problem instance.
        :param problem_name: Name of the problem.
        :param dim: Dimension of the problem.
        :param analysis_type: Type of analysis (e.g., "summary", "detailed").
        :param features: Optional list of features to plot. If None, plots all columns.
        """

        df = self.get_problem_features_samples_df(
            problem_name, dim, analysis_type, path
        )

        # If no specific features are provided, use all numeric columns
        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()

            # Apply z-score normalization
            df[features] = df[features].apply(zscore)

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

    def run_pca(self):

        df = self.custom_drop_na(self.ignore_specified_features(self.features_df))

        scaler = StandardScaler()
        self.features_scaled = scaler.fit_transform(
            FeaturesDashboard.get_numerical_data_from_df(df)
        )

        # Applying PCA
        self.pca = PCA()
        self.pca.fit(self.features_scaled)

        # Find ideal number of PCs.
        minimum_expl_variance = 0.8
        expl_variance = np.cumsum(self.pca.explained_variance_ratio_)
        num_pcs = int(np.argwhere(minimum_expl_variance < expl_variance)[0] + 1)

        print(
            f"We need {num_pcs} PCs to explain {minimum_expl_variance*100}% of the variance"
        )

        return num_pcs

    def plot_scree_plot(self):
        if self.pca is None:
            print("Please run PCA first using run_pca method.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(
            range(1, len(self.pca.explained_variance_ratio_) + 1),
            np.cumsum(self.pca.explained_variance_ratio_),
        )
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("Scree Plot")
        plt.grid(True)
        plt.show()

    def get_pca_loadings(self, n_components, analysis_type=None, pct=True):
        df = self.get_numerical_data_from_df(
            self.custom_drop_na(self.ignore_specified_features(self.features_df))
        )

        if analysis_type:
            df = self.get_features_for_analysis_type(df, analysis_type=analysis_type)

        # Determine how much of the variance is explained by the chosen number of components.
        expl_variance = np.cumsum(self.pca.explained_variance_ratio_)[n_components - 1]

        print(
            f"Using {n_components} components explains {expl_variance*100:.2f}% of the variance."
        )

        # Adjust the index in loadings DataFrame to match the filtered features
        if analysis_type:
            # Ensure the components correspond to the filtered features
            component_indices = [
                df.columns.get_loc(c)
                for c in df.columns
                if c in self.features_df.columns
            ]
            loadings = pd.DataFrame(
                self.pca.components_[
                    :n_components, component_indices
                ].T,  # Slice components for filtered features
                columns=[f"PC{i+1}" for i in range(n_components)],
                index=df.columns,
            )
        else:
            loadings = pd.DataFrame(
                self.pca.components_[:n_components].T,
                columns=[f"PC{i+1}" for i in range(n_components)],
                index=df.columns,
            )

        loadings.index = loadings.index.str.replace("_mean", "")

        # Normalize loadings to percentage
        if pct:
            loadings = loadings.apply(lambda x: abs(x) / abs(x).sum(), axis=0) * 100

        print(
            f"There are {len(loadings.index)} features that have been kept in the PCA."
        )

        return loadings

    def plot_pca_loadings(self, n_components, top_features=None, analysis_type=None):
        if self.pca is None:
            print("Please run PCA first using the run_pca method.")
            return

        loadings = self.get_pca_loadings(
            n_components=n_components, analysis_type=analysis_type
        ).apply(
            lambda x: x.abs()
        )  # Take the absolute value

        # Creating a series of subplots - one for each principal component
        fig, axes = plt.subplots(
            n_components, 1, figsize=(10, n_components * 5), squeeze=False
        )

        # Plotting the bar chart for each component's loadings
        for i, ax in enumerate(axes.flat):
            if top_features is not None:
                # Sort by absolute loadings and take top N for each component
                loadings_sorted = (
                    loadings[f"PC{i+1}"].sort_values(ascending=False).head(top_features)
                )
            else:
                loadings_sorted = loadings[f"PC{i+1}"].sort_values(ascending=False)

            if top_features:
                loadings_sorted = loadings_sorted.head(top_features)

            loadings_sorted.plot(kind="bar", ax=ax)
            ax.axhline(y=100 / len(loadings[f"PC{i+1}"]), color="r", linestyle="--")
            ax.set_title(f"Loadings for PC{i+1}")
            ax.set_ylabel("Absolute Loading Value")
            ax.set_xlabel("Features")

        plt.tight_layout()
        plt.show()

    def plot_stacked_pca_loadings(
        self, n_components, top_features=None, analysis_type=None
    ):
        if self.pca is None:
            print("Please run PCA first using the run_pca method.")
            return

        loadings = self.get_pca_loadings(
            n_components=n_components, analysis_type=analysis_type
        ).apply(
            lambda x: x.abs()
        )  # Take the absolute value

        # Sort the features based on their maximum contribution across all components
        loadings["max_loading"] = loadings.sum(axis=1)
        loadings = loadings.sort_values("max_loading", ascending=False).drop(
            "max_loading", axis=1
        )

        if top_features:
            loadings = loadings.head(top_features)

        # Creating the stacked bar chart
        loadings.plot(kind="bar", stacked=True, figsize=(10, 6))
        plt.title(
            f"Stacked PCA Loadings for Top {top_features} Features"
            if top_features
            else "Stacked PCA Loadings"
        )
        plt.ylabel("Absolute Loading Value")
        plt.xlabel("Features")
        plt.legend(title="Principal Components")

        plt.tight_layout()
        plt.show()

    def plot_total_contribution(
        self, n_components, top_features=None, analysis_type=None
    ):
        if self.pca is None:
            print("Please run PCA first using the run_pca method.")
            return

        if n_components > len(self.pca.explained_variance_ratio_):
            print("Number of components exceeds the total available components")
            return

        loadings = self.get_pca_loadings(
            n_components=n_components, analysis_type=analysis_type
        ).apply(
            lambda x: x.abs()
        )  # Take the absolute value

        eigenvalues = self.pca.explained_variance_[:n_components]

        # Calculate the total contribution for each feature across n components
        total_contributions = loadings.apply(
            lambda x: (x * eigenvalues).sum() / eigenvalues.sum(), axis=1
        )

        # Calculate the expected average contribution (uniform distribution)
        num_variables = len(loadings.index)
        expected_contribution = (
            sum([1 / num_variables * eig for eig in eigenvalues])
            / sum(eigenvalues)
            * 100
        )

        # Sort the contributions in descending order
        total_contributions = total_contributions.sort_values(ascending=False)

        # If top_features is specified, only show the top N features
        if top_features is not None:
            total_contributions = total_contributions.head(top_features)

        # Plotting the total contributions
        total_contributions.plot(kind="bar", figsize=(10, 6))
        plt.axhline(y=expected_contribution, color="r", linestyle="--")
        plt.title(
            f"Total Contribution to the First {n_components} Principal Components"
        )
        plt.ylabel("Total Contribution")
        plt.xlabel("Features")

        plt.tight_layout()
        plt.show()

    def visualise_3d_pca(self):

        if self.pca is None:
            print("Please run PCA first using run_pca method.")
            return

        # Use the PCA results from the run_pca method
        pca_transformed = self.pca.transform(self.features_scaled)

        # Create a DataFrame for the first three PCs
        pca_df = pd.DataFrame(
            data=pca_transformed[:, :3], columns=["PC1", "PC2", "PC3"]
        )

        # Assuming 'Suite' is a column in self.features_df
        pca_df["Suite"] = self.features_df["Suite"].values

        # Plotting
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Iterate over each unique Suite value to plot them in different colors
        for suite in pca_df["Suite"].unique():
            subset = pca_df[pca_df["Suite"] == suite]
            ax.scatter(
                subset["PC1"],
                subset["PC2"],
                subset["PC3"],
                s=50,
                alpha=0.5,
                label=suite,
            )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title("3D PCA Plot")
        ax.legend(title="Suite")
        plt.show()

    def plot_parallel_coordinates(
        self,
        class_column="Suite",
        separate_rw_analytical=False,
        features=None,
        suite_names=None,
        dims=None,
        colormap="Set1",
    ):
        """
        Generate a parallel coordinates plot.

        :param class_column: The column name in df to use for coloring the lines in the plot. Default is 'Suite'.
        :param features: A list of column names to include in the plot. If None, all numeric columns are included.
        :param suite_names: Suites to filter by (not used in this snippet but assumed to be part of your filtering logic).
        :param dims: Dimensions to filter by (not used in this snippet but assumed to be part of your filtering logic).
        :param colormap: The colormap to use for different classes.
        """

        df_filtered = self.filter_df_by_suite_and_dim(
            self.features_df, suite_names=suite_names, dims=dims
        )

        if features:
            cols = [
                f"{f}_mean" if f"{f}_mean" in df_filtered.columns else f
                for f in features
            ]
        else:
            cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()

        cols = list(set(cols + [class_column]))
        features_to_normalize = [col for col in cols if col not in ["D", "Suite"]]

        scaler = MinMaxScaler()
        df_filtered[features_to_normalize] = scaler.fit_transform(
            df_filtered[features_to_normalize]
        )

        # Label suites as "Analytical" if they are in self.analytical_problems
        if separate_rw_analytical:
            df_filtered["Suite"] = df_filtered["Suite"].apply(
                lambda suite: (
                    "Analytical" if suite in self.analytical_problems else suite
                )
            )

        data_to_plot = df_filtered[cols]

        if class_column not in data_to_plot.columns:
            raise ValueError(
                f"The specified class_column '{class_column}' does not exist in the DataFrame."
            )

        plt.figure(figsize=(12, 9))
        parallel_coordinates(data_to_plot, class_column, colormap=colormap, alpha=0.8)
        plt.title("Parallel Coordinates Plot")
        plt.xlabel("Features")
        plt.ylabel("Values")
        plt.xticks(rotation=90)
        plt.grid(True)
        plt.show()

    def get_feature_coverage(
        self, target_suite, analysis_type, features=None, ignore_features=True
    ):
        df = FeaturesDashboard.get_features_for_analysis_type(
            self.features_df, analysis_type
        )

        # Remove the columns to ignore right at the beginning
        if ignore_features:
            df = self.ignore_specified_features(df)

        if features:
            features = [
                f"{f}_mean" if f"{f}_mean" in df.columns else f for f in features
            ]
        else:
            features = [
                col for col in df.columns if col not in ["Name", "D", "Date", "Suite"]
            ]

        # Normalize the features. Drop any NaNs
        df = self.custom_drop_na(df)
        scaler = MinMaxScaler()
        df.loc[:, features] = scaler.fit_transform(df[features])

        # Initialize a DataFrame to store coverage values
        suites = df["Suite"].unique()
        if target_suite is not None:
            suites = suites[suites != target_suite]  # Exclude the target suite
        coverage_values = pd.DataFrame(index=features, columns=suites)

        # Calculate coverage for each feature against the target suite for each non-target suite
        for suite in suites:
            for feature in features:
                # Determine target values based on the target_suite
                if target_suite is None:
                    target_values = df[df["Suite"] != suite][feature]
                else:
                    target_values = df[df["Suite"] == target_suite][feature]

                suite_values = df[df["Suite"] == suite][feature]

                # Calculate distances and coverage
                distances = [min(abs(value - suite_values)) for value in target_values]

                # Store coverage as 1 minus the mean of distances
                coverage_values.at[feature, suite] = 1 - np.mean(distances)

        # Remove the mean from names.
        coverage_values.index = coverage_values.index.str.replace("_mean", "")

        return coverage_values

    def get_poor_coverage_features(self, coverage_values, top_features=None):
        """
        Identify the top features with poor coverage and generate a parallel coordinates plot for them.

        :param coverage_values: The DataFrame containing coverage values.
        :param top_features: The number of features to consider with the lowest coverage. If None, all features are considered.
        """
        # Find the mean coverage across all suites for each feature
        coverage_values["mean_coverage"] = coverage_values.mean(axis=1)

        # Sort the features by their mean coverage
        sorted_features = coverage_values.sort_values(
            by="mean_coverage", ascending=True
        )

        # If top_features is specified, select the top N features with the lowest coverage
        if top_features is not None:
            sorted_features = sorted_features.head(top_features)

        # Extract the feature names
        features_to_plot = sorted_features.index.tolist()

        return features_to_plot

    def plot_coverage_heatmap(
        self, target_suite, analysis_type, features=None, ignore_features=True
    ):
        """
        Calculate and plot the coverage heatmap.

        :param target_suite: The suite to compare all other suites against. If None, compare against the entire dataset.
        :param analysis_type: The type of analysis to filter features.
        :param features: The features to include in the coverage calculation.
        """

        coverage_values = self.get_feature_coverage(
            target_suite=target_suite,
            analysis_type=analysis_type,
            features=features,
            ignore_features=ignore_features,
        )

        suites = coverage_values.columns
        features = coverage_values.index

        # Determine optimal figure size based on the number of features and suites
        fig_width = max(12, len(suites) * 1.2)
        fig_height = max(8, len(features) * 0.4)

        plt.figure(figsize=(fig_width, fig_height))
        cmap = LinearSegmentedColormap.from_list(
            "custom_blue", ["blue", "white"], N=256
        )
        sns.heatmap(
            coverage_values.astype(float),
            annot=False,
            cmap=cmap,
            cbar_kws={"shrink": 0.5},
            vmin=0.6,
            vmax=1,
            square=True,
        )
        plt.title(
            f"Coverage Heatmap for {analysis_type} features. Relative to {'all suites' if target_suite is None else target_suite}"
        )
        plt.xlabel("Suites")
        plt.ylabel("Features")
        plt.xticks(
            rotation=45, fontsize=10
        )  # Rotate x-axis labels for better readability
        plt.tight_layout()
        plt.show()

    def plot_radviz(
        self,
        class_column="Suite",
        features=None,
        suite_names=None,
        dims=None,
        colormap="Set1",
        use_analytical_problems=False,
    ):
        """
        Generate a RadViz plot.

        :param class_column: The column name in df to use for coloring the lines in the plot. Default is 'Suite'.
        :param features: A list of column names to include in the plot. If None, all numeric columns are included.
        :param suite_names: Suites to filter by (not used in this snippet but assumed to be part of your filtering logic).
        :param dims: Dimensions to filter by (not used in this snippet but assumed to be part of your filtering logic).
        :param colormap: The colormap to use for different classes.
        """

        df_filtered = self.filter_df_by_suite_and_dim(
            self.ignore_specified_features(self.features_df),
            suite_names=suite_names,
            dims=dims,
        )

        if features:
            # Append "_mean" to each feature if it's not already there and ensure the feature exists in the DataFrame
            cols = [f if f in df_filtered.columns else f for f in features]
        else:
            # If no features specified, use all numeric columns
            cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()

        # Further filtering.
        cols = [c for c in cols if not df_filtered[c].nunique() == 1]

        cols.append("Suite")

        # Ensure "D" and "Suite" are included for plotting but not normalized
        # cols = list(set(cols + [class_column]))
        # features_to_normalize = [col for col in cols if col not in ["D", "Suite"]]

        # # Normalize the features
        # scaler = StandardScaler()
        # df_filtered[features_to_normalize] = scaler.fit_transform(
        #     df_filtered[features_to_normalize]
        # )

        if use_analytical_problems:
            df_filtered["Suite"] = df_filtered["Suite"].apply(
                lambda x: "Analytical" if x in self.analytical_problems else x
            )

        # Create a subset DataFrame with the selected features and class_column
        data_to_plot = df_filtered[cols]

        # Check if the class_column exists
        if class_column not in data_to_plot.columns:
            raise ValueError(
                f"The specified class_column '{class_column}' does not exist in the DataFrame."
            )

        # Generate the parallel coordinates plot
        plt.figure(figsize=(12, 9))
        pd.plotting.radviz(data_to_plot, "Suite", colormap=colormap, s=2)
        plt.title("RadViz Plot")
        # plt.grid(True)
        plt.show()

    def plot_tSNE(
        self,
        analysis_type=None,
        features=None,
        suite_names=None,
        dims=None,
        colormap="Set1",
        use_analytical_problems=False,
    ):
        df_filtered = self.custom_drop_na(
            self.filter_df_by_suite_and_dim(
                self.ignore_specified_features(self.features_df),
                suite_names=suite_names,
                dims=dims,
            )
        )

        if analysis_type:
            df_filtered = self.get_features_for_analysis_type(
                df_filtered, analysis_type
            )

        if features:
            cols = [f if f in df_filtered.columns else f for f in features]
        else:
            cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()

        cols = [c for c in cols if not df_filtered[c].nunique() == 1]

        if use_analytical_problems:
            df_filtered["Suite"] = df_filtered["Suite"].apply(
                lambda x: "Analytical" if x in self.analytical_problems else x
            )

        tsne = TSNE(n_components=2, random_state=0, perplexity=30)
        tsne_results = tsne.fit_transform(df_filtered[cols])

        plt.figure(figsize=(8, 6))
        unique_suites = df_filtered["Suite"].unique()
        for suite in unique_suites:
            indices = df_filtered["Suite"] == suite
            plt.scatter(
                tsne_results[indices, 0], tsne_results[indices, 1], s=6, label=suite
            )

        plt.title("$t$-SNE")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.grid()
        plt.legend()
        plt.show()

    def plot_som(
        self,
        analysis_type=None,
        features=None,
        suite_names=None,
        dims=None,
        colormap="Set1",
        use_analytical_problems=False,
        net_size=10,  # Network size for SimpSOM
    ):
        df_filtered = self.custom_drop_na(
            self.filter_df_by_suite_and_dim(
                self.ignore_specified_features(self.features_df),
                suite_names=suite_names,
                dims=dims,
            )
        )

        if analysis_type:
            df_filtered = self.get_features_for_analysis_type(
                df_filtered, analysis_type
            )

        if features:
            cols = [f if f in df_filtered.columns else f for f in features]
        else:
            cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()

        cols = [c for c in cols if not df_filtered[c].nunique() == 1]

        if use_analytical_problems:
            df_filtered["Suite"] = df_filtered["Suite"].apply(
                lambda x: "Analytical" if x in self.analytical_problems else x
            )

        # Normalize the features
        data = df_filtered[cols].values
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

        # Initialize and train the SOM
        net = sps.SOMNet(net_size, net_size, data, PBC=True)
        net.train(train_algo="batch", start_learning_rate=0.01)

        # Project the data onto the SOM
        proj = net.project(data)

        _ = net.plot_map_by_feature(feature=100, show=True, print_out=True)

        # Plotting
        plt.figure(figsize=(8, 8))
        # for i, (x, y) in enumerate(proj):
        #     plt.text(
        #         x,
        #         y,
        #         df_filtered["Suite"].iloc[i],
        #         color=plt.cm.get_cmap(colormap)(i / len(df_filtered)),
        #         fontdict={"weight": "bold", "size": 9},
        #     )

        plt.title("SOM Projection")
        plt.show()
