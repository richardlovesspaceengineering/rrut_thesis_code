import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shutil
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE
from scipy.stats import zscore
import umap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.patches as mpatches  # For custom legend
import matplotlib.lines as mlines  # For custom legend lines
import warnings
import copy
from cycler import cycler
from matplotlib.colors import ListedColormap
import re

warnings.simplefilter(action="ignore", category=FutureWarning)


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
    def __init__(self, results_dict, new_save_path, report_mode):
        """
        Initialize the FeaturesDashboard with a dictionary mapping problem names to lists of result storage folders.
        :param results_dict: Dictionary where keys are problem names and values are lists of storage folder names.
        :param new_save_path: Path to the directory where new files will be saved.
        """
        self.results_dict = results_dict
        self.new_save_path = new_save_path
        self.report_mode = report_mode

        # can be altered by the set_dashboard_analysis_type method
        self.analysis_type = "all"

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
            "XA": [f"XA{x}" for x in range(2, 9, 1)],
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

        # Generate suite colours using custom colour palette.
        self.apply_custom_colors()
        self.suite_color_map = self.generate_suite_colors()

    def generate_suite_colors(self):

        # Get a list of unique suites
        suites = list(self.get_suite_names(ignore_aerofoils=False))

        # Get the color cycle from the current matplotlib style
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        # Assign colors to suites, cycling through the color list if necessary
        suite_colors = {
            suite: colors[i + 1 % len(colors)] for i, suite in enumerate(suites)
        }

        return suite_colors

    def get_suite_names(self, ignore_aerofoils=False):

        suite_names = list(self.suites_problems_dict.keys())

        if ignore_aerofoils:
            suite_names.remove("XA")

        return suite_names

    def check_dashboard_is_global_only_for_aerofoils(self, ignore_aerofoils):
        if self.analysis_type == "glob" and not ignore_aerofoils:
            print("Aerofoils are being considered for this global only feature set.")
        elif self.analysis_type != "glob" and not ignore_aerofoils:
            print(
                f"Aerofoils have been specified for consideration, but the dataset for {self.analysis_type} samples is being considered. Restrict to global features only."
            )
        elif ignore_aerofoils:
            print(
                f"Aerofoils are being ignored for analysis type: {self.analysis_type}."
            )

    def define_plot_sizes(self):

        self.plt_width_full = 6.3

    def apply_custom_colors(self):
        plt.rc(
            "axes",
            prop_cycle=cycler(
                "color",
                [
                    "#000000",
                    "#2394d6",
                    "#dc3220",
                    "#ffc20a",
                    "#994f00",
                    "#3c69e1",
                    "#8bc34a",
                    "#7d2e8d",
                    "#e66100",
                    "#595959",
                    "#0e0354",
                ],
            ),
        )

    def apply_custom_matplotlib_style(self, fontsize=12, legendfontsize=10):
        # Line settings
        plt.rc("lines", linewidth=2.25, markersize=10)

        # Axes settings
        plt.rc(
            "axes",
            linewidth=1.75,
            labelsize=fontsize,
            titlesize=fontsize,
        )

        # Tick settings
        plt.rc("xtick", labelsize=fontsize)
        plt.rc("ytick", labelsize=fontsize)

        # Legend settings
        plt.rc("legend", fontsize=legendfontsize)

        # Text settings
        if self.report_mode:
            plt.rc("text", usetex=True)
            plt.rc("font", family="serif", serif=["Computer Modern"])

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

    def get_feature_names(
        self, shortened=True, ignore_features=False, analysis_type=None
    ):

        df = FeaturesDashboard.get_numerical_data_from_df(self.features_df)

        if ignore_features:
            df = self.ignore_specified_features(df=df)

        if analysis_type:
            df = self.get_features_for_analysis_type(df, analysis_type=analysis_type)

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
        df = self.features_df
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
                    self.features_to_ignore.append(
                        feature + "_mean" if "_mean" not in feature else feature
                    )

    def use_these_features_only(self, features_to_keep, use_pre_ignored=True):
        names_tracker = copy.deepcopy(features_to_keep)

        if not use_pre_ignored:
            self.features_to_ignore = []

        for feature in self.get_feature_names():
            if (
                feature not in self.features_to_ignore
                and feature not in features_to_keep
            ):
                self.features_to_ignore.append(
                    feature + "_mean" if "_mean" not in feature else feature
                )
            else:
                names_tracker.remove(feature)

        if len(names_tracker) == 0:
            print("All specified features have been considered")
        else:
            print("The following features were not added: ")
            print(names_tracker)

    def ignore_specified_features(self, df):
        if self.features_to_ignore:
            return df[[col for col in df.columns if col not in self.features_to_ignore]]
        else:
            print("No features to ignore have been set as attributes of the instance.")
            return df

    def set_dashboard_analysis_type(self, analysis_type):
        self.features_df = self.get_features_for_analysis_type(
            self.features_df, analysis_type=analysis_type
        )
        self.analysis_type = analysis_type

    def set_dashboard_dimensionality(self, d):

        if not isinstance(d, list):
            d = [d]

        self.features_df = self.filter_df_by_suite_and_dim(self.features_df, dims=d)

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
        self,
        feature_name,
        suite_names=None,
        dims=None,
        show_plot=True,
        ignore_aerofoils=True,
    ):
        """
        Generates a 1xN grid of violin plots for a specified feature across different suites, with points overlaying the violin plots.
        Each violin plot represents a different suite, with distinct colors used for each dimension.
        :param feature_name: The name of the feature to plot. Can be a landscape feature or algo performance.
        :param dims: A list of dimensions to plot.
        :param suite_names: Optional. A list of benchmark suite names to filter the features by.
        """

        # Check data compatibility.
        self.check_dashboard_is_global_only_for_aerofoils(
            ignore_aerofoils=ignore_aerofoils
        )

        # Get plot style ready.
        self.apply_custom_matplotlib_style()

        feature_name_with_mean = feature_name + "_mean"

        if not suite_names:
            suite_names = self.get_suite_names(ignore_aerofoils=ignore_aerofoils)

        # Use the filtering method to get the filtered DataFrame based on suite names and dimensions
        df_filtered = self.filter_df_by_suite_and_dim(
            self.features_df,
            suite_names=suite_names,
            dims=dims,
            ignore_aerofoils=ignore_aerofoils,
        )

        # Define markers for different dimensions.
        marker_dict, unique_d_values = self.get_dimension_markers(df=df_filtered)

        global_min, global_max = self.compute_global_maxmin_for_plot(
            df_filtered, feature_name_with_mean
        )

        fig, axes = plt.subplots(1, len(suite_names), figsize=(15, 4), sharey=True)

        for i, ax in enumerate(axes):
            suite_name = suite_names[i]
            df_suite_filtered = df_filtered[df_filtered["Suite"] == suite_name]

            # Plot a single violin plot for the combined data
            if not df_suite_filtered.empty:
                vp = sns.violinplot(
                    ax=ax,
                    y=df_suite_filtered[feature_name_with_mean],
                    color=self.suite_color_map[suite_name],
                )

                # Reduce transparency
                plt.setp(vp.collections, alpha=0.4)

                # Overlay sample size.
                num_points = df_suite_filtered.shape[0]
                ax.text(
                    0.925,
                    0.95,
                    rf"$n={num_points}$",  # Position inside the subplot, top right corner
                    verticalalignment="top",
                    horizontalalignment="right",
                    transform=ax.transAxes,  # Coordinate system relative to the axes
                    color="black",
                    fontsize=8,
                )

                # Overlay points for each suite.
                for dim in unique_d_values:
                    df_suite_dim_filtered = df_suite_filtered[
                        df_suite_filtered["D"] == dim
                    ]
                    if not df_suite_dim_filtered.empty:
                        sns.stripplot(
                            ax=ax,
                            y=df_suite_dim_filtered[feature_name_with_mean],
                            color=self.suite_color_map[
                                suite_name
                            ],  # Default to gray if suite not in colors
                            marker=marker_dict[dim],
                            size=4,
                            alpha=0.9,
                            jitter=True,
                        )

            ax.grid(False)
            ax.set_title(f"{suite_name}")
            ax.set_ylabel(
                r"\texttt{" + feature_name_with_mean[:-5] + "}" if i == 0 else ""
            )
            ax.set_xlabel("")
            ax.set_ylim(global_min, global_max)

        # Adjust the layout to make space for the top legend
        # plt.subplots_adjust(top=0.85, bottom=0.15)
        plt.tight_layout()

        # Add legend after applying tight layout.
        self.create_custom_legend_for_dimension(
            fig=fig, marker_dict=marker_dict, bottom_box_anchor=-0.1
        )
        # plt.savefig(
        #     "../../rrut_thesis_report/Figures/test.pdf",
        #     format="pdf",
        #     bbox_inches="tight",
        # )
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
        ignore_aerofoils=True,
    ):
        """
        Returns the top n correlation pairs from the correlation matrix, sorted by absolute correlation value.

        :param n: Number of top correlation pairs to return.
        """

        # Check data compatibility.
        self.check_dashboard_is_global_only_for_aerofoils(
            ignore_aerofoils=ignore_aerofoils
        )

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

        # Get a list of all suites if not provided.
        if not suite_names:
            suite_names = self.get_suite_names(ignore_aerofoils=ignore_aerofoils)

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
            suite_names = self.get_suite_names(ignore_aerofoils=False)

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

    def filter_df_by_suite_and_dim(
        self, df, suite_names=None, dims=None, ignore_aerofoils=True
    ):
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

        # Always remove aerofoils unless we are looking at global features.
        if ignore_aerofoils:
            df_filtered = df_filtered[df_filtered["Suite"] != "XA"]

        # Filter by dimension if dims is specified
        if dims:
            df_filtered = df_filtered[df_filtered["D"].isin(dims)]

        return df_filtered

    def plot_features_comparison(
        self,
        feature_x,
        feature_y,
        suite_names=None,
        dims=None,
        ignore_aerofoils=True,
        use_analytical_problems=False,
        figsize=(10, 6),
        text_annotations_for_suites=None,
        zoom_to_suite=None,
        zoom_margin=0.2,
    ):
        """
        Plots two features against each other, with point colors corresponding to either dimension or suite.
        Allows specifying suite names and dimensions for slicing the data before plotting.
        """

        # Get plot style ready.
        self.apply_custom_matplotlib_style()

        # Check data compatibility.
        self.check_dashboard_is_global_only_for_aerofoils(
            ignore_aerofoils=ignore_aerofoils
        )

        feature_x = feature_x + "_mean"
        feature_y = feature_y + "_mean"

        if not suite_names:
            suite_names = self.get_suite_names(ignore_aerofoils=False)

        # Extract required data.
        df_filtered, cols = self.get_filtered_df_for_dimension_reduced_plot(
            analysis_type=None,
            features=[feature_x, feature_y],
            suite_names=suite_names,
            dims=dims,
            use_analytical_problems=use_analytical_problems,
            ignore_aerofoils=ignore_aerofoils,
            ignore_scr=False,
            ignore_features=False,
        )

        # We will use different markers for each dimension.
        marker_dict, unique_d_values = self.get_dimension_markers(df=df_filtered)

        # Plotting
        fig, ax = plt.subplots(figsize=figsize)

        # Bounds for zooming initialisation.
        min_x, max_x, min_y, max_y = np.inf, -np.inf, np.inf, -np.inf

        for suite in df_filtered["Suite"].unique():
            for d_value in unique_d_values:
                indices = (df_filtered["Suite"] == suite) & (
                    df_filtered["D"] == d_value
                )
                x_values = df_filtered[indices][feature_x]
                y_values = df_filtered[indices][feature_y]
                names = df_filtered[indices]["Name"]

                sc = ax.scatter(
                    x_values,
                    y_values,
                    color=self.suite_color_map[suite],
                    s=20,
                    marker=marker_dict[d_value],
                    zorder=3,  # renders on top of grid
                )

                # Check if this is the suite to zoom into
                if (
                    zoom_to_suite is not None
                    and suite == zoom_to_suite
                    and len(names) > 0
                ):
                    min_x = min(min_x, x_values.min())
                    max_x = max(max_x, x_values.max())
                    min_y = min(min_y, y_values.min())
                    max_y = max(max_y, y_values.max())

                # Annotate each point with the extracted number
                if text_annotations_for_suites is not None:
                    if suite in text_annotations_for_suites:
                        for i, txt in enumerate(names):
                            print(txt)
                            num = re.search(r"([^\d]+)(\d+)_", txt).group(2)
                            ax.annotate(
                                num,
                                (x_values[i], y_values[i]),
                                textcoords="offset points",
                                xytext=(0, 5),
                                ha="center",
                            )

        if zoom_to_suite is not None:
            # Set the zoom into the suite
            ax.set_xlim(
                min_x - zoom_margin * (max_x - min_x),
                max_x + zoom_margin * (max_x - min_x),
            )  # Add a margin
            ax.set_ylim(
                min_y - zoom_margin * (max_y - min_y),
                max_y + zoom_margin * (max_y - min_y),
            )

        # Grid
        self.apply_custom_grid(ax=ax)

        plt.title(f"{feature_y} vs. {feature_x}")
        plt.xlabel(feature_x)
        plt.ylabel(feature_y)
        self.create_custom_legend_for_dimension(fig=fig, marker_dict=marker_dict)
        self.create_custom_legend_for_suite(
            fig=fig, df=df_filtered, ignore_aerofoils=ignore_aerofoils
        )
        plt.tight_layout()

        # Adjust the layout to make space for the top legend
        # plt.subplots_adjust(top=1, bottom=0)
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

    def calc_n_principal_compons_for_var(self, minimum_expl_variance):
        # Find number of PCs required to explain a percentage of the variance (given as decimal)
        expl_variance = np.cumsum(self.pca.explained_variance_ratio_)
        num_pcs = int(np.argwhere(minimum_expl_variance < expl_variance)[0] + 1)

        print(
            f"We need {num_pcs} PCs to explain {minimum_expl_variance*100}% of the variance"
        )
        return num_pcs

    def run_pca(self, scaler_type="StandardScaler"):

        df = self.custom_drop_na(self.ignore_specified_features(self.features_df))

        # Scale the features based on the specified scaler_type
        if scaler_type == "MinMaxScaler":
            scaler = MinMaxScaler()
        elif scaler_type == "StandardScaler":
            scaler = StandardScaler()
        self.features_scaled = scaler.fit_transform(
            FeaturesDashboard.get_numerical_data_from_df(df)
        )

        # Applying PCA
        self.pca = PCA()
        self.pca.fit(self.features_scaled)

        num_pcs = self.calc_n_principal_compons_for_var(minimum_expl_variance=0.8)

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

    def get_total_pca_contribution(
        self, n_components, top_features=None, analysis_type=None, worst=False
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

        # Sort the contributions in descending order
        total_contributions = total_contributions.sort_values(ascending=False)

        # If top_features is specified, only show the top N features
        if top_features is not None:

            if worst:
                total_contributions = total_contributions.tail(top_features)
            else:
                total_contributions = total_contributions.head(top_features)

        return total_contributions

    def plot_total_contribution(
        self, n_components, top_features=None, analysis_type=None
    ):

        eigenvalues = self.pca.explained_variance_[:n_components]

        total_contributions = self.get_total_pca_contribution(
            n_components=n_components,
            top_features=top_features,
            analysis_type=analysis_type,
        )

        loadings = self.get_pca_loadings(
            n_components=n_components, analysis_type=analysis_type
        ).apply(
            lambda x: x.abs()
        )  # Take the absolute value

        # Calculate the expected average contribution (uniform distribution)
        num_variables = len(loadings.index)
        expected_contribution = (
            sum([1 / num_variables * eig for eig in eigenvalues])
            / sum(eigenvalues)
            * 100
        )

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
        suite_in_focus=None,
        features=None,
        suite_names=None,
        dims=None,
        ignore_features=True,
        ignore_aerofoils=True,
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
            self.features_df,
            suite_names=suite_names,
            dims=dims,
            ignore_aerofoils=ignore_aerofoils,
        )

        if ignore_features:
            df_filtered = self.ignore_specified_features(df_filtered)

        if features:
            cols = [
                f"{f}_mean" if f"{f}_mean" in df_filtered.columns else f
                for f in features
            ]
        else:
            cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()

        cols = list(set(cols + [class_column]))

        if "D" in cols:
            cols.remove("D")  # no need to consider dimension
        features_to_normalize = [col for col in cols if col not in ["Suite"]]

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

        # Create a colormap object from the suite colors
        unique_suites = self.get_suite_names(ignore_aerofoils=False)
        suite_colors = [self.suite_color_map[suite] for suite in unique_suites]

        # Create a categorical type based on the custom order - this helps in sorting the DataFrame
        data_to_plot.loc[:, "Suite"] = pd.Categorical(
            data_to_plot["Suite"], categories=unique_suites, ordered=True
        )

        # Sort the DataFrame by 'Suite' to ensure it follows the custom order
        data_to_plot = data_to_plot.sort_values("Suite")

        # Sort the DataFrame to ensure the suite_in_focus is at the end
        if suite_in_focus is not None:
            # Create a boolean series to determine if each row is in the suite_in_focus
            is_suite_in_focus = data_to_plot["Suite"] == suite_in_focus

            # Split the DataFrame into two: one for the suite_in_focus and one for the others
            df_suite_in_focus = data_to_plot[is_suite_in_focus]
            df_others = data_to_plot[~is_suite_in_focus]

            # Concatenate the DataFrames, placing the suite_in_focus at the end
            data_to_plot = pd.concat([df_others, df_suite_in_focus])

            # Update suite_colors to have the suite_in_focus color last
            suite_colors = [
                (
                    "#9e9b9b"
                    if suite != suite_in_focus
                    else self.suite_color_map[suite_in_focus]
                )
                for suite in unique_suites
            ]
            suite_colors.sort(key=lambda x: x == self.suite_color_map[suite_in_focus])

        # Create a colormap with the updated colors
        colormap = ListedColormap(suite_colors)

        if class_column not in data_to_plot.columns:
            raise ValueError(
                f"The specified class_column '{class_column}' does not exist in the DataFrame."
            )

        fig, ax = plt.subplots(figsize=(12, 9))
        self.apply_custom_matplotlib_style()
        parallel_coordinates(
            data_to_plot,
            class_column,
            colormap=colormap,
            alpha=0.8,
            ax=ax,
            linewidth=plt.rcParams["lines.linewidth"],
        )

        if suite_in_focus is None:
            title_text = "Parallel Coordinates Plot"
        else:
            ax.legend_.remove()
            title_text = suite_in_focus

        plt.title(title_text)
        plt.xlabel("Features")
        plt.ylabel("Normalised Values")
        plt.xticks(rotation=90)
        plt.grid(True)
        plt.show()

    def get_feature_coverage(
        self,
        target_suite=None,
        analysis_type=None,
        features=None,
        ignore_features=True,
        ignore_aerofoils=True,
    ):

        # Check data compatibility.
        self.check_dashboard_is_global_only_for_aerofoils(
            ignore_aerofoils=ignore_aerofoils
        )

        df = self.features_df

        if analysis_type:
            df = FeaturesDashboard.get_features_for_analysis_type(df, analysis_type)

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

        # Initialize a DataFrame to store coverage values
        suites = list(
            set(df["Suite"].unique()).intersection(
                set(self.get_suite_names(ignore_aerofoils=ignore_aerofoils))
            )
        )
        if target_suite is not None:
            # Exclude the target suite
            suites = [s for s in suites if s != target_suite]
        coverage_values = pd.DataFrame(index=features, columns=suites)

        # Normalize the features. Drop any NaNs
        df = self.custom_drop_na(
            self.filter_df_by_suite_and_dim(
                df, suite_names=None, dims=None, ignore_aerofoils=ignore_aerofoils
            )
        )
        scaler = MinMaxScaler()
        df.loc[:, features] = scaler.fit_transform(df[features])

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

    def get_top_coverage_features(
        self,
        coverage_values,
        worst=True,
        top_features=None,
        aslist=True,
    ):
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

            if worst:
                sorted_features = sorted_features.head(top_features)
            else:
                sorted_features = sorted_features.tail(top_features)

        if not aslist:
            return sorted_features
        else:
            # Extract the feature names
            features_to_plot = sorted_features.index.tolist()

            return features_to_plot

    def plot_coverage_heatmap(
        self,
        target_suite=None,
        analysis_type=None,
        features=None,
        ignore_features=True,
        ignore_aerofoils=True,
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
            ignore_aerofoils=ignore_aerofoils,
        )

        suites = coverage_values.columns
        features = coverage_values.index

        # Determine optimal figure size based on the number of features and suites
        fig_width = max(12, len(suites) * 1.2)
        fig_height = max(8, len(features) * 0.4)

        plt.figure(figsize=(fig_width, fig_height))

        if target_suite is None:
            cmap = LinearSegmentedColormap.from_list(
                "custom_blue", ["blue", "white"], N=256
            )
        else:
            cmap = LinearSegmentedColormap.from_list(
                "custom", [self.suite_color_map[target_suite], "white"], N=256
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
        scaler_type="MinMaxScaler",
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

        df_filtered = self.custom_drop_na(
            self.filter_df_by_suite_and_dim(
                self.ignore_specified_features(self.features_df),
                suite_names=suite_names,
                dims=dims,
            )
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

        # if scaler_type == "MinMaxScaler":
        #     scaler = MinMaxScaler()
        #     df_filtered[cols] = scaler.fit_transform(df_filtered[cols])
        # elif scaler_type == "StandardScaler":
        #     scaler = StandardScaler()
        #     df_filtered[cols] = scaler.fit_transform(df_filtered[cols])

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
        plt.show()

    def get_filtered_df_for_dimension_reduced_plot(
        self,
        analysis_type=None,
        features=None,
        suite_names=None,
        dims=None,
        use_analytical_problems=False,
        ignore_aerofoils=True,
        ignore_features=True,
        ignore_scr=True,
    ):

        df = self.features_df

        if ignore_features:
            df = self.ignore_specified_features(df)

        df_filtered = self.filter_df_by_suite_and_dim(
            df,
            suite_names=suite_names,
            dims=dims,
            ignore_aerofoils=ignore_aerofoils,
        )

        if ignore_features:
            df_filtered = self.custom_drop_na(df_filtered)

        if analysis_type:
            df_filtered = self.get_features_for_analysis_type(
                df_filtered, analysis_type
            )

        if features:
            cols = [f if f in df_filtered.columns else f for f in features]
        else:
            cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()

        # Remove solver-crash related features from consideration to not affect the results; also remove constant columns.

        if ignore_scr:
            cols = [
                c
                for c in cols
                if "scr" not in c
                and "ncr" not in c
                and not df_filtered[c].nunique() == 1
            ]

        if use_analytical_problems:
            df_filtered["Suite"] = df_filtered["Suite"].apply(
                lambda x: "Analytical" if x in self.analytical_problems else x
            )

        # No need to include dimension in scaling.
        if "D" in cols:
            cols.remove("D")

        print(f"This dataframe contains {len(cols)} features.")

        return df_filtered, cols

    def get_dimension_markers(self, df, special_d_values=[5, 10, 20, 30]):
        # Map 'D' values to markers
        markers = ["D", "X", "s", "^"]  # One marker for each special D value
        marker_dict = {
            d: markers[i] if d in special_d_values else "o"
            for i, d in enumerate(
                special_d_values
                + [d for d in df["D"].unique() if d not in special_d_values]
            )
        }
        unique_d_values = df["D"].unique()

        return marker_dict, unique_d_values

    def apply_feature_scaling(self, df, cols, scaler_type):
        # Scale the features
        if scaler_type == "MinMaxScaler":
            scaler = MinMaxScaler()
            df[cols] = scaler.fit_transform(df[cols])
        elif scaler_type == "StandardScaler":
            scaler = StandardScaler()
            df[cols] = scaler.fit_transform(df[cols])

        return df

    def create_custom_legend_for_dimension(
        self,
        fig,
        marker_dict,
        special_d_values=[5, 10, 20, 30],
        bottom_box_anchor=-0.1,
    ):
        # Create legend for dimensions
        dim_patches = [
            mlines.Line2D(
                [],
                [],
                color="black",
                marker=marker_dict[d],
                linestyle="None",
                markersize=6,
                label=d,
            )
            for d in special_d_values
        ]
        # Use fig.legend() again to place the second legend relative to the figure
        dim_legend = fig.legend(
            handles=dim_patches,
            loc="lower center",
            ncol=4,
            bbox_to_anchor=(0.5, bottom_box_anchor),
            title="Dimensions",
        )

    def create_custom_legend_for_suite(
        self, fig, df, top_box_anchor=1.15, ignore_aerofoils=True
    ):

        suites = [
            s
            for s in self.get_suite_names(ignore_aerofoils=False)
            if s in df["Suite"].unique()
        ]
        if ignore_aerofoils:
            suites = [s for s in suites if s != "XA"]
            ncol = 3
        else:
            top_box_anchor = top_box_anchor - 0.075
            ncol = 5

        suite_patches = [
            mpatches.Patch(color=self.suite_color_map[suite], label=suite)
            for suite in suites
        ]

        # Use fig.legend() to place the legend relative to the figure
        suite_legend = fig.legend(
            handles=suite_patches,
            loc="upper center",
            ncol=ncol,
            bbox_to_anchor=(0.5, top_box_anchor),
            title="Suites",
        )

    def apply_custom_grid(self, ax):
        # Grid
        ax.grid(True, which="major", linestyle="-", linewidth=0.75, zorder=0)
        ax.minorticks_on()
        ax.grid(True, which="minor", linestyle="--", linewidth=0.15, zorder=0)

    def plot_tSNE(
        self,
        analysis_type=None,
        features=None,
        suite_names=None,
        dims=None,
        scaler_type="MinMaxScaler",
        use_analytical_problems=False,
        ignore_aerofoils=True,
        perplexities=[10, 30, 60],  # Add perplexities as an optional argument
        text_annotations_for_suites=None,
        zoom_to_suite=None,
        zoom_margin=0.2,
    ):

        # Check data compatibility.
        self.check_dashboard_is_global_only_for_aerofoils(
            ignore_aerofoils=ignore_aerofoils
        )

        # Get plot style ready.
        self.apply_custom_matplotlib_style()

        # Extract required data.
        df_filtered, cols = self.get_filtered_df_for_dimension_reduced_plot(
            analysis_type=analysis_type,
            features=features,
            suite_names=suite_names,
            dims=dims,
            use_analytical_problems=use_analytical_problems,
            ignore_aerofoils=ignore_aerofoils,
        )

        # We will use different markers for each dimension.
        marker_dict, unique_d_values = self.get_dimension_markers(df=df_filtered)

        # Apply feature scaling
        df_filtered = self.apply_feature_scaling(
            df=df_filtered, cols=cols, scaler_type=scaler_type
        )

        # Now make plot axes.
        fig, axes = plt.subplots(
            1, len(perplexities), figsize=(5 * len(perplexities), 5)
        )

        if len(perplexities) == 1:  # Adjust if only one subplot
            axes = [axes]

        for ax, perplexity in zip(axes, perplexities):
            tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity)
            tsne_results = tsne.fit_transform(df_filtered[cols])

            # Bounds for zooming initialisation.
            min_x, max_x, min_y, max_y = np.inf, -np.inf, np.inf, -np.inf

            for suite in df_filtered["Suite"].unique():
                for d_value in unique_d_values:
                    indices = (df_filtered["Suite"] == suite) & (
                        df_filtered["D"] == d_value
                    )
                    x_values = tsne_results[indices, 0]
                    y_values = tsne_results[indices, 1]
                    names = df_filtered[indices]["Name"]

                    sc = ax.scatter(
                        x_values,
                        y_values,
                        color=self.suite_color_map[suite],
                        s=6,
                        marker=marker_dict[d_value],
                        zorder=3,  # renders on top of grid
                    )

                    # Check if this is the suite to zoom into
                    if (
                        zoom_to_suite is not None
                        and suite == zoom_to_suite
                        and len(names) > 0
                    ):
                        min_x = min(min_x, x_values.min())
                        max_x = max(max_x, x_values.max())
                        min_y = min(min_y, y_values.min())
                        max_y = max(max_y, y_values.max())

                    # Annotate each point with the extracted number
                    if text_annotations_for_suites is not None:
                        if suite in text_annotations_for_suites:
                            for i, txt in enumerate(names):
                                num = re.search(r"([^\d]+)(\d+)_", txt).group(2)
                                ax.annotate(
                                    num,
                                    (x_values[i], y_values[i]),
                                    textcoords="offset points",
                                    xytext=(0, 5),
                                    ha="center",
                                )

            if zoom_to_suite is not None:
                # Set the zoom into the suite
                ax.set_xlim(
                    min_x - zoom_margin * (max_x - min_x),
                    max_x + zoom_margin * (max_x - min_x),
                )  # Add a margin
                ax.set_ylim(
                    min_y - zoom_margin * (max_y - min_y),
                    max_y + zoom_margin * (max_y - min_y),
                )

            ax.set_title(f"$t$-SNE with perplexity = {perplexity}")
            self.apply_custom_grid(ax=ax)

        # Avoid repeated axis labels
        axes[np.floor(len(axes) / 2).astype(int)].set_xlabel("Component 1 [--]")
        axes[0].set_ylabel("Component 2 [--]")

        # Create legend for suites and dimensions
        self.create_custom_legend_for_dimension(fig=fig, marker_dict=marker_dict)
        self.create_custom_legend_for_suite(
            fig=fig, df=df_filtered, ignore_aerofoils=ignore_aerofoils
        )

        plt.tight_layout()
        # Adjust the layout to make space for the top legend
        plt.subplots_adjust(top=0.85, bottom=0.15)
        plt.show()

    def plot_UMAP(
        self,
        analysis_type=None,
        features=None,
        suite_names=None,
        dims=None,
        scaler_type="MinMaxScaler",
        use_analytical_problems=False,
        n_neighbors=[15, 30, 60],
        min_dist=0.2,
        random_state=42,
        ignore_aerofoils=True,
        train_with_aerofoils=False,
        text_annotations_for_suites=None,
        zoom_to_suite=None,
        zoom_margin=0.2,
    ):

        # Get plot style ready.
        self.apply_custom_matplotlib_style()

        # Check data compatibility.
        self.check_dashboard_is_global_only_for_aerofoils(
            ignore_aerofoils=ignore_aerofoils
        )

        # Extract required data.
        df_filtered, cols = self.get_filtered_df_for_dimension_reduced_plot(
            analysis_type=analysis_type,
            features=features,
            suite_names=suite_names,
            dims=dims,
            use_analytical_problems=use_analytical_problems,
            ignore_aerofoils=ignore_aerofoils,
        )

        # We will use different markers for each dimension.
        marker_dict, unique_d_values = self.get_dimension_markers(df=df_filtered)

        # Apply feature scaling
        df_filtered = self.apply_feature_scaling(
            df=df_filtered, cols=cols, scaler_type=scaler_type
        )

        # Now make plot axes.
        fig, axes = plt.subplots(1, len(n_neighbors), figsize=(5 * len(n_neighbors), 5))

        if len(n_neighbors) == 1:  # Adjust if only one subplot
            axes = [axes]

        for ax, n_neighbor in zip(axes, n_neighbors):
            umap_model = umap.UMAP(
                n_neighbors=n_neighbor,
                min_dist=min_dist,
                n_components=2,
                n_jobs=1,
                random_state=random_state,
            )

            if not train_with_aerofoils and self.analysis_type == "glob":
                # Fit UMAP to data without aerofoils.
                umap_model.fit(df_filtered.loc[df_filtered["Suite"] != "XA", cols])

                # Show results for all suites.
                umap_results = umap_model.transform(df_filtered[cols])
            else:
                umap_results = umap_model.fit_transform(df_filtered[cols])

            # Bounds for zooming initialisation.
            min_x, max_x, min_y, max_y = np.inf, -np.inf, np.inf, -np.inf

            for suite in df_filtered["Suite"].unique():
                for d_value in unique_d_values:
                    indices = (df_filtered["Suite"] == suite) & (
                        df_filtered["D"] == d_value
                    )
                    x_values = umap_results[indices, 0]
                    y_values = umap_results[indices, 1]
                    names = df_filtered[indices]["Name"]

                    sc = ax.scatter(
                        x_values,
                        y_values,
                        color=self.suite_color_map[suite],
                        s=6,
                        marker=marker_dict[d_value],
                        zorder=3,  # renders on top of grid
                    )

                    # Check if this is the suite to zoom into
                    if (
                        zoom_to_suite is not None
                        and suite == zoom_to_suite
                        and len(names) > 0
                    ):
                        min_x = min(min_x, x_values.min())
                        max_x = max(max_x, x_values.max())
                        min_y = min(min_y, y_values.min())
                        max_y = max(max_y, y_values.max())

                    # Annotate each point with the extracted number
                    if text_annotations_for_suites is not None:
                        if suite in text_annotations_for_suites:
                            for i, txt in enumerate(names):
                                num = re.search(r"([^\d]+)(\d+)_", txt).group(2)
                                ax.annotate(
                                    num,
                                    (x_values[i], y_values[i]),
                                    textcoords="offset points",
                                    xytext=(0, 5),
                                    ha="center",
                                )

            if zoom_to_suite is not None:
                # Set the zoom into the suite
                ax.set_xlim(
                    min_x - zoom_margin * (max_x - min_x),
                    max_x + zoom_margin * (max_x - min_x),
                )  # Add a margin
                ax.set_ylim(
                    min_y - zoom_margin * (max_y - min_y),
                    max_y + zoom_margin * (max_y - min_y),
                )

            ax.set_title(f"UMAP with n_neighbors = {n_neighbor}")

            # Grid
            ax.grid(True, which="major", linestyle="-", linewidth=0.75)
            ax.minorticks_on()
            ax.grid(True, which="minor", linestyle="--", linewidth=0.15)

        # Avoid repeated axis labels
        axes[np.floor(len(axes) / 2).astype(int)].set_xlabel("Component 1 [--]")
        axes[0].set_ylabel("Component 2 [--]")

        # Create legend for suites and dimensions.
        self.create_custom_legend_for_dimension(fig=fig, marker_dict=marker_dict)
        self.create_custom_legend_for_suite(
            fig=fig, df=df_filtered, ignore_aerofoils=ignore_aerofoils
        )

        plt.tight_layout()
        # Adjust the layout to make space for the top legend
        plt.subplots_adjust(top=0.85, bottom=0.15)
        plt.show()

    def train_random_forest(
        self, test_size=0.3, random_state=42, ignore_aerofoils=True
    ):
        """
        Train a Random Forest classifier to predict 'Suite' from the features.

        :param test_size: Fraction of the dataset to be used as test data.
        :param random_state: Seed used by the random number generator.
        """

        # Check data compatibility.
        self.check_dashboard_is_global_only_for_aerofoils(
            ignore_aerofoils=ignore_aerofoils
        )

        # Ensure that 'Suite' is one of the columns in the dataframe
        if "Suite" not in self.features_df.columns:
            raise ValueError("DataFrame must contain 'Suite' column")

        # Splitting the data into features and target variable
        df_filtered = self.custom_drop_na(
            self.filter_df_by_suite_and_dim(
                self.ignore_specified_features(self.features_df),
                suite_names=None,
                dims=None,
                ignore_aerofoils=ignore_aerofoils,
            )
        )

        X = self.get_numerical_data_from_df(df_filtered)
        y = df_filtered["Suite"]

        print(f"This RF model training has considered {len(X.columns)} features.")

        # Splitting the dataset into training and testing sets

        if test_size == 0:
            X_train = X
            X_test = X
            y_train = y
            y_test = y
            print("CAREFUL: all metrics are given on the full training set!")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

        # Initializing the Random Forest classifier
        self.classifier = RandomForestClassifier(random_state=random_state)

        # Training the classifier
        self.classifier.fit(X_train, y_train)

        # Making predictions on the test set
        y_pred = self.classifier.predict(X_test)

        # Evaluating the classifier
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Accuracy Score:", accuracy_score(y_test, y_pred))

    def get_feature_importances(self, top_features=None, worst=False):
        """
        Get a DataFrame of feature importances from the trained Random Forest model.

        :param n: Number of top features to return. If None, returns all features.
        :return: A DataFrame with feature importances.
        """
        if self.classifier is None:
            raise ValueError(
                "The classifier has not been trained yet. Call train_random_forest first."
            )

        importances = self.classifier.feature_importances_
        feature_names = self.get_feature_names(ignore_features=True)
        feature_importances = pd.DataFrame(
            importances, index=feature_names, columns=["Importance"]
        ).sort_values(by="Importance", ascending=False)

        if top_features is not None:

            if worst:
                return feature_importances.tail(top_features)
            else:
                return feature_importances.head(top_features)

        return feature_importances

    def plot_feature_importances(self, feature_importances_df):
        """
        Plot the feature importances in descending order.

        :param feature_importances_df: A DataFrame containing the feature importances.
        """
        # Sort the DataFrame if it's not sorted already
        feature_importances_df = feature_importances_df.sort_values(
            by="Importance", ascending=False
        )

        # Plotting
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x="Importance", y=feature_importances_df.index, data=feature_importances_df
        )
        plt.title("Feature Importances")
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.show()

    @staticmethod
    def generate_scatterplot_matrix(df, columns=None, color_by=None, show=False):
        """
        Generate a scatterplot matrix with correlation coefficients on the upper half.

        :param df: DataFrame containing the data.
        :param columns: List of columns to consider for the plot. If None, all columns are used.
        """
        if columns is not None:

            if color_by is not None and color_by not in columns:
                columns.append(color_by)

            df = df[columns]
        # else:
        #     # Select only numerical columns if no columns specified
        #     # df = df.select_dtypes(include=[np.number])

        # Initialize the pairplot
        g = sns.pairplot(df, hue=color_by)

        # Loop through the upper matrix to annotate with correlation coefficients
        df_num = df.select_dtypes(include=[np.number])
        for i, j in zip(*np.triu_indices_from(g.axes, 1)):
            corr_value = df_num.corr().iloc[i, j]
            g.axes[i, j].annotate(
                f"{corr_value:.2f}",
                (0.5, 0.5),
                xycoords="axes fraction",
                ha="center",
                va="center",
                fontsize=24,
            )

        if show:
            plt.show()

        return g

    def rank_each_feature(
        self,
        pca_var_expl=[0.7, 0.8, 0.9],
        rf_test_sizes=[0, 0.2, 0.5],
        coverage_suites=[None],
    ):

        # Initialise dataframe.
        feature_ranks = pd.DataFrame(index=self.get_feature_names(ignore_features=True))

        # Get sampling type.
        feature_ranks["Type"] = feature_ranks.index.to_series().apply(
            lambda x: (
                "Global"
                if "_glob" in x
                else ("RW" if "_rw" in x else ("AW" if "_aw" in x else "Other"))
            )
        )

        # Coverage rankings.
        for s in coverage_suites:
            print(s)
            coverages = self.get_feature_coverage(target_suite=s)
            coverages["mean_coverage"] = coverages.mean(axis=1)
            feature_ranks[f"Coverage_{s}"] = coverages["mean_coverage"].rank(
                ascending=True
            )

        # PCA rankings. Need to rerun PCA quickly.
        self.run_pca()
        for var in pca_var_expl:
            num_pcs = self.calc_n_principal_compons_for_var(minimum_expl_variance=var)
            best_pca_cont = self.get_total_pca_contribution(
                n_components=num_pcs, top_features=None, worst=False
            )
            feature_ranks[f"PCAVar{var*100:.0f}"] = best_pca_cont.rank(ascending=False)

        # RF classification model feature importance rankings.
        for t in rf_test_sizes:
            self.train_random_forest(test_size=t)
            best_rf_cont = self.get_feature_importances(top_features=None)
            feature_ranks[f"RFTest{t*100:.0f}"] = best_rf_cont.rank(ascending=False)

        # Compute means for each method.
        col_names = ["Coverage", "PCA", "RF"]
        for c in col_names:
            cols = feature_ranks.columns[feature_ranks.columns.str.contains(c)]
            feature_ranks[f"{c}Mean"] = feature_ranks[cols].mean(axis=1)

        # Calculate overall mean rank.
        feature_ranks["MeanRank"] = feature_ranks[[c + "Mean" for c in col_names]].mean(
            axis=1
        )

        # Calculate overall ranks.
        feature_ranks["RankOfRanks"] = feature_ranks["MeanRank"].rank(ascending=True)
        feature_ranks = feature_ranks.sort_values(by="RankOfRanks", ascending=True)

        return feature_ranks
