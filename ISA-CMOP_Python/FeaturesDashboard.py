import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shutil


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
            "DASCMOP": [f"CTP{x}" for x in range(1, 10, 1)],
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
        }

        # Process the results_dict to populate features_path_list
        missing_folders = []
        for folders in results_dict.values():
            if folders is not None:
                for folder in folders:
                    full_path = os.path.join(self.base_results_dir, folder)
                    # Check if the directory exists before appending
                    if os.path.isdir(full_path):
                        self.features_path_list.append(full_path)
                    else:

                        # Ensures warning only gets printed once.
                        if folder not in missing_folders:
                            missing_folders.append(folder)
                            print(f"Warning: The directory {full_path} does not exist.")

        # Collate all results into one folder.
        self.copy_directory_contents()
        self.features_df = self.get_landscape_features_df(give_sd=True)

    def get_landscape_features_df(self, give_sd):
        """
        Collates features_{timestamp}.csv files that have been copied to self.new_save_path,
        adding an additional column to indicate the source timestamp.
        """
        # Initialize an empty list to store dataframes
        df_list = []

        # Iterate over all files in the new_save_path
        for filename in os.listdir(self.new_save_path):
            # Check if the file matches the pattern of features_{timestamp}.csv
            if (
                filename.startswith("features_")
                and filename.endswith(".csv")
                and "collated" not in filename  # avoid circular import
            ):
                # Construct the full path to the file
                file_path = os.path.join(self.new_save_path, filename)

                # Apply any specific processing function if required
                # For example, you mentioned `get_df_from_filepath` which is not defined here.
                # Assuming it's a function that processes the DataFrame.
                temp_df = get_df_from_filepath(file_path, give_sd)

                # Extract the timestamp from the filename
                timestamp = filename.replace("features_", "").replace(".csv", "")

                # Add the timestamp as a new column in the dataframe
                temp_df["Date"] = timestamp

                # Ensure 'Date' column is the second column
                cols = temp_df.columns.tolist()
                # Move 'Date' to the second position (index 1) assuming the first column is something you want to keep at the start
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
                "No features_{timestamp}.csv files found in the new save path."
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

    def copy_directory_contents(self):
        """
        Copies the contents of directories from self.features_path_list into self.new_save_path.
        Each item (file or directory) will have a timestamp derived from its source directory appended to its name to ensure uniqueness.
        """
        # Ensure the new_save_path exists
        os.makedirs(self.new_save_path, exist_ok=True)

        for src_dir in self.features_path_list:
            if os.path.isdir(src_dir):
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
                new_filename = f"{file_base}{file_extension}"
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
        file_name = f"{problem_name}_d{dim}_{analysis_type}_features.csv"
        file_path = os.path.join(folder_path, file_name)

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

    def get_features_for_suite(self, df, suite_name, dim=None):
        # Ensure suite_name is valid and in the suites_problems_dict
        if suite_name not in self.suites_problems_dict:
            raise ValueError(
                f"Invalid suite '{suite_name}'. Please check the suite name and try again."
            )

        # Retrieve the list of problems for the suite
        problems_list = self.suites_problems_dict[suite_name]

        # If dimension is specified, refine the problems list to include only those with the specified dimension
        if dim:
            problems_list = [f"{problem}_d{dim}" for problem in problems_list]

        # Use the refined problems_list to filter rows in the dataframe
        pattern = "|".join(
            problems_list
        )  # Creates a regex pattern that matches any of the problem names
        df_filtered = df[df["Name"].str.contains(pattern, na=False, regex=True)]

        return df_filtered

    def compute_global_maxmin(self, feature_name_with_mean, suite_names, dims):

        # Determine the global y-axis limits across all suites and dimensions
        global_min, global_max = float("inf"), -float("inf")
        for suite_name in suite_names:
            for dim in dims if dims else [None]:
                df_filtered = self.get_features_for_suite(
                    self.features_df, suite_name, dim
                )
                if feature_name_with_mean in df_filtered.columns:
                    current_min = df_filtered[feature_name_with_mean].min()
                    current_max = df_filtered[feature_name_with_mean].max()
                    global_min = min(global_min, current_min)
                    global_max = max(global_max, current_max)

        # Add a buffer to the global y-axis limits
        buffer_percent = 0.05  # For example, 5% buffer
        buffer = (global_max - global_min) * buffer_percent
        global_min -= buffer
        global_max += buffer

        return global_min, global_max

    def plot_feature_across_suites(self, feature_name, suite_names=None, dims=None):
        """
        Generates a 1xN grid of violin plots for a specified feature across different benchmark suites, with points overlaying the violin plots.
        A single violin plot is shown for all dimensions, with different marker types used for each dimension, all in violet color. A legend is included.
        :param feature_name: The name of the feature to plot. Can be a landscape feature or algo performance.
        :param suite_names: A list of benchmark suite names.
        :param dims: Optional. A list of dimensions to filter the features by and to differentiate in the plot with markers.
        """
        plt.figure(figsize=(15, 6))
        feature_name_with_mean = feature_name + "_mean"

        # Define marker types for different dimensions.
        dim_colors = {
            5: "#1f77b4",
            10: "#ff7f0e",
            20: "#2ca02c",
            30: "#d62728",
        }
        marker_type = "o"  # Consistent color for all markers

        if not suite_names:
            suite_names = self.suites_problems_dict.keys()

        if not dims:
            dims = [5, 10, 20, 30]  # Default dimensions if none provided

        global_min, global_max = self.compute_global_maxmin(
            feature_name_with_mean, suite_names, dims
        )

        for i, suite_name in enumerate(suite_names, start=1):
            ax = plt.subplot(1, len(suite_names), i)
            combined_df = pd.DataFrame()  # Initialize empty DataFrame for combined data

            # Combine data across dimensions
            for dim in dims:
                df_filtered = self.get_features_for_suite(
                    self.features_df, suite_name, dim
                )
                if feature_name_with_mean in df_filtered.columns:
                    combined_df = pd.concat(
                        [combined_df, df_filtered], ignore_index=True
                    )

            # Plot a single violin plot for the combined data
            if not combined_df.empty:
                sns.violinplot(y=combined_df[feature_name_with_mean], color="lightgrey")

            # Overlay points for each dimension
            for dim in dims:
                df_filtered = self.get_features_for_suite(
                    self.features_df, suite_name, dim
                )
                if feature_name_with_mean in df_filtered.columns:
                    sns.stripplot(
                        y=df_filtered[feature_name_with_mean],
                        marker=marker_type,
                        color=dim_colors[dim],
                        label=f"{dim}D",
                        size=5,
                        alpha=0.5,
                    )

            plt.title(suite_name)
            plt.ylabel(feature_name_with_mean if i == 1 else "")
            plt.xlabel("")

            # Only add legend to the first subplot
            if i == 1:
                plt.legend()

            else:
                ax.get_legend().remove()

            # Set the y-axis limits with buffer for the subplot
            ax.set_ylim(global_min, global_max)

        plt.tight_layout()
        plt.show()

    def get_features_for_dim(self, df, dim, suite_name=None):
        """
        Filters the DataFrame for features related to a specified dimension. Optionally,
        further refines the filtering to include only problems from a specified benchmark suite.
        :param df: The DataFrame containing features data.
        :param dim: The dimension to filter the features by.
        :param suite_name: Optional. A specific benchmark suite name to further refine the filtering.
        :return: A filtered DataFrame based on the specified dimension, and optionally, the suite name.
        """
        # Filter the DataFrame based on the specified dimension.
        pattern_dim = f"_d{dim}"
        df_filtered = df[df["Name"].str.contains(pattern_dim, na=False, regex=True)]

        # If suite_name is specified, further refine the filtering.
        if suite_name:
            # Ensure suite_name is valid and in the suites_problems_dict
            if suite_name not in self.suites_problems_dict:
                raise ValueError(
                    f"Invalid suite '{suite_name}'. Please check the suite name and try again."
                )
            # Retrieve the list of problems for the suite
            problems_list = self.suites_problems_dict[suite_name]
            # Use the problems list to filter rows in the dataframe
            pattern_suite = "|".join(
                problems_list
            )  # Creates a regex pattern that matches any of the problem names
            # Ensure only rows matching both the dimension and suite are included
            df_filtered = df_filtered[
                df_filtered["Name"].str.contains(pattern_suite, na=False, regex=True)
            ]

        return df_filtered

    def plot_feature_across_dims(self, feature_name, dims=None, suite_names=None):
        """
        Generates a 1xN grid of violin plots for a specified feature across different dimensions, with points overlaying the violin plots.
        Each violin plot represents a different dimension, with distinct colors used for each dimension.
        :param feature_name: The name of the feature to plot. Can be a landscape feature or algo performance.
        :param dims: A list of dimensions to plot.
        :param suite_names: Optional. A list of benchmark suite names to filter the features by.
        """
        plt.figure(figsize=(15, 6))
        feature_name_with_mean = feature_name + "_mean"

        # Define colors for different dimensions.
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

        global_min, global_max = self.compute_global_maxmin(
            feature_name_with_mean, suite_names, dims
        )

        for i, dim in enumerate(dims, start=1):
            ax = plt.subplot(1, len(dims), i)
            combined_df = pd.DataFrame()  # Initialize empty DataFrame for combined data

            # Combine data across suites
            for suite_name in suite_names:
                df_filtered = self.get_features_for_dim(
                    self.features_df, dim, suite_name
                )
                if feature_name_with_mean in df_filtered.columns:
                    combined_df = pd.concat(
                        [combined_df, df_filtered], ignore_index=True
                    )

            # Plot a single violin plot for the combined data
            if not combined_df.empty:
                sns.violinplot(
                    y=combined_df[feature_name_with_mean],
                    color="lightgrey",
                )
                # Annotate number of points
                num_points = combined_df.shape[0]  # Calculate number of points
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
                df_filtered = self.get_features_for_dim(
                    self.features_df, dim, suite_name
                )
                if feature_name_with_mean in df_filtered.columns:
                    sns.stripplot(
                        y=df_filtered[feature_name_with_mean],
                        color=suite_colors[suite_name],
                        marker=marker_type,
                        label=suite_name,
                        size=5,
                        alpha=0.5,
                        jitter=True,
                    )

            plt.title(f"{dim}D")
            plt.ylabel(feature_name_with_mean if i == 1 else "")
            plt.xlabel("")

            # Only add legend to the first subplot
            if i == 1:
                plt.legend()
            else:
                ax.get_legend().remove()

            # Set the y-axis limits with buffer for the subplot
            ax.set_ylim(global_min, global_max)

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
