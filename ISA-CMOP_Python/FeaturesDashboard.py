import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shutil
from matplotlib.backends.backend_pdf import PdfPages


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
            "DASCMOP": [f"DASCMOP{x}" for x in range(1, 10, 1)],
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
        self.features_df = self.get_landscape_features_df(give_sd=True)

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
            suite_names=suite_names, dims=dims
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

        plt.tight_layout()
        plt.show()

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
        df_filtered = self.filter_df_by_suite_and_dim(
            suite_names=suite_names, dims=dims
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
                self.plot_feature_across_suites(feature_name, suite_names, dims)
                pdf.savefig()  # saves the current figure into a pdf page
                plt.close()  # close the figure to prevent it from being displayed

    def filter_df_by_suite_and_dim(self, suite_names=None, dims=None):
        """
        Filters the features DataFrame based on specified suite names and dimensions.
        :param suite_names: List of suite names to filter by. If None, no suite-based filtering is applied.
        :param dims: List of dimensions to filter by. If None, no dimension-based filtering is applied.
        :return: A filtered DataFrame based on the specified suite names and dimensions.
        """
        # Start with the full DataFrame
        df_filtered = self.features_df.copy()

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

        filtered_df = self.filter_df_by_suite_and_dim(suite_names, dims)

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
