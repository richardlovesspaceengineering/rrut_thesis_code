import pandas as pd
import os


class FeaturesDashboard:
    def __init__(self, features_path, algo_perf_path):
        """
        Initialize the FeaturesDashboard with paths to the directories containing features.csv and algo_performance.csv.
        :param features_path: Path to the folder containing the features.csv file.
        :param algo_perf_path: Path to the folder containing the algo_performance.csv file.
        """
        self.features_path = features_path
        self.algo_perf_path = algo_perf_path
        self.overall_features_df = self.get_overall_features_df()

    def get_overall_features_df(self, give_sd=True):
        """
        Reads the features.csv and algo_performance.csv files from their respective directories
        and joins them into a single DataFrame, resolving any 'D' column duplication.
        :return: A pandas DataFrame containing the joined data with a single 'D' column.
        """
        # Construct the paths to the CSV files
        features_file_path = os.path.join(self.features_path, "features.csv")
        algo_perf_file_path = os.path.join(self.algo_perf_path, "algo_performance.csv")

        # Check if the files exist
        if not os.path.exists(features_file_path):
            raise FileNotFoundError(f"The file {features_file_path} does not exist.")
        if not os.path.exists(algo_perf_file_path):
            raise FileNotFoundError(f"The file {algo_perf_file_path} does not exist.")

        # Read the CSV files into DataFrames
        features_df = pd.read_csv(features_file_path)
        algo_perf_df = pd.read_csv(algo_perf_file_path)

        # Join the DataFrames, specifying suffixes for overlapping column names other than the join key
        overall_df = pd.merge(
            features_df, algo_perf_df, on="Name", how="inner", suffixes=("", "_drop")
        )

        # Drop the redundant 'D' column from the right DataFrame (algo_performance.csv)
        # and any other unwanted duplicate columns that were suffixed with '_drop'
        overall_df.drop(
            [col for col in overall_df.columns if "drop" in col], axis=1, inplace=True
        )

        # If give_sd is False, remove columns ending with '_std'
        if not give_sd:
            overall_df = overall_df[
                [col for col in overall_df.columns if not col.endswith("_std")]
            ]

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
