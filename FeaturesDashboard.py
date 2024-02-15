import pandas as pd
import os


class FeaturesDashboard:
    def __init__(self, folder_path):
        """
        Initialize the FeaturesDashboard with the path to the folder containing features.csv.
        :param folder_path: Path to the folder containing the features.csv file.
        """
        self.folder_path = folder_path
        self.overall_features_df = self.get_overall_features_df()

    def get_overall_features_df(self, give_sd=True):
        """
        Reads the features.csv file from the folder specified during initialization.
        Optionally removes columns with the suffix '_std' if give_sd is False.
        :param give_sd: If True, include standard deviation columns. If False, exclude them.
        :return: A pandas DataFrame containing the data from features.csv, optionally without _std columns.
        """
        # Construct the path to the CSV file
        file_path = os.path.join(self.folder_path, "features.csv")

        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # If give_sd is False, remove columns ending with '_std'
        if not give_sd:
            df = df[[col for col in df.columns if not col.endswith("_std")]]

        return df

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
        file_path = os.path.join(self.folder_path, f"{problem_name}_d{dim}", file_name)

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
