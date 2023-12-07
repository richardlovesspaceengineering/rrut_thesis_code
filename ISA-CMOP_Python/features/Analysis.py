import numpy as np
import time
import os
import os
from datetime import datetime
import pandas as pd
from features.feature_helpers import *


class Analysis:
    """
    Calculate all features generated from samples.
    """

    def __init__(self, normalisation_values):
        """
        Populations must already be evaluated.
        """
        self.normalisation_values = normalisation_values
        self.features = {}

    def eval_features(self, pop):
        pass


class MultipleAnalysis:
    """
    Aggregate features across populations/walks.
    """

    def __init__(self, single_sample_analyses, normalisation_values):
        self.analyses = single_sample_analyses
        self.normalisation_values = normalisation_values
        self.feature_arrays = {}

    def extract_features_from_all_populations(self):
        """
        Evaluate features for all populations.
        """

        cls_name = self.__class__.__name__
        if cls_name == "MultipleGlobalAnalysis":
            s1 = "Global"
            s2 = "sample"
        elif cls_name == "MultipleRandomWalkAnalysis":
            s1 = "RW"
            s2 = "walk"
        elif cls_name == "MultipleAdaptiveWalkAnalysis":
            s1 = "AW"
            s2 = "walk"

        for ctr, a in enumerate(self.analyses):
            a.eval_features()

            print(
                "Evaluated {} features for {} {} out of {} in {:.2f} seconds.".format(
                    s1, s2, ctr + 1, len(self.analyses), elapsed_time
                )
            )

        # Generate corresponding arrays.
        self.generate_feature_arrays()

    def generate_array_for_feature(self, feature_name):
        feature_array = []
        for analysis in self.analyses:
            feature_array.append(analysis.features[feature_name])
        return np.array(feature_array)

    def generate_feature_arrays(self):
        """
        Collate features into an array. Must be run after eval_features_for_all_populations.
        """
        # Loop over feature names from dictionary.
        for feature_name in self.analyses[0].features:
            self.feature_arrays[feature_name] = self.generate_array_for_feature(
                feature_name
            )

    def export_unaggregated_features(
        self, instance_name, method_suffix, save_arrays, export_norm=True
    ):
        """
        Write dictionary of raw features results to a csv file.

        Also has an option to export the normalisation values used in computing the features.
        """

        if save_arrays:
            # Create a folder if it doesn't exist
            results_folder = "instance_results"
            if not os.path.exists(results_folder):
                os.makedirs(results_folder)

            # Get the current date and time
            current_time = datetime.now().strftime("%b%d_%H%M")

            # Create the file path
            file_path = os.path.join(
                results_folder,
                f"{instance_name}_{method_suffix}_features_{current_time}.csv",
            )

            # Create a dataframe and save to a CSV file
            dat = pd.DataFrame({k: list(v) for k, v in self.feature_arrays.items()})
            dat.to_csv(file_path, index=False)
            print(
                "\nSuccessfully saved {} sample results to csv file for {}.\n".format(
                    method_suffix, instance_name
                )
            )

            # Export normalisation values.
            if export_norm:
                # Now normalisation values are listed per dimension rather than arrays.
                flattened_normalisation_dict = flatten_dict(self.normalisation_values)
                norm_dat = pd.DataFrame(flattened_normalisation_dict, index=[0])
                # Create the file path
                norm_file_path = os.path.join(
                    results_folder,
                    f"{instance_name}_{method_suffix}_norm_{current_time}.csv",
                )
                norm_dat.to_csv(norm_file_path, index=False)

        else:
            print("\nSaving of feature arrays for {} disabled.\n".format(method_suffix))

    @staticmethod
    def concatenate_multiple_analyses(multiple_analyses, AnalysisType):
        """
        Concatenate feature arrays from multiple MultipleAnalysis objects into one.
        """

        # Determine the type of the first element in multiple_analyses
        first_analysis = multiple_analyses[0]

        # Create a new MultipleAnalysis object with the combined populations based on the type of the first element
        if isinstance(first_analysis, AnalysisType):
            combined_analysis = AnalysisType(
                [], [], first_analysis.normalisation_values
            )  # TODO: update to work with global features too.

        # Extract the feature names too.
        feature_names = first_analysis.feature_arrays.keys()

        # Iterate through feature names
        for feature_name in feature_names:
            feature_arrays = [
                ma.feature_arrays[feature_name] for ma in multiple_analyses
            ]
            combined_array = np.concatenate(feature_arrays)

            # Save as attribute array in the combined_analysis object
            combined_analysis.feature_arrays[feature_name] = combined_array

        return combined_analysis
