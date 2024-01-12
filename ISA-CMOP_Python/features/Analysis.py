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

    def __init__(self, normalisation_values, results_dir):
        """
        Populations must already be evaluated.
        """
        self.normalisation_values = normalisation_values
        self.features = {}
        self.results_dir = results_dir

    def eval_features(self, pop):
        pass

    @staticmethod
    def concatenate_single_analyses(single_analyses):
        """
        Concatenate feature values from multiple Analysis objects into one.

        Used when generating a single sample feature set for random/adaptive walks, where a sample contains multiple independent walks.
        """

        # Determine the class type of the first element in single_analyses
        analysis_type = type(single_analyses[0])

        # Create a new MultipleAnalysis object with the combined populations based on the inferred class type
        combined_analysis = analysis_type(
            single_analyses[0].normalisation_values, single_analyses[0].results_dir
        )

        # Extract the feature names
        feature_names = single_analyses[0].features.keys()

        # Iterate through feature names
        for feature_name in feature_names:
            feature_values = [sa.features[feature_name] for sa in single_analyses]
            # Remove NaN values
            clean_feature_values = [
                value for value in feature_values if not np.isnan(value)
            ]
            # Count the NaN entries that were removed
            nan_count = len(feature_values) - len(clean_feature_values)

            # Take the average of the non-NaN values
            if clean_feature_values:  # Check if the list is not empty
                combined_feature_value = np.mean(clean_feature_values)
            else:
                combined_feature_value = (
                    np.nan
                )  # or some other placeholder for features with all NaN values

            # Print how many NaN entries were removed
            if nan_count > 0:
                print(
                    f"Removed {nan_count} NaN entries (out of {len(feature_values)}) for {feature_name} when concatenating within-sample feature arrays."
                )

            # Save as attribute array in the combined_analysis object
            combined_analysis.features[feature_name] = combined_feature_value

        return combined_analysis


class MultipleAnalysis:
    """
    Aggregate features across populations/walks.
    """

    def __init__(self, single_sample_analyses, normalisation_values):
        self.analyses = single_sample_analyses
        self.normalisation_values = normalisation_values
        self.feature_arrays = {}
        self.results_dir = single_sample_analyses[0].results_dir

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
            # Create the results folder if it doesn't exist
            results_folder = os.path.join(self.results_dir, instance_name)
            if not os.path.exists(results_folder):
                os.makedirs(results_folder)

            # Create the file path without the current time appended
            file_path = os.path.join(
                results_folder,
                f"{instance_name}_{method_suffix}_features.csv",
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
                # Create the file path without the current time appended
                norm_file_path = os.path.join(
                    results_folder,
                    f"{instance_name}_{method_suffix}_norm.csv",
                )
                norm_dat.to_csv(norm_file_path, index=False)

        else:
            print("\nSaving of feature arrays for {} disabled.\n".format(method_suffix))
