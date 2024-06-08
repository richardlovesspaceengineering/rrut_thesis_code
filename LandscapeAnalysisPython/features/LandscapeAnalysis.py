import numpy as np
import pandas as pd


class LandscapeAnalysis:
    """
    Collates all features for one sample.
    """

    def __init__(self, globalanalysis, randomwalkanalysis, adaptivewalkanalysis):
        """
        Give instances of MultipleGlobalAnalysis and MultipleRandomWalkAnalysis here.
        """
        self.analyses = {
            "glob": globalanalysis,
            "rw": randomwalkanalysis,
            "aw": adaptivewalkanalysis,
        }
        self.feature_arrays = {}
        self.aggregated_features = {}
        self.combine_all_feature_dicts()

    def append_to_features_dict(
        self, existing_features_dict, new_features_dict, method_suffix
    ):
        for feature_name, feature_value in new_features_dict.items():
            existing_features_dict[feature_name + "_" + method_suffix] = feature_value

        return existing_features_dict

    def combine_all_feature_dicts(self):
        new_dict = self.feature_arrays
        for suffix, a in self.analyses.items():
            if a:
                new_dict = self.append_to_features_dict(
                    new_dict, a.feature_arrays, suffix
                )

        # Save
        self.feature_arrays = new_dict

    def compute_statistic_for_feature(self, array, feature_name, stat="mean"):
        # Remove NaN values from the array and count them
        clean_array = np.array(array)[~np.isnan(array)]
        nan_count = np.count_nonzero(np.isnan(array))

        # Print the message about NaN removal
        if nan_count > 0:
            print(
                f"Removed {nan_count} NaN entries for feature '{feature_name}' when performing final aggregation."
            )

        # Compute the requested statistic on the array without NaNs
        if stat == "mean":
            return np.mean(clean_array)
        elif stat == "median":
            return np.median(clean_array)
        elif stat == "min":
            return np.min(clean_array)
        elif stat == "max":
            return np.max(clean_array)
        elif stat == "std":
            return np.std(clean_array)
        else:
            raise ValueError(
                "Invalid statistic choice. Use 'mean', 'median', 'min', 'max', or 'std'."
            )

    def aggregate_features(self):
        """
        Aggregate feature for all populations. Must be called after extract_feature_arrays.
        """

        for feature_name in self.feature_arrays:
            if feature_name in ["nrfbx"]:
                statistics = ["mean", "min", "max", "median"]
            else:
                statistics = ["mean", "std"]

            if isinstance(statistics, list):
                # Compute and set multiple statistics
                for stat in statistics:
                    # Replace feature name with more descriptive names.
                    new_name = f"{feature_name}_{stat}"

                    # Compute the statistic for the feature array
                    aggregated_value = self.compute_statistic_for_feature(
                        self.feature_arrays[feature_name],
                        feature_name,
                        stat,
                    )

                    # Store the aggregated value in the dictionary
                    self.aggregated_features[new_name] = aggregated_value

    def make_aggregated_feature_table(self, instance_name):
        """
        Create a 1-row table of all the features to allow comparison.
        """
        # Create a DataFrame with all features
        dat = pd.DataFrame(self.aggregated_features, index=[0])

        # Add problem name and number of dimensions
        dat.insert(0, "D", instance_name.split("d")[1])
        dat.insert(0, "Name", instance_name)

        return dat
