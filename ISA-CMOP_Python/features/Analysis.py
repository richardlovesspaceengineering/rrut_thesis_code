import numpy as np
import time
import os
import os
from datetime import datetime
import pandas as pd
import numpy as np
import copy
import warnings
import scipy.stats
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from collections.abc import Iterable
from optimisation.model.population import Population

import datetime


def flatten_list(nested_list):
    """
    Recursive function to flatten all lists and tuples.
    """
    result = []
    for item in nested_list:
        if isinstance(item, (list, tuple)):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result


def flatten_dict(original_dict):
    flattened_dict = {}

    for key, values in original_dict.items():
        if not isinstance(values, Iterable) or isinstance(values, str):
            values = [values]  # Convert scalar to a list

        for i, value in enumerate(values, start=1):
            if "_" in key:
                prefix, suffix = key.rsplit("_", 1)
                flattened_key = f"{prefix}{i}_{suffix}"
            else:
                flattened_key = f"{key}{i}"

            flattened_dict[flattened_key] = value

    return flattened_dict


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

    def eval_features(self):
        pass

    @staticmethod
    def compute_solver_crash_ratio(full_pop, trimmed_pop):
        obj_full = full_pop.extract_obj()
        obj_trimmed = trimmed_pop.extract_obj()

        scr = 1 - obj_trimmed.shape[0] / obj_full.shape[0]

        return scr

    @staticmethod
    def trim_obj_using_nadir(obj, nadir):
        # Create a boolean mask
        mask = np.all(obj <= nadir, axis=1)

        # Use the mask to select the rows from obj
        trimmed_obj = obj[mask]

        return trimmed_obj, mask

    @staticmethod
    def fit_linear_mdl(xdata, ydata):
        # Fit linear model to xdata and ydata.
        mdl = LinearRegression().fit(xdata, ydata)

        # R2 (adjusted) has to be computed from the unadjusted value.
        num_obs = ydata.shape[0]
        num_coef = xdata.shape[1]
        r2_unadj = mdl.score(xdata, ydata)
        mdl_r2 = 1 - (1 - r2_unadj) * (num_obs - 1) / (num_obs - num_coef - 1)

        # Range of magnitudes. Ignore the intercepts.
        range_coeff = np.abs(np.max(mdl.coef_)) - np.abs(np.min(mdl.coef_))

        return mdl_r2, range_coeff

    @staticmethod
    def remove_imag_rows(matrix):
        """
        Remove rows which have at least one imaginary value in them
        """

        new_matrix = copy.deepcopy(matrix)
        rmimg = np.sum(np.imag(new_matrix) != 0, axis=1)
        new_matrix = new_matrix[np.logical_not(rmimg)]
        return new_matrix

    @staticmethod
    def generate_bounds_from_problem(problem_instance):
        # Bounds of the varision variables.
        x_lower = problem_instance.xl
        x_upper = problem_instance.xu
        bounds = np.vstack((x_lower, x_upper))
        return bounds

    @staticmethod
    def corr_coef(xdata, ydata, spearman=True, significance_level=0.05):
        """
        Get correlation coefficient and pvalue, suppressing warnings when a constant vector is input.
        """
        with warnings.catch_warnings():
            # Suppress warnings where corr is NaN - will just set to 0 in this case.
            warnings.simplefilter("ignore", scipy.stats.ConstantInputWarning)

            # Method for computing correlation.
            if spearman:
                method = scipy.stats.spearmanr
            else:
                method = scipy.stats.pearsonr

            # Ensure shapes are compatible. Should be okay to squeeze because xdata and ydata will always be vectors.
            result = method(np.squeeze(xdata), np.squeeze(ydata))
            corr = result.statistic
            pvalue = result.pvalue

            # Signficance test. Null hypothesis is samples are uncorrelated.
            if pvalue > significance_level:
                corr = 0

            elif np.isnan(corr):
                # Make correlation 0 if there is no change in one vector.
                corr = 0

        return corr

    @staticmethod
    def autocorr(data, lag, spearman=True, significance_level=0.05):
        """
        Compute autocorrelation of data with applied lag.
        """
        return Analysis.corr_coef(data[:-lag], data[lag:], spearman, significance_level)

    @staticmethod
    def compute_correlation_matrix(matrix, correlation_type, alpha=0.05):
        """
        Compute the correlation matrix of a given square matrix and trim based on significance.

        Note that computed p-values are only valid for > 500 observations - otherwise a parametric test for significance should be done.

        Parameters:
        - matrix: 2D numpy array, square matrix
        - correlation_type: str, either 'pearson' or 'spearman'
        - alpha: float, significance level

        Returns:
        - correlation_matrix: 2D numpy array, correlation matrix with trimmed values
        - significance_matrix: 2D numpy array, matrix indicating significance (True/False)
        """

        if correlation_type not in ["pearson", "spearman"]:
            raise ValueError("Invalid correlation type. Use 'pearson' or 'spearman'.")
        if correlation_type == "pearson":
            correlation_matrix, p_values = pearsonr(matrix.T)
        elif correlation_type == "spearman":
            correlation_matrix, p_values = spearmanr(matrix, axis=0)

        if correlation_matrix.ndim == 0:  # If the result is a scalar (2x2 matrix case)
            correlation_matrix = np.array(
                [[1, correlation_matrix], [correlation_matrix, 1]]
            )
            p_values = np.array([[1, p_values], [p_values, 1]])

        significance_matrix = p_values > alpha

        # Trim values based on significance
        correlation_matrix[significance_matrix] = 0

        return correlation_matrix, p_values

    @staticmethod
    def apply_normalisation(var, fmin, fmax, scale=1):
        if np.all(fmax == fmin):
            return var
        else:
            return (var - fmin) / ((fmax - fmin) * scale)

    # TODO: move this
    def normalise_objective_space_for_hv_calc(
        pop_walk, pop_neighbours, PF, scale_offset=1.1, region_of_interest=False
    ):
        """
        Normalise all objectives for HV calculation.

        If computing population HV values, set region_of_interest to True to ensure objectives lie in the region of interest from Vodopija2023.

        If computing neighbourhood HV values, set region_of_interest to False as neighbours generally do not fall in the region of interest.

        Scale offset 1.1 is equivalent to using a nadir of (1.1,1.1,...)
        """

        # Merge walk objectives and neighbourhood objectives into one matrix.
        merged_obj = pop_walk.extract_obj()
        for pop_neighbourhood in pop_neighbours:
            merged_obj = np.vstack((merged_obj, pop_neighbourhood.extract_obj()))

        fmin = np.minimum(np.min(merged_obj, axis=0), np.min(PF, axis=0))

        if region_of_interest:
            fmax = np.max(PF, axis=0)
        else:
            fmax = np.maximum(np.max(PF, axis=0), np.max(merged_obj, axis=0))

        # Create copies to save these new objectives to.
        pop_walk_normalised = copy.deepcopy(pop_walk)

        # Normalise walk objectives and update the population.
        obj_walk_normalised = Analysis.apply_normalisation(
            pop_walk.extract_obj(), fmin, fmax, scale=scale_offset
        )
        pop_walk_normalised.set_obj(obj_walk_normalised)

        # Normalise neighbourhood objectives and update.
        pop_neighbours_normalised = []
        for pop_neighbourhood in pop_neighbours:
            obj_neighbourhood_normalised = Analysis.apply_normalisation(
                pop_neighbourhood.extract_obj(), fmin, fmax, scale=scale_offset
            )
            pop_neighbourhood_normalised = copy.deepcopy(pop_neighbourhood)
            pop_neighbourhood_normalised.set_obj(obj_neighbourhood_normalised)
            pop_neighbours_normalised.append(pop_neighbourhood_normalised)

        # Normalise PF.
        PF_normalised = Analysis.apply_normalisation(PF, fmin, fmax, scale=scale_offset)

        # To keep us in the region of interest, remove any objectives larger than the nadir.
        if region_of_interest:
            obj_normalised = obj_normalised[~np.any(obj_normalised > 1, axis=1)]

        return pop_walk_normalised, pop_neighbours_normalised, PF_normalised

    @staticmethod
    def use_no_normalisation(n_var, n_obj):
        normalisation_values = {}
        variables = ["var", "obj", "cv"]

        for which_variable in variables:
            if which_variable == "var":
                fmin = np.zeros(n_var)
                fmax = np.ones(n_var)
            elif which_variable == "obj":
                fmin = np.zeros(n_obj)
                fmax = np.ones(n_obj)
            else:
                fmin = 0
                fmax = 1

            normalisation_values[f"{which_variable}_min"] = fmin
            normalisation_values[f"{which_variable}_max"] = fmax

            # For f95th, use fmax
            normalisation_values[f"{which_variable}_95th"] = fmax

        return normalisation_values

    def extract_norm_values(self, norm_method):
        """
        Extract normalisation values from the dictionary formed using compute_all_normalisation_values
        """

        # Extract normalisation values.
        if norm_method == "maximin":
            s_lb = "_min"
            s_ub = "_max"
        elif norm_method == "95th":
            s_lb = "_min"
            s_ub = "_95th"
        else:
            # If norm_method is None, set all lbs to 0 and ubs to 1, since this is equivalent to not applying normalisation in Analysis.apply_normalisation.
            var_lb = obj_lb = cv_lb = 0
            var_ub = obj_ub = cv_ub = 1
            return var_lb, var_ub, obj_lb, obj_ub, cv_lb, cv_ub

        var_lb = self.normalisation_values["var" + s_lb]
        var_ub = self.normalisation_values["var" + s_ub]
        obj_lb = self.normalisation_values["obj" + s_lb]
        obj_ub = self.normalisation_values["obj" + s_ub]
        cv_lb = self.normalisation_values["cv" + s_lb]
        cv_ub = self.normalisation_values["cv" + s_ub]

        return var_lb, var_ub, obj_lb, obj_ub, cv_lb, cv_ub

    @staticmethod
    def combine_arrays_for_pops(pop_list, which_variable):
        if which_variable == "var":
            method = (
                lambda x: x.extract_var()
            )  # Lambda function to extract_var from each object
        elif which_variable == "obj":
            method = (
                lambda x: x.extract_obj()
            )  # Lambda function to extract_obj from each object
        elif which_variable == "cv":
            method = (
                lambda x: x.extract_cv()
            )  # Lambda function to extract_cv from each object

        array_list = [method(pop) for pop in pop_list]
        combined_array = np.vstack(array_list)

        return combined_array

    @staticmethod
    def concatenate_single_analyses(single_analyses):
        """
        Concatenate feature values from multiple Analysis objects into one.

        Used when generating a single sample feature set for random/adaptive walks, where a sample contains multiple independent walks.
        """

        # Check if single_analyses is not a list (i.e., a single element), and wrap it in a list
        if not isinstance(single_analyses, list):
            single_analyses = [single_analyses]

        # Determine the class type of the first element in single_analyses
        analysis_type = type(single_analyses[0])

        # Create a new MultipleAnalysis object with the combined populations based on the inferred class type
        combined_analysis = analysis_type(
            None,
            single_analyses[0].normalisation_values,
            single_analyses[0].results_dir,
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

            # Print how man y NaN entries were removed
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
