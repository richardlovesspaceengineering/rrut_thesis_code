import numpy as np
from features.randomwalkfeatures import randomwalkfeatures
from scipy.stats import yeojohnson


class Analysis:
    """
    Calculate all features generated from samples.
    """

    def __init__(self, pop):
        """
        Populations must already be evaluated.
        """
        self.pop = pop
        self.pareto_front = pop[0].pareto_front

    def eval_features(self):
        pass


class MultipleAnalysis:
    """
    Aggregate features across populations/walks.
    """

    def __init__(self, pops, AnalysisType):
        self.pops = pops
        self.analyses = []
        for pop in pops:
            self.analyses.append(AnalysisType(pop))
        self.feature_names = []

    def initialize_arrays_and_scalars(self):
        # Initialising feature arrays.
        for feature in self.feature_names:
            setattr(
                self,
                (f"{feature}_array"),
                np.empty(len(self.pops)),
            )

        # Initialising feature values.
        for feature in self.feature_names:
            setattr(
                self,
                (f"{feature}"),
                np.nan,
            )

    def eval_features_for_all_populations(self):
        """
        Evaluate features for all populations.
        """

        for ctr, a in enumerate(self.analyses):
            a.eval_features()

            # TODO: make string specific to subclass.
            print(
                "Evaluated features for population {} of {}".format(
                    ctr + 1, len(self.analyses)
                )
            )

        # Generate corresponding arrays.
        self.generate_feature_arrays()

    def generate_array_for_feature(self, feature_name):
        feature_array = []
        for analysis in self.analyses:
            feature_array.append(getattr(analysis, feature_name))
        return np.array(feature_array)

    def generate_feature_arrays(self):
        """
        Collate features into an array. Must be run after eval_features_for_all_populations.
        """
        for feature_name in self.feature_names:
            setattr(
                self,
                (f"{feature_name}_array"),
                self.generate_array_for_feature(feature_name),
            )

    def apply_YJ_transform(self, array):
        return yeojohnson(array)[0]

    def aggregate_array_for_feature(self, array, YJ_transform=True):
        if YJ_transform:
            return np.mean(self.apply_YJ_transform(array))
        else:
            return np.mean(array)

    def aggregate_features(self, YJ_transform=True):
        """
        Aggregate feature for all populations.
        """
        for feature_name in self.feature_names:
            setattr(
                self,
                feature_name,
                self.aggregate_array_for_feature(
                    getattr(self, f"{feature_name}_array")
                ),
            )
