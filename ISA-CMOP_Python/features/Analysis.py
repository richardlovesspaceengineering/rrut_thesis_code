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

    def generate_array_for_attribute(self, attribute_name):
        attribute_array = []
        for analysis in self.analyses:
            attribute_array.append(getattr(analysis, attribute_name))
        return np.array(attribute_array)

    def generate_feature_arrays(self):
        pass

    def apply_YJ_transform(self, array):
        return yeojohnson(array)[0]

    def aggregate_feature_array(self, array, YJ_transform=True):
        if YJ_transform:
            return np.mean(self.apply_YJ_transform(array))
        else:
            return np.mean(array)

    def aggregate_features_from_names_list(self, attribute_names, YJ_transform=True):
        """
        Aggregate feature for all populations.
        """
        for attribute_name in attribute_names:
            setattr(
                self,
                attribute_name,
                self.aggregate_feature_array(getattr(self, f"{attribute_name}_array")),
            )
