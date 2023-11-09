import numpy as np
import time


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
        self.feature_names = []

    def __setattr__(self, name, value):
        # Call the parent class's __setattr__ to perform the actual assignment
        super().__setattr__(name, value)

        # Add the variable name to the feature_names list. This will only run when a feature (a float) is passed
        if (
            name not in ["pop", "pareto_front", "feature_names", "pop_neighbours_list"]
            and name not in self.feature_names
            and isinstance(value, (float, int))
        ):
            self.feature_names.append(name)

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

    def initialize_arrays(self):
        # Initialising feature arrays.
        for feature in self.feature_names:
            setattr(
                self,
                (f"{feature}_array"),
                np.empty(len(self.pops)),
            )

    def eval_features_for_all_populations(self):
        """
        Evaluate features for all populations.
        """

        cls_name = self.__class__.__name__
        if cls_name == "MultipleGlobalAnalysis":
            s = "Global"
        elif cls_name == "MultipleRandomWalkAnalysis":
            s = "RW"

        print("\nInitialising feature evaluation for {} features.".format(s))
        for ctr, a in enumerate(self.analyses):
            start_time = time.time()
            a.eval_features()

            # Once evaluated, save feature names and initialise arrays for each feature.
            if ctr == 0:
                self.feature_names = a.feature_names
                self.initialize_arrays()
                # print(self.feature_names)

            end_time = time.time()  # Record the end time
            elapsed_time = end_time - start_time
            print(
                "Evaluated {} features for sample {} out of {} in {:.2f} seconds.".format(
                    s, ctr + 1, len(self.analyses), elapsed_time
                )
            )

        print("\nEvaluated all {} features\n".format(s))

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
