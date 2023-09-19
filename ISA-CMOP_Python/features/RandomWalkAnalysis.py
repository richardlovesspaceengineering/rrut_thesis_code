import numpy as np
from features.randomwalkfeatures import randomwalkfeatures


class RandomWalkAnalysis:
    """
    Calculate all features generated from random walk samples.

    Populations is a list of populations that represents a walk, each entry is a solution and its neighbours.
    """

    def __init__(self, pop):
        """
        Populations must already be evaluated.
        """
        self.pop = pop
        self.pareto_front = pop[0].pareto_front

    def eval_rw_features(self):
        dist_f_dist_x_avg, dist_c_dist_x_avg, bhv = randomwalkfeatures(
            self.pop, self.pareto_front, Instances=None
        )
        self.bhv = bhv
        self.dist_c_dist_x_avg = dist_c_dist_x_avg
        self.dist_f_dist_x_avg = dist_f_dist_x_avg


class MultipleRandomWalkAnalysis(np.ndarray):
    """
    Aggregate RW features across populations/walks.
    """

    def __new__(cls, pops):
        obj = (
            super(MultipleRandomWalkAnalysis, cls)
            .__new__(cls, len(pops), dtype=cls)
            .view(cls)
        )
        for i in range(len(pops)):
            obj[i] = RandomWalkAnalysis(pops[i])

        return obj

    def eval_features_for_all_populations(self):
        """
        Evaluate features for all populations.
        """

        for i in range(len(self)):
            self[i].eval_rw_features()

    def generate_array_for_attribute(self, attribute_name):
        attribute_array = []
        for i in range(len(self)):
            attribute_array.append(getattr(self[i], attribute_name))
        return np.asarray(attribute_array)

    def aggregate_features(self):
        """
        Aggregate features for all populations. Must be run after eval_features_for_all_populations.
        """
        self.dist_f_dist_x_avg_rws = np.mean(
            self.generate_array_for_attribute("dist_f_dist_x_avg")
        )
        self.dist_c_dist_x_avg_rws = np.mean(
            self.generate_array_for_attribute("dist_c_dist_x_avg")
        )
        self.bhv_avg_rws = np.mean(self.generate_array_for_attribute("bhv"))
