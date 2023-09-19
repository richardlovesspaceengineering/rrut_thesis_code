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

    def extract_dist_f_dist_x_avg_array(self):
        dist_f_dist_x_avg_array = []
        for i in range(len(self)):
            dist_f_dist_x_avg_array.append(self[i].dist_f_dist_x_avg)
        return np.asarray(dist_f_dist_x_avg_array)

    def extract_dist_c_dist_x_avg_array(self):
        dist_c_dist_x_avg_array = []
        for i in range(len(self)):
            dist_c_dist_x_avg_array.append(self[i].dist_c_dist_x_avg)
        return np.asarray(dist_c_dist_x_avg_array)

    def extract_bhv_array(self):
        bhv_array = []
        for i in range(len(self)):
            bhv_array.append(self[i].bhv)
        return np.asarray(bhv_array)

    def aggregate_features(self):
        """
        Aggregate features for all populations. Must be run after eval_features_for_all_populations.
        """
        self.dist_f_dist_x_avg_rws = np.mean(self.extract_dist_f_dist_x_avg_array())
        self.dist_c_dist_x_avg_rws = np.mean(self.extract_dist_c_dist_x_avg_array())
        self.bhv_avg_rws = np.mean(self.extract_bhv_array())
