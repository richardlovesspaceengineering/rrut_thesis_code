import numpy as np
from optimisation.operators.sampling.AdaptiveWalk import AdaptiveWalk
import matplotlib.pyplot as plt
from optimisation.model.population import Population
from features.feature_helpers import generate_bounds_from_problem
from pymoo.problems import get_problem
import pygmo
from features.Analysis import Analysis, MultipleAnalysis
from features.RandomWalkAnalysis import RandomWalkAnalysis, MultipleRandomWalkAnalysis


class AdaptiveWalkAnalysis(RandomWalkAnalysis):
    def eval_features(self):
        # Determine PHC walk features.
        self.dummy_aw_feature = 1


class MultipleAdaptiveWalkAnalysis(MultipleRandomWalkAnalysis):
    def __init__(self, pops_walks, pops_neighbours_list):
        self.pops = pops_walks
        self.analyses = []

        if len(self.pops) != 0:
            for ctr, pop in enumerate(pops_walks):
                self.analyses.append(
                    AdaptiveWalkAnalysis(pop, pops_neighbours_list[ctr])
                )
