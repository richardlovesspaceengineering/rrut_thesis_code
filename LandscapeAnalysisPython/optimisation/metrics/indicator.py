import numpy as np

from optimisation.metrics.util import calculate_igd, calculate_gd
from optimisation.util.calculate_hypervolume import calculate_hypervolume


class Indicator:

    def __init__(self, metric):

        self.name = metric.lower()
        self.metric = None

        if 'igd' in self.name:   # Inverted Generational Distance
            self.metric = calculate_igd
        elif 'gd' in self.name:  # Generational Distance
            self.metric = calculate_gd
        elif 'hv' in self.name:  # Hyper volume
            self.metric = calculate_hypervolume
        else:
            raise Exception(f"Metric {metric} is not implemented!")

        self.archive = []
        self.generations = []

    def do(self, problem, population, n_gen, return_value=False):
        # Extract objectives
        pop_obj_arr = population.extract_obj()
        pareto_obj_arr = problem.pareto_set.extract_obj()

        if len(pareto_obj_arr) == 0:
            return np.NaN

        # Calculate metric for current population
        if 'hv' in self.name:
            metric = self.metric(pop_obj_arr)
        else:
            metric = self.metric(pop_obj_arr, pareto_obj_arr)

        # Store value to archive
        self._do(metric, n_gen)

        if return_value:
            return metric

    def _do(self, metric, n_gen):
        # Store value to archive
        self.archive.append(metric)
        self.generations.append(n_gen)

    def extract_archive(self):
        ind_arr = np.array(self.archive)
        gen_arr = np.array(self.generations)
        archive_arr = np.hstack((gen_arr[:, None], ind_arr[:, None]))
        return archive_arr

    def previous_value(self):
        if len(self.archive) > 0:
            return self.archive[-1]
        else:
            return np.inf
