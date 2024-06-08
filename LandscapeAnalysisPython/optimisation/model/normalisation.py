import numpy as np


class Normalisation:
    def __init__(self):
        self.ideal = np.inf
        self.nadir = -np.inf

    def do(self, population, **kwargs):
        return self._do(population, **kwargs)

    def _do(self, population, **kwargs):
        # Extract objectives
        obj_array = population.extract_obj()

        # Use current ideal and nadir points or re-calculate
        if 'recalculate' in kwargs:
            # Find lower and upper bounds
            f_min = np.min(obj_array, axis=0)
            f_max = np.max(obj_array, axis=0)

            # Update the ideal and nadir points
            self.ideal = np.minimum(f_min, self.ideal)
            self.nadir = f_max

        # Normalise to [0, 1]^n and re-assign the objectives
        obj_array = (obj_array - self.ideal) / (self.nadir - self.ideal)
        population.assign_obj(obj_array)

        return population
