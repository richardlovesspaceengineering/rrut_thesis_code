import numpy as np
from scipy.spatial import distance


def calculate_igd(population, pareto_front):
    population = np.atleast_2d(population)
    pareto_front = np.atleast_2d(pareto_front)

    # Inverted Generational Distance
    distances = distance.cdist(pareto_front, population)
    min_dist = np.min(distances, axis=1)
    igd = np.mean(min_dist)

    return igd


def calculate_gd(population, pareto_front):
    population = np.atleast_2d(population)
    pareto_front = np.atleast_2d(pareto_front)

    # Generational Distance
    distances = distance.cdist(population, pareto_front)  # TODO: Double check
    min_dist = np.min(distances, axis=1)
    gd = np.linalg.norm(min_dist) / len(pareto_front)

    return gd
