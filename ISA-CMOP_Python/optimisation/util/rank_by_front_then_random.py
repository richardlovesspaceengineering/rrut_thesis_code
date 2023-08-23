import numpy as np

from optimisation.util.non_dominated_sorting import NonDominatedSorting
from optimisation.util.calculate_crowding_distance import calculate_crowding_distance


def rank_by_front_then_random(pop, n_survive, cons_val=None):

    # Extract the objective function values from the population
    obj_array = pop.extract_obj()

    # Conduct non-dominated sorting until the front is split
    fronts = NonDominatedSorting().do(obj_array, cons_val=cons_val, n_stop_if_ranked=n_survive)

    # Initialise array of indices of surviving individuals
    survivors = np.zeros(0, dtype=np.int)

    # Cycle through fronts
    for k, front in enumerate(fronts):

        # Calculate crowding distance of the front
        front_crowding_distance = calculate_crowding_distance(obj_array[front, :])

        # Save rank and crowding to the individuals
        for j, i in enumerate(front):
            pop[i].rank = k
            pop[i].crowding_distance = front_crowding_distance[j]

        idx_arr = np.arange(len(front))
        np.random.shuffle(idx_arr)
        if len(survivors) + len(front) > n_survive:
            idx_arr = idx_arr[:(n_survive - len(survivors))]

        # Concatenate survivors from current front to existing survivors
        survivors = np.concatenate((survivors, front[idx_arr]))

    return survivors

