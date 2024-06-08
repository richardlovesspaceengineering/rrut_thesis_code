import numpy as np
from optimisation.util.calculate_crowding_distance import calculate_crowding_distance


def rank_fronts(fronts, obj_array, pop, survivors):

    # Cycle through fronts
    for k, front in enumerate(fronts):

        # Calculate crowding distance of the front
        front_crowding_distance = calculate_crowding_distance(obj_array[front, :])

        # Save rank and crowding to the individuals
        for j, i in enumerate(front):
            pop[i].rank = k
            pop[i].crowding_distance = front_crowding_distance[j]

        # Sort current front by crowding distance
        idx_arr = front_crowding_distance.argsort()[::-1]

        # Concatenate survivors from current front to existing survivors
        survivors = np.concatenate((survivors, front[idx_arr]))

    rank = np.zeros(len(survivors))
    rank[survivors] = range(len(survivors))

    return rank
