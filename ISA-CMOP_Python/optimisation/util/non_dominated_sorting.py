import numpy as np

from optimisation.util import dominator


class NonDominatedSorting(object):

    def __init__(self, method='fast_non_dominated_sort'):

        self.method = method

    def do(self, obj_val, cons_val=None, return_rank=False, only_non_dominated_front=False, n_stop_if_ranked=None):

        if self.method == 'fast_non_dominated_sort':
            func = self.fast_non_dominated_sort
        else:
            raise Exception('Unknown non-dominated sorting method: %s' % self.method)

        if n_stop_if_ranked is None:
            n_stop_if_ranked = int(1e8)

        # Calculate each front using fast non-dominated sort
        fronts = func(obj_val, cons_val)

        # Convert each front to a numpy array, and filter by n_stop_if_ranked if desired
        _fronts = []
        n_ranked = 0
        for front in fronts:

            # Convert front to numpy array
            _fronts.append(np.asarray(front, dtype=np.int))

            # Increment the n_ranked solution counter
            n_ranked += len(front)

            # Stop if more than n_stop_if_ranked solutions are ranked
            if n_ranked >= n_stop_if_ranked:
                break

        # Update fronts to list of numpy arrays
        fronts = _fronts

        # If only the non-dominated front is desired, return it
        if only_non_dominated_front:
            return fronts[0]

        # If we need to return the rank also
        if return_rank:
            rank = rank_from_fronts(fronts, obj_val.shape[0])
            return fronts, rank

        return fronts

    @staticmethod
    def fast_non_dominated_sort(obj_val, cons_val=None):

        # Calculate domination matrix
        m = dominator.calculate_domination_matrix(obj_val, cons_val)

        # Domination matrix shape
        n = m.shape[0]

        # Initialise fronts as empty
        fronts = []

        # If no members in domination matrix, return empty front
        if n == 0:
            return fronts

        # Final rank
        n_ranked = 0
        ranked = np.zeros(n, dtype=np.int)

        # For each individual, create a list of all individuals dominated by that particular individual
        is_dominating = [[] for idx in range(n)]

        # Storing number of solutions dominated by each individual
        n_dominated = np.zeros(n)

        current_front = []
        for i in range(n):
            for j in range(i+1, n):
                rel = m[i, j]
                if rel == 1:
                    is_dominating[i].append(j)
                    n_dominated[j] += 1
                elif rel == -1:
                    is_dominating[j].append(i)
                    n_dominated[i] += 1

            if n_dominated[i] == 0:
                current_front.append(i)
                ranked[i] = 1.0
                n_ranked += 1

        # Append the first front to the current front
        fronts.append(current_front)

        # While not all solutions are assigned to a Pareto front
        while n_ranked < n:

            next_front = []

            # For each individual in the current front
            for i in current_front:

                # All solutions that are dominated by this individual
                for j in is_dominating[i]:
                    n_dominated[j] -= 1
                    if n_dominated[j] == 0:
                        next_front.append(j)
                        ranked[j] = 1.0
                        n_ranked += 1

            fronts.append(next_front)
            current_front = next_front

        return fronts


def rank_from_fronts(fronts, n):
    # create the rank array and set values
    rank = np.full(n, 1e16, dtype=np.int)
    for i, front in enumerate(fronts):
        rank[front] = i

    return rank
