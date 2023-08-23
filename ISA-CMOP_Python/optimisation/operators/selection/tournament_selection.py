import math
import numpy as np

from optimisation.util import dominator
from optimisation.model.selection import Selection
from optimisation.util.misc import random_permutations


class TournamentSelection(Selection):

    def __init__(self, comp_func=None, pressure=2):

        super().__init__()

        # Selection pressure (default is binary tournament)
        self.pressure = pressure

        # Function to compare two individuals
        if comp_func is None:
            comp_func = binary_tournament
        self.comp_func = comp_func

        # Selection sequence
        self.sequence = None

    def _do(self, pop, n_select, n_parents=1):

        # Number of random individuals needed
        n_random = n_select*n_parents*self.pressure

        # Number of permutations needed
        n_perms = math.ceil(n_random/len(pop))

        # Get random permutations
        permutations = random_permutations(n_perms, len(pop))[:n_random]
        # Reshape permutations to required size
        permutations = np.reshape(permutations, (n_select*n_parents, self.pressure))

        # Compare using binary tournament
        self.sequence = self.comp_func(pop, permutations)

        return np.reshape(self.sequence, (n_select, n_parents))


def binary_tournament(pop, permutations, tournament_type=None):

    # Tournament type
    if tournament_type is None:
        tournament_type = 'comp_by_dom_and_crowding'
    tournament_type = tournament_type

    # Check that binary tournament can be conducted
    if permutations.shape[1] != 2:
        raise ValueError('Binary tournament requires a selection pressure of 2')

    # Initialise selection index array
    selection = np.full(permutations.shape[0], np.nan)

    # Cycle through permutations
    for i in range(permutations.shape[0]):

        # Extract indices from permutation array
        idx_a = permutations[i, 0]
        idx_b = permutations[i, 1]

        # Check if either solution is has a non-zero constraint sum
        if pop[idx_a].cons_sum > 0.0 or pop[idx_b].cons_sum > 0.0:
            # Select the individual with the lower constraint value
            selection[i] = compare(idx_a, idx_b, pop[idx_a].cons_sum, pop[idx_b].cons_sum, return_random_if_equal=True)

        else:   # Both solutions are feasible

            if tournament_type == 'comp_by_dom_and_crowding':
                # Compare individuals by objective function value domination
                relation = dominator.get_relation(pop[idx_a].obj, pop[idx_b].obj)

                if relation == 1:       # Individual idx_a dominates individual idx_b
                    selection[i] = idx_a
                elif relation == -1:    # Individual idx_b dominates individual idx_a
                    selection[i] = idx_b

            elif tournament_type == 'comp_by_rank_and_crowding':
                # Compare individuals by rank
                selection[i] = compare(idx_a, idx_b, pop[idx_a].rank, pop[idx_b].rank)

            else:
                raise Exception('Unknown tournament type')

            # If domination relation or rank was indifferent, compare by crowding distance
            if selection[i] is None or np.isnan(selection[i]):
                selection[i] = compare(idx_a, idx_b, pop[idx_a].crowding_distance, pop[idx_b].crowding_distance, method='maximise', return_random_if_equal=True)

    return selection.astype(np.int, copy=False)


def comp_by_cv_then_random(pop, permutations):

    # Initialise selection index array
    selection = np.full(permutations.shape[0], np.nan)

    for i in range(permutations.shape[0]):

        # Extract indices from permutation array
        idx_a = permutations[i, 0]
        idx_b = permutations[i, 1]

        # If at least one solution is infeasible
        if pop[idx_a].cons_sum > 0.0 or pop[idx_b].cons_sum > 0.0:
            selection[i] = compare(idx_a, idx_b, pop[idx_a].cons_sum, pop[idx_b].cons_sum, method='minimise', return_random_if_equal=True)
        else:   # If both solutions are feasible, select randomly
            selection[i] = np.random.choice([idx_a, idx_b])

    return selection.astype(np.int, copy=False)


def compare(a, b, a_val, b_val, method='minimise', return_random_if_equal=False):

    if method == 'minimise':
        if a_val < b_val:
            return a
        elif a_val > b_val:
            return b
        else:
            if return_random_if_equal:
                return np.random.choice([a, b])
            else:
                return None

    elif method == 'maximise':
        if a_val > b_val:
            return a
        elif a_val < b_val:
            return b
        else:
            if return_random_if_equal:
                return np.random.choice([a, b])
            else:
                return None

    else:
        raise Exception('Unknown comparison method')


