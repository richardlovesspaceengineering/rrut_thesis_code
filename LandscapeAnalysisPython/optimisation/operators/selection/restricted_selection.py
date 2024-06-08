import numpy as np
import math

from optimisation.operators.selection.tournament_selection import TournamentSelection
from optimisation.util.non_dominated_sorting import NonDominatedSorting
from optimisation.util.dominator import get_relation
from optimisation.util.misc import random_permutations


class RestrictedSelection(TournamentSelection):

    def _do(self, hybrid_pop, n_select, n_parents=2, **kwargs):

        n_pop = len(hybrid_pop) // 2

        # Extract objective array
        obj_array = hybrid_pop.extract_obj()

        # Calculate ranks
        _, rank = NonDominatedSorting().do(obj_array, return_rank=True)

        # Calculate proportion of non-dominated solutions of c_a in hybrid population
        p_c = np.sum(rank[:n_pop] == 0)/len(hybrid_pop)
        # Calculate proportion of non-dominated solutions of d_a in hybrid population
        p_d = np.sum(rank[n_pop:] == 0)/len(hybrid_pop)

        # Number of random individuals needed
        n_random = n_select * n_parents * self.pressure

        # Number of permutations needed
        n_perms = math.ceil(n_random/n_pop)

        # Get random permutations
        permutations = random_permutations(n_perms, n_pop)[:n_random]
        # Reshape permutations to required size
        permutations = np.reshape(permutations, (n_select*n_parents, self.pressure))

        if p_c <= p_d:      # Choose from d_a
            permutations[::n_parents, :] += n_pop

        p_f = np.random.random(n_select)
        permutations[1::n_parents, :][p_f >= p_c] += n_pop

        # Compute using tournament function
        self.sequence = self.comp_func(hybrid_pop, permutations, **kwargs)

        return np.reshape(self.sequence, (n_select, n_parents))


def comp_by_cv_dom_then_random(pop, permutations):

    # Initialise selection index array
    selection = np.full(permutations.shape[0], np.nan)

    for i in range(permutations.shape[0]):

        # Extract indices from permutation array
        idx_a = permutations[i, 0]
        idx_b = permutations[i, 1]

        # If both solutions are feasible, use domination
        if pop[idx_a].cons_sum <= 0.0 and pop[idx_b].cons_sum <= 0.0:
            rel = get_relation(pop[idx_a].obj, pop[idx_b].obj)
            if rel == 1:
                selection[i] = idx_a
            elif rel == -1:
                selection[i] = idx_b
            else:
                selection[i] = np.random.choice([idx_a, idx_b])
        elif pop[idx_a].cons_sum <= 0.0:
            selection[i] = idx_a
        elif pop[idx_b].cons_sum <= 0.0:
            selection[i] = idx_b
        else:
            selection[i] = np.random.choice([idx_a, idx_b])

    return selection.astype(np.int, copy=False)
