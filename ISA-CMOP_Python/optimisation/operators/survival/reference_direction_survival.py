import numpy as np

from optimisation.model.survival import Survival
from optimisation.util.hyperplane_normalisation import HyperplaneNormalisation
from optimisation.util.non_dominated_sorting import NonDominatedSorting
from optimisation.util.misc import calc_perpendicular_distance, intersect


class ReferenceDirectionSurvival(Survival):

    def __init__(self, ref_dirs, filter_infeasible=True):

        super().__init__(filter_infeasible=filter_infeasible)
        self.ref_dirs = ref_dirs
        self.filter_infeasible = filter_infeasible

        self.opt = None
        self.norm = HyperplaneNormalisation(ref_dirs.shape[1])

    def _do(self, problem, pop, n_survive, cons_val=None, gen=None, max_gen=None, **kwargs):

        # Extract the objective function values from the population
        obj_array = pop.extract_obj()

        # Calculate the Pareto fronts from the population
        fronts, rank = NonDominatedSorting().do(obj_array, cons_val=cons_val, return_rank=True, n_stop_if_ranked=n_survive)
        non_dominated, last_front = fronts[0], fronts[-1]

        # Update the hyperplane based boundary estimation
        hyp_norm = self.norm
        hyp_norm.update(obj_array, nds=non_dominated)
        ideal, nadir = hyp_norm.ideal_point, hyp_norm.nadir_point

        # Consider the whole population
        idxs = np.concatenate(fronts)
        pop, rank, obj_array = pop[idxs], rank[idxs], obj_array[idxs]

        # Update front indices for population
        counter = 0
        for i in range(len(fronts)):
            for j in range(len(fronts[i])):
                fronts[i][j] = counter
                counter += 1
        last_front = fronts[-1]

        # Associate individuals to niches
        niche_of_individuals, dist_to_niche, dist_matrix = associate_to_niches(obj_array, self.ref_dirs, ideal, nadir)

        # Save rank and crowding to the individuals
        for i in range(len(pop)):
            pop[i].rank = rank[i]

        # Set the optimum, first front and closest to all reference directions
        closest = np.unique(dist_matrix[:, np.unique(niche_of_individuals)].argmin(axis=0))
        self.opt = pop[intersect(fronts[0], closest)]

        # Select individuals to survive
        if len(pop) > n_survive:

            if len(fronts) == 1:    # There is only one front
                n_remaining = n_survive
                until_last_front = np.array([], dtype=np.int)
                niche_count = np.zeros(len(self.ref_dirs), dtype=np.int)
            else:                   # Some survivors already selected
                until_last_front = np.concatenate(fronts[:-1])
                niche_count = calc_niche_count(len(self.ref_dirs), niche_of_individuals[until_last_front])
                n_remaining = n_survive - len(until_last_front)

            s = niching(pop[last_front], n_remaining, niche_count, niche_of_individuals[last_front], dist_to_niche[last_front])

            temp = np.concatenate((until_last_front, last_front[s].tolist()))
            survivors = idxs[temp]
        else:
            survivors = idxs

        return survivors


def niching(pop, n_remaining, niche_count, niche_of_individuals, dist_to_niche):

    survivors = []

    # Boolean array of elements considered for each iteration
    mask = np.ones(len(pop), dtype=bool)

    while len(survivors) < n_remaining:

        # Number of individuals to select
        n_select = n_remaining - len(survivors)

        # Niches to which new individuals can be assigned
        next_niches_list = np.unique(niche_of_individuals[mask])
        # Update corresponding niche count
        next_niche_count = niche_count[next_niches_list]

        # Minimum niche count
        min_niche_count = next_niche_count.min()

        # All niches with the minimum niche count (truncate randomly if more niches than remaining individuals)
        next_niches = next_niches_list[np.where(next_niche_count == min_niche_count)[0]]
        next_niches = next_niches[np.random.permutation(len(next_niches))[:n_select]]

        for next_niche in next_niches:

            # Indices of individuals that are considered
            next_idx = np.where(np.logical_and(niche_of_individuals == next_niche, mask))[0]

            # Shuffle to break tie (equal perpendicular distance), else select randomly
            np.random.shuffle(next_idx)

            if niche_count[next_niche] == 0:
                next_idx = next_idx[np.argmin(dist_to_niche[next_idx])]
            else:
                # Already randomised through shuffling
                next_idx = next_idx[0]

            # Add selected individual to survivors
            mask[next_idx] = False
            survivors.append(int(next_idx))

            # Update corresponding niche count
            niche_count[next_niche] += 1

    return survivors


def associate_to_niches(obj_array, niches, ideal_point, nadir_point, utopian_epsilon=0.0):

    # Calculate utopian point
    utopian_point = ideal_point - utopian_epsilon

    # Calculate denominator
    denom = nadir_point - utopian_point
    denom[denom == 0] = 1e-12

    # Normalize by ideal point and intercepts
    normalised_obj = (obj_array - utopian_point) / denom

    # Calculate perpendicular distance matrix
    dist_matrix = calc_perpendicular_distance(normalised_obj, niches)

    # Calculate niches
    niche_of_individuals = np.argmin(dist_matrix, axis=1)
    dist_to_niche = dist_matrix[np.arange(obj_array.shape[0]), niche_of_individuals]

    return niche_of_individuals, dist_to_niche, dist_matrix


def calc_niche_count(n_niches, niche_of_individuals):

    niche_count = np.zeros(n_niches, dtype=np.int)
    index, count = np.unique(niche_of_individuals, return_counts=True)
    niche_count[index] = count

    return niche_count

