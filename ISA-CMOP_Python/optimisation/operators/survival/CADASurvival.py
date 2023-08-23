import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

from optimisation.model.population import Population
from optimisation.operators.decomposition.get import get_decomposition
from optimisation.util.misc import calc_perpendicular_distance
from optimisation.util.split_by_feasibility import split_by_feasibility
from optimisation.util.non_dominated_sorting import NonDominatedSorting


class CADASurvival:

    def __init__(self, ref_dirs):

        self.ref_dirs = ref_dirs

        self.opt = None
        self.ideal_point = np.full(ref_dirs.shape[1], np.inf)

        self.decomposition = get_decomposition('asf', weight_0=1e-4)
        self.calculate_perpendicular_distance = calc_perpendicular_distance

    def do(self, _, pop, da, n_survive, **kwargs):

        # Offspring are the last of the merged population
        off = pop[-n_survive:]

        # Update ideal point
        off_obj_array = off.extract_obj()
        self.ideal_point = np.min(np.vstack((self.ideal_point, off_obj_array)), axis=0)

        # Update convergence archive
        pop = self._update_CA(pop, n_survive)

        # Update diversity archive
        hybrid_pop = Population.merge(da, off)
        da = self._update_DA(pop, hybrid_pop, n_survive)

        return pop, da

    def _associate(self, pop):

        """Associate each individual with a weight vector and calculate decomposed fitness value"""

        # Extract objective values
        obj_array = pop.extract_obj()

        distance_matrix = self.calculate_perpendicular_distance(obj_array - self.ideal_point, self.ref_dirs)
        niche_of_individuals = np.argmin(distance_matrix, axis=1)

        # Calculate decomposed fitness value
        fv = self.decomposition.do(obj_array, weights=self.ref_dirs[niche_of_individuals, :], ideal_point=self.ideal_point)

        return niche_of_individuals, fv

    def _update_CA(self, pop, n_survive):

        # Extract constraint value
        cons_array = pop.extract_cons_sum()

        # Split population
        feasible, infeasible = split_by_feasibility(cons_array, sort_infeasible_by_cv=True)
        feasible_pop, infeasible_pop = pop[feasible], pop[infeasible]

        if len(feasible_pop) == n_survive:      # Exactly n_survive feasible individuals
            # Extract objective array
            obj_array = feasible_pop.extract_obj()

            # Conduct non-dominated sorting until the front is split
            fronts, rank = NonDominatedSorting().do(obj_array, return_rank=True)

            # Survivors are entire feasible population
            survivors = feasible_pop

            # Assign ranks
            for idx in range(len(survivors)):
                survivors[idx].rank = rank[idx]

            # Update optimum
            self.opt = survivors[fronts[0]]

        elif len(feasible_pop) < n_survive:     # Not enough feasible individuals

            # Determine number of infeasible survivors required
            n_remainder = n_survive - len(feasible_pop)

            cons_array_infeasible = infeasible_pop.extract_cons_sum()
            _, f_2 = self._associate(infeasible_pop)
            f_sub = np.column_stack((cons_array_infeasible, f_2))
            fronts = NonDominatedSorting().do(f_sub, n_stop_if_ranked=n_remainder)

            survival_idxs = []
            for front in fronts:
                if len(survival_idxs) + len(front) <= n_remainder:
                    survival_idxs.append(front)
                else:
                    n_required = n_remainder - len(survival_idxs)
                    survival_idxs.append(front[:n_required])

            # Survivors are feasible population plus required number of sorted infeasible individuals
            survival_idxs = np.concatenate(survival_idxs)
            survivors = feasible_pop + infeasible_pop[survival_idxs]
            obj_array = survivors.extract_obj()
            fronts, rank = NonDominatedSorting().do(obj_array, return_rank=True)
            # Assign ranks
            for idx in range(len(survivors)):
                survivors[idx].rank = rank[idx]

            # Update optimum
            self.opt = survivors[fronts[0]]

        else:       # Too many feasible individuals

            # Extract objective array
            obj_array = feasible_pop.extract_obj()

            # Conduct non-dominated sorting until the front is split
            fronts, rank = NonDominatedSorting().do(obj_array, return_rank=True, n_stop_if_ranked=n_survive)
            # Population indices
            idxs = np.concatenate(fronts)
            _feasible_pop, _rank, _obj_array = feasible_pop[idxs], rank[idxs], obj_array[idxs]

            if len(_feasible_pop) > n_survive:      # Remove individual in most crowded niche and with worst decomposed fitness value

                # Calculate niche and decomposed fitness
                niche_of_individuals, fv = self._associate(_feasible_pop)
                index, count = np.unique(niche_of_individuals, return_counts=True)

                # Initialise temporary survival mask
                _survivors = np.ones(len(_feasible_pop), dtype=bool)

                # Continue removing individuals while number of survivors is above required number
                while np.count_nonzero(_survivors) > n_survive:
                    most_crowded_niches, = np.where(count == count.max())
                    worst_idx = None
                    worst_niche = None
                    worst_fit = -1e3

                    for niche in most_crowded_niches:
                        most_crowded, = np.where((niche_of_individuals == index[niche]) & _survivors)
                        local_worst = most_crowded[fv[most_crowded].argmax()]

                        # Calculate distance to worst (maximum fitness) individual
                        distance_to_max_fitness = cdist(_obj_array[[local_worst], :], _obj_array).flatten()
                        distance_to_max_fitness[local_worst] = np.inf
                        distance_to_max_fitness[~_survivors] = np.inf
                        min_distance_to_max_fitness = np.min(distance_to_max_fitness)

                        # Calculate distance between individuals in current niche
                        distance_in_niche = squareform(pdist(_obj_array[most_crowded]))
                        np.fill_diagonal(distance_in_niche, np.inf)

                        delta_distance = distance_in_niche - min_distance_to_max_fitness
                        min_distance_idxs = np.unravel_index(np.argmin(delta_distance, axis=None), distance_in_niche.shape)

                        if (delta_distance[min_distance_idxs] < 0.0) or (delta_distance[min_distance_idxs] == 0 and (fv[most_crowded[list(min_distance_idxs)]] > local_worst).any()):
                            min_distance_idxs = list(min_distance_idxs)
                            np.random.shuffle(min_distance_idxs)
                            closest = most_crowded[min_distance_idxs]
                            local_worst = closest[np.argmax(fv[closest])]
                        if fv[local_worst] > worst_fit:
                            worst_fit = fv[local_worst]
                            worst_idx = local_worst
                            worst_niche = niche

                    # Update survival mask
                    _survivors[worst_idx] = False
                    count[worst_niche] -= 1

                # Update survivors
                _feasible_pop, _rank = _feasible_pop[_survivors], _rank[_survivors]

            # Survivors are ranked feasible population less any excess individuals
            survivors = _feasible_pop

            # Assign ranks
            for idx in range(len(survivors)):
                survivors[idx].rank = _rank[idx]

            # Update optimum
            self.opt = survivors[_rank == 0]

        return survivors

    def _update_DA(self, pop, hybrid_pop, n_survive):

        niche_hybrid_pop, fv = self._associate(hybrid_pop)
        niche_ca, _ = self._associate(pop)

        itr = 1
        survivors = []
        while len(survivors) < n_survive:
            for i in range(n_survive):
                current_ca, = np.where(niche_ca == i)
                if len(current_ca) < itr:
                    for j in range(itr - len(current_ca)):
                        current_da = np.where(niche_hybrid_pop == i)[0]
                        if current_da.size > 0:
                            # Extract objective function
                            obj_array = hybrid_pop[current_da].extract_obj()
                            nd = NonDominatedSorting().do(obj_array, only_non_dominated_front=True, n_stop_if_ranked=0)
                            # Extract best individual
                            i_best = current_da[nd[np.argmin(fv[current_da[nd]])]]
                            niche_hybrid_pop[i_best] = -1
                            if len(survivors) < n_survive:
                                survivors.append(i_best)
                        else:
                            break

                if len(survivors) == n_survive:
                    break
            itr += 1

        return hybrid_pop[survivors]
