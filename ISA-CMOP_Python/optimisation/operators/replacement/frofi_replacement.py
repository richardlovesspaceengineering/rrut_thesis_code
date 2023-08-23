import numpy as np
import random

from optimisation.model.replacement import ReplacementStrategy

from optimisation.util.non_dominated_sorting import NonDominatedSorting
from optimisation.util.rank_by_front_then_random import rank_by_front_then_random
from optimisation.model.population import Population


class FROFIReplacement(ReplacementStrategy):

    def do(self, problem, pop, off, m=1, **kwargs):

        pop = self._do(problem, pop, off, m=m, **kwargs)

        return pop

    def _do(self, problem, pop, off, m=1, **kwargs):

        # Calculate replacement & archive via domination
        idxs, archive_idxs = self.domination_replacement(pop, off)

        # Conduct replacement & construct archive
        pop[idxs] = off[idxs]
        archive = off[archive_idxs]

        # Conduct replacement via archive
        pop = self.archive_replacement(problem, pop, archive, m=m)

        return pop

    @staticmethod
    def domination_replacement(pop, off):

        # Replacement boolean array
        replace = np.zeros(len(pop), dtype=bool)

        # Archive boolean array
        archive = np.zeros(len(off), dtype=bool)

        # Population parameters
        pop_obj_array, pop_cons_array = pop.extract_obj(), pop.extract_cons_sum()
        # Offspring parameters
        off_obj_array, off_cons_array = off.extract_obj(), off.extract_cons_sum()

        # Determining if new (offspring) solution dominates current solution
        for i in range(len(pop)):

            # Concatenating objective and constraint arrays to compare population and offspring
            obj_array = np.array([pop_obj_array[i, :], off_obj_array[i, :]])
            cons_array = np.array([[pop_cons_array[i]], [off_cons_array[i]]])

            # Non-dominated sort
            _, rank = NonDominatedSorting().do(obj_array, cons_val=cons_array, return_rank=True)

            # If offspring dominates current solution, then replace
            if rank[1] < rank[0]:
                replace[i] = True
            elif rank[1] == rank[0]:
                replace[i] = random.choice([False, True])

            # Non-dominated sort (ignoring constraints)
            _, rank_obj = NonDominatedSorting().do(obj_array, cons_val=None, return_rank=True)

            # If offspring dominates current solution (ignoring constraints), then add to archive
            if not replace[i] and (rank_obj[1] < rank_obj[0]):
                archive[i] = True

        return replace, archive

    @staticmethod
    def archive_replacement(problem, pop, archive, m=1):

        # Sort population by objective function
        idxs = rank_by_front_then_random(pop, len(pop), cons_val=None)
        pop = pop[idxs]

        # Divide population into m parts based on above ranking (reverse - so that worst solutions get replaced by
        # archive)
        split_indices = np.array_split(np.arange(0, len(pop)), m)
        pop_group = []
        for indices in split_indices:
            pop_group.append(pop[indices])

        # Conduct replacement via archive
        i = 0
        while len(archive) > 0 and i < m:

            # Population group parameters
            pop_group_cons_array = pop_group[i].extract_cons_sum()

            # Archive parameters
            archive_cons_array = archive.extract_cons_sum()

            # Select individual with the maximum constraint violation from the ith population group
            idx_a = np.argmax(pop_group_cons_array)

            # Select individual with the minimum constraint violation from the archive
            idx_b = np.argmin(archive_cons_array)

            # Concatenating objective arrays to compare both individuals
            obj_array = np.array([pop_group_cons_array[idx_a, :], archive_cons_array[idx_b, :]])

            # Non-dominated sort (ignoring constraints)
            _, rank_obj = NonDominatedSorting().do(obj_array, cons_val=None, return_rank=True)

            if rank_obj[1] < rank_obj[0]:
                # If individual B dominates individual A (ignoring constraints), then replace individual A with
                # individual B
                pop_group[i][idx_a] = archive[idx_b]

                # Remove individual B from the archive
                mask = np.zeros(len(archive), dtype=bool)
                mask[idx_b] = True
                archive = archive[~mask]

            # Increment population group
            i += 1

        # Reconstruct population from pop_group
        pop = Population(problem, n_individuals=0)
        for group in pop_group:
            pop = Population.merge(pop, group)

        return pop
