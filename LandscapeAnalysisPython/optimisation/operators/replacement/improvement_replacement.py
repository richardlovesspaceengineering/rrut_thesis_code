import numpy as np
import random

from optimisation.model.replacement import ReplacementStrategy

from optimisation.util.non_dominated_sorting import NonDominatedSorting


class ImprovementReplacement(ReplacementStrategy):

    def _do(self, problem, pop, off, **kwargs):

        # Replacement boolean array
        replace = np.zeros(len(pop), dtype=bool)

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

        return replace
