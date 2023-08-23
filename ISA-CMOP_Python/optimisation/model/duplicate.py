import numpy as np
import scipy.spatial


class DuplicateElimination:

    def __init__(self, epsilon=1e-16):

        self.epsilon = epsilon

    def do(self, pop, *args, return_indices=False, to_itself=True):

        original = pop

        if len(pop) == 0:
            return pop

        if to_itself:
            _duplicate = self._do(pop, None, np.full(len(pop), False))
            # pop = [pop[idx] for idx in range(len(pop)) if ~_duplicate[idx]]
            pop = pop[~_duplicate]

        for arg in args:
            if len(arg) > 0:

                if len(pop) == 0:
                    break
                elif len(arg) == 0:
                    continue
                else:
                    _duplicate = self._do(pop, arg, np.full(len(pop), False))
                    # pop = [pop[idx] for idx in range(len(pop)) if ~_duplicate[idx]]
                    pop = pop[~_duplicate]

        if return_indices:

            h_idx_arr = {}
            for k, ind in enumerate(original):
                h_idx_arr[ind] = k

            no_duplicate = [h_idx_arr[ind] for ind in pop]
            is_duplicate = [i for i in range(len(original)) if i not in no_duplicate]

            return pop, no_duplicate, is_duplicate
        else:
            return pop

    def _do(self, pop, other, is_duplicate):
        pass


class DefaultDuplicateElimination(DuplicateElimination):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def _do(self, pop, other, is_duplicate):

        dist = self.calc_dist(pop, other)
        dist[np.isnan(dist)] = np.inf

        is_duplicate[np.any(dist < self.epsilon, axis=1)] = True

        return is_duplicate

    def calc_dist(self, pop, other=None):

        pop_var = pop.extract_var()

        if other is None:
            dist = scipy.spatial.distance.cdist(pop_var, pop_var)
            dist[np.triu_indices(len(pop_var))] = np.inf
        else:
            other_var = other.extract_var()
            if pop_var.ndim == 1:
                pop_var = pop_var[None, :]
            if other_var.ndim == 1:
                other_var = other_var[None, :]
            dist = scipy.spatial.distance.cdist(pop_var, other_var)

        return dist


class NoDuplicateElimination(DuplicateElimination):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def _do(self, pop, other, is_duplicate):
        return is_duplicate
