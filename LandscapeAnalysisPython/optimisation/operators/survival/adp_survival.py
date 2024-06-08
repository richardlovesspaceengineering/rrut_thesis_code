import numpy as np

from optimisation.model.survival import Survival
# from optimisation.util.misc import vectorized_cdist
# from optimisation.util.split_by_feasibility import split_by_feasibility

from optimisation.util.misc import calc_gamma


class APDSurvival(Survival):

    def __init__(self, ref_dirs, filter_infeasible=True, alpha=2.0):
        super().__init__(filter_infeasible=filter_infeasible)
        n_dim = ref_dirs.shape[1]

        self.alpha = alpha
        self.niches = None
        self.V, self.gamma = None, None
        self.ideal, self.nadir = np.full(n_dim, np.inf), None

        self.ref_dirs = ref_dirs

    def _do(self, problem, pop, n_survive, gen=None, max_gen=None, cons_val=None, **kwargs):

        # Update from new reference vectors
        self.gamma = calc_gamma(self.V)

        # get the objective space values
        obj_arr = pop.extract_obj()

        # store the ideal and nadir point estimation for adapt - (and ideal for transformation)
        self.ideal = np.minimum(obj_arr.min(axis=0), self.ideal)

        # translate the population to make the ideal point the origin
        obj_arr = obj_arr - self.ideal

        # the distance to the ideal point
        dist_to_ideal = np.linalg.norm(obj_arr, axis=1)
        dist_to_ideal[dist_to_ideal < 1e-64] = 1e-64

        # normalize by distance to ideal
        obj_prime = obj_arr / dist_to_ideal[:, None]

        # calculate for each solution the acute angles to ref dirs
        acute_angle = np.arccos(obj_prime @ self.V.T)
        niches = acute_angle.argmin(axis=1)

        # assign to each reference direction the solution
        niches_to_ind = [[] for _ in range(len(self.V))]
        selected_from_niche = [[] for _ in range(len(self.V))]

        for k, i in enumerate(niches):
            niches_to_ind[i].append(k)
            selected_from_niche[i].append(False)

        # all individuals which will be surviving
        survived_indices = []

        # Ensuring all population is given an ADP value (up to n_survive)
        while len(survived_indices) < n_survive:

            # for each reference direction
            for k in range(len(self.V)):

                # individuals assigned to the niche
                assigned_to_niche = np.array(niches_to_ind[k])

                # Exit of number of individuals reached
                if len(survived_indices) >= n_survive:
                    break

                # if niche is not empty
                if len(assigned_to_niche) > 0:
                    # the angle of niche to nearest neighboring niche
                    gamma = self.gamma[k]

                    # the angle from the individuals of this niches to the niche itself
                    theta = acute_angle[assigned_to_niche, k]

                    # the penalty which is applied for the metric
                    M = problem.n_obj if problem.n_obj > 2.0 else 1.0
                    penalty = M * ((gen / max_gen) ** self.alpha) * (theta / gamma)

                    # calculate the angle-penalized penalized (APD)
                    apd = dist_to_ideal[assigned_to_niche] * (1 + penalty)

                    # Continue to next niche if all individuals already selected
                    niche_mask = np.invert(selected_from_niche[k])
                    if len(apd[niche_mask]) == 0:
                        continue

                    # the individual which survives
                    index = apd[niche_mask].argmin()
                    survivor = assigned_to_niche[niche_mask][index]

                    # Set flag to not re-select this individual
                    selected_from_niche[k][np.argwhere(assigned_to_niche == survivor)[0][0]] = True

                    # select the one with smallest APD value
                    survived_indices.append(survivor)

        # Storage
        self.niches = niches_to_ind
        self.nadir = pop[survived_indices].extract_obj().max(axis=0)

        return survived_indices
