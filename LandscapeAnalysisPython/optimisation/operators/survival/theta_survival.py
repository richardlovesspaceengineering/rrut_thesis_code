# import copy
import numpy as np

from optimisation.model.survival import Survival
from optimisation.util.hyperplane_normalisation import HyperplaneNormalisation
from optimisation.util.non_dominated_sorting import NonDominatedSorting

import matplotlib.pyplot as plt
import matplotlib
plt.style.use('seaborn-talk')
np.set_printoptions(suppress=True)
# matplotlib.use('TkAgg')
line_colors = ['green', 'blue', 'red', 'orange', 'cyan', 'lawngreen', 'm', 'orangered', 'sienna', 'gold', 'violet', 'indigo', 'cornflowerblue']


class ThetaSurvival(Survival):

    def __init__(self, ref_dirs, theta=5.0, filter_infeasible=True):

        super().__init__(filter_infeasible=filter_infeasible)
        self.ref_dirs = ref_dirs
        self.theta = theta
        self.filter_infeasible = filter_infeasible
        self.norm = HyperplaneNormalisation(ref_dirs.shape[1])
        self.ideal = np.inf
        self.nadir = -np.inf

    def _do(self, problem, pop, n_survive, cons_val=None, gen=None, max_gen=None, **kwargs):

        # Extract the objective function values from the population
        obj_array = pop.extract_obj()

        # Calculate the Pareto fronts from the population
        fronts = NonDominatedSorting().do(obj_array, cons_val=cons_val, return_rank=False)
        non_dominated, last_front = fronts[0], fronts[-1]

        # Update the hyperplane based boundary estimation
        self.norm.update(obj_array, nds=non_dominated)
        self.ideal, self.nadir = self.norm.ideal_point, self.norm.nadir_point

        # TODO: min/ max normalisation doesn't work great
        # f_min = np.min(obj_array, axis=0)
        # f_max = np.max(obj_array, axis=0)
        #
        # self.ideal = np.minimum(f_min, self.ideal)
        # self.nadir = np.maximum(f_max, self.nadir)

        # Normalise population
        obj_array = (obj_array - self.ideal) / (self.nadir - self.ideal)

        # Assign population to clusters using ref_dirs and d2 distance
        clusters, d1_mat, d2_mat = cluster_association(obj_array, self.ref_dirs)

        # Conduct theta-nondominated sorting
        t_fronts, t_rank = self.theta_nondominated_sorting(obj_array, clusters, d1_mat, d2_mat, self.ref_dirs, self.theta)

        if 'return_rank_only' in kwargs:
            return t_rank

        # If only the first front is desired
        if 'first_rank_only' in kwargs:
            survivors = t_fronts[0]
            return survivors

        # Concatenate population until n_survive is reached
        survivors = t_fronts[0]
        idx = 0
        while len(survivors) + len(t_fronts[idx]) < n_survive:
            survivors = np.hstack((survivors, t_fronts[idx]))
            idx += 1

        # Random shuffle last front to achive full population size
        np.random.shuffle(t_fronts[idx])
        n_to_fill = int(n_survive) - len(survivors)
        survivors = np.hstack((survivors, t_fronts[idx][:n_to_fill]))

        # TODO: remove eventually once satisfied random shuffling last front is better
        # for idx in range(1, len(t_fronts)):
        #     survivors = np.hstack((survivors, t_fronts[idx]))
        #     if len(survivors) > n_survive:
        #         break

        # TODO: DEBUG remove eventually
        # self.plot_pareto(ref_vec=self.ref_dirs, obj_array=obj_array, fronts=t_fronts)
        # self.plot_pareto(ref_vec=self.ref_dirs, obj_array=obj_array, fronts=fronts)

        return survivors[:n_survive]

    @staticmethod
    def theta_nondominated_sorting(obj_array, clusters, d1_mat, d2_mat, ref_dirs, theta):
        n_obj = len(obj_array[0])

        # Initialise fronts and ranking
        fronts = [[] for _ in range(len(obj_array))]
        ranks = np.full(len(obj_array), 1e16, dtype=np.int)

        for ref_idx, niche in enumerate(clusters):

            # if niche is not empty
            if len(niche) > 0:

                # Extract PBI distances of niche individuals
                d1, d2 = np.array(d1_mat[ref_idx]), np.array(d2_mat[ref_idx])

                # Assign large theta to maintain extreme points
                if (np.count_nonzero(ref_dirs[ref_idx] == 0.0) + np.count_nonzero(ref_dirs[ref_idx] == 1.0)) == n_obj:
                    pbi = calc_pbi_func(d1, d2, 1e6)
                else:
                    pbi = calc_pbi_func(d1, d2, theta)

                # Assign ranks and store indices
                rank_indices = np.argsort(pbi)
                for i, index in enumerate(rank_indices):
                    fronts[i].append(niche[index])
                    ranks[niche[index]] = i

        # Return proper formatted fronts and ranks
        fronts_list = []
        for j, frnt in enumerate(fronts):
            if len(frnt) > 0:
                fronts_list.append(np.array(frnt))
        ranks_array = np.array(ranks).flatten()

        return fronts_list, ranks_array

    @staticmethod
    def plot_pareto(ref_vec=None, obj_array=None, fronts=None):

        n_obj = len(obj_array[0])

        # 2D Plot
        if n_obj == 2:
            fig, ax = plt.subplots(1, 1, figsize=(9, 7))
            fig.supxlabel('Obj 1', fontsize=14)
            fig.supylabel('Obj 2', fontsize=14)

            # Plot reference vectors
            if ref_vec is not None:
                scaling = 1.5
                origin = np.zeros(len(ref_vec))
                x_vec = scaling * np.vstack((origin, ref_vec[:, 0])).T
                y_vec = scaling * np.vstack((origin, ref_vec[:, 1])).T
                for i in range(len(x_vec)):
                    if i == 0:
                        ax.plot(x_vec[i], y_vec[i], color='black', linewidth=0.5, label='reference vectors')
                    else:
                        ax.plot(x_vec[i], y_vec[i], color='black', linewidth=0.5)

            if obj_array is not None:
                for i, frnt in enumerate(fronts):
                    ax.scatter(obj_array[frnt, 0], obj_array[frnt, 1], color=line_colors[i], s=75, label=f"rank {i}")

        # 3D Plot
        elif n_obj == 3:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.set_xlabel('Obj 1')
            ax.set_ylabel('Obj 2')
            ax.set_zlabel('Obj 3')

            # Plot reference vectors
            if ref_vec is not None:
                scaling = 4.0
                origin = np.zeros(len(ref_vec))
                x_vec = scaling * np.vstack((origin, ref_vec[:, 0])).T
                y_vec = scaling * np.vstack((origin, ref_vec[:, 1])).T
                z_vec = scaling * np.vstack((origin, ref_vec[:, 2])).T
                for i in range(len(x_vec)):
                    if i == 0:
                        ax.plot(x_vec[i], y_vec[i], z_vec[i], color='black', linewidth=0.5, label='reference vectors')
                    else:
                        ax.plot(x_vec[i], y_vec[i], z_vec[i], color='black', linewidth=0.5)

            if obj_array is not None:
                for i, frnt in enumerate(fronts):
                    ax.scatter3D(obj_array[frnt, 0], obj_array[frnt, 1], obj_array[frnt, 2], color=line_colors[i], s=50, label=f"rank {i}")

        plt.legend(loc='best', frameon=False)
        plt.show()
        # plt.savefig('/home/juan/PycharmProjects/optimisation_framework/multi_obj/results/zdt1_mmode_gen_' + str(len(self.surrogate.population)) + '.png')


def cluster_association(obj_array, ref_dirs):

    # Compute d2 distances between population and reference directions
    d1_dist, d2_dist = calc_pbi_distances(obj_array, ref_dirs)

    # Indices of niches
    niches = d2_dist.argmin(axis=1)

    # Initialise clusters
    clusters = [[] for _ in range(len(ref_dirs))]
    d1_distances = [[] for _ in range(len(ref_dirs))]
    d2_distances = [[] for _ in range(len(ref_dirs))]

    # Assign niches to clusters
    for k, i in enumerate(niches):
        clusters[i].append(k)
        d1_distances[i].append(d1_dist[k, i])
        d2_distances[i].append(d2_dist[k, i])

    return clusters, d1_distances, d2_distances


def calc_pbi_distances(obj_array, ref_dirs):
    ref_norm = np.linalg.norm(ref_dirs, axis=1)
    obj_norm = np.linalg.norm(obj_array, axis=1).reshape(-1, 1)

    d1 = np.dot(obj_array, ref_dirs.T) / ref_norm
    d2 = np.sqrt(obj_norm ** 2 - d1 ** 2)

    return d1, d2


def calc_pbi_func(d1, d2, theta):
    return d1 + theta * d2
