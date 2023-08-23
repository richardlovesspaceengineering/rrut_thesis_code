import numpy as np

from optimisation.model.individual import Individual

from optimisation.util.non_dominated_sorting import NonDominatedSorting
from optimisation.util.calculate_crowding_distance import calculate_crowding_distance


class Population(np.ndarray):

    def __new__(cls, problem, n_individuals=0):
        obj = super(Population, cls).__new__(cls, n_individuals, dtype=cls).view(cls)
        for i in range(n_individuals):
            obj[i] = Individual(problem)

        return obj

    @classmethod
    def merge(cls, a, b):

        if isinstance(a, Population) and isinstance(b, Population):
            if len(a) == 0:
                return b
            elif len(b) == 0:
                return a
            else:
                obj = np.concatenate([a, b]).view(Population)
                return obj
        else:
            raise Exception('Both a and b must be Population instances')

    def extract_var(self):

        var_array = []
        for i in range(len(self)):
            if i == 0:
                var_array = self[i].var
            else:
                var_array = np.vstack((var_array, self[i].var))

        return var_array

    def extract_obj(self):

        obj_array = []
        for i in range(len(self)):
            if i == 0:
                obj_array = self[i].obj
            else:
                obj_array = np.vstack((obj_array, self[i].obj))

        return obj_array

    def extract_cons(self):

        cons_array = []
        for i in range(len(self)):
            if i == 0:
                cons_array = self[i].cons
            else:
                cons_array = np.vstack((cons_array, self[i].cons))

        return cons_array

    def extract_cons_sum(self):

        obj_array = []
        for i in range(len(self)):
            obj_array.append(self[i].cons_sum)

        return np.asarray(obj_array)

    def extract_rank(self):

        rank_array = []
        for i in range(len(self)):
            rank_array.append(self[i].rank)

        return np.asarray(rank_array)

    def extract_crowding(self):

        crowding_array = []
        for i in range(len(self)):
            crowding_array.append(self[i].crowding_distance)

        return np.asarray(crowding_array)

    def assign_var(self, problem, var_array):

        for i in range(len(self)):
            self[i].set_var(problem, var_array[i, :])

    def assign_obj(self, obj_array):

        for i in range(len(self)):
            self[i].obj = obj_array[i, :]

    def assign_cons(self, cons_array):

        for i in range(len(self)):
            self[i].cons = cons_array[i, :]

    def assign_rank_and_crowding(self):

        # Extract the objective function values from the population
        obj_array = self.extract_obj()
        cons_array = self.extract_cons_sum()

        # Conduct non-dominated sorting (considering constraints & objectives)
        fronts = NonDominatedSorting().do(obj_array, cons_val=cons_array, n_stop_if_ranked=len(self))

        # Cycle through fronts
        for k, front in enumerate(fronts):

            # Calculate crowding distance of the front
            front_crowding_distance = calculate_crowding_distance(obj_array[front, :])

            # Save rank and crowding to the individuals
            for j, i in enumerate(front):
                self[i].rank = k
                self[i].crowding_distance = front_crowding_distance[j]

