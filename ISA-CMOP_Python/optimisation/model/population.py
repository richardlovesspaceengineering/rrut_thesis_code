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
            raise Exception("Both a and b must be Population instances")

    @classmethod
    def merge_multiple(cls, *args):
        base = None
        for item in args:
            if isinstance(item, Population):
                if isinstance(base, Population):
                    base = np.concatenate([base, item]).view(Population)
                else:
                    base = item

        return base

    ### GETTERS
    def extract_var(self):
        # Extract decision variables from each individual. Should return an n x m array where n is the number of individuals, m is the number of objectives.
        var_array = []
        for i in range(len(self)):
            if i == 0:
                var_array = self[i].var
            else:
                var_array = np.vstack((var_array, self[i].var))

        return np.asarray(var_array)

    def extract_obj(self):
        # Extract objectives from each individual. Should return an n x m array where n is the number of individuals, m is the number of objectives.
        obj_array = []
        for i in range(len(self)):
            if i == 0:
                obj_array = self[i].obj
            else:
                obj_array = np.vstack((obj_array, self[i].obj))

        return np.asarray(obj_array)

    def extract_cons(self):
        # Extract constraints from each individual. Should return an n x m array where n is the number of individuals, m is the number of constraints.
        cons_array = []
        for i in range(len(self)):
            if i == 0:
                cons_array = self[i].cons
            else:
                cons_array = np.vstack((cons_array, self[i].cons))

        return np.asarray(cons_array)

    def extract_cv(self):
        # Extract CV from each individual. Should return an n x 1 array where n is the number of individuals.
        cv_array = []
        for i in range(len(self)):
            if i == 0:
                cv_array = self[i].cv
            else:
                cv_array = np.vstack((cv_array, self[i].cv))

        return np.asarray(cv_array)

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

    def extract_pf(self):
        return self[0].pareto_front

    def extract_nondominated(self):
        """
        Extract non-dominated solutions from the population.

        Can only run once all objectives, constraints have been evaluated i.e after a call to self.evaluate(x).

        Creates a new population which is a subset of the original.
        """
        # Number of best-ranked solutions.
        num_best = np.count_nonzero(self.extract_rank() == 0)

        # Initialize new population.
        obj = self.__new__(Population, self[0].problem, n_individuals=num_best)

        # Loop through and save.
        best_ctr = 0
        for i in range(len(self)):
            if self[i].rank == 0:
                obj[best_ctr] = self[i]
                best_ctr += 1
        return obj

    def extract_feasible(self):
        """
        Extract feasible solutions from the population.

        Can only run once all objectives, constraints have been evaluated i.e after a call to self.evaluate(x).

        Creates a new population which is a subset of the original.
        """
        # Number of feasible
        num_feas = np.count_nonzero(self.extract_cv() <= 0)

        # Initialize new population.
        obj = self.__new__(Population, self[0].problem, n_individuals=num_feas)

        # Loop through and save.
        best_ctr = 0
        for i in range(len(self)):
            if self[i].cv <= 0:
                obj[feas_ctr] = self[i]
                feas_ctr += 1
        return obj

    def eval_rank_and_crowding(self, constrained=True):
        # Extract the objective function values from the population
        obj_array = self.extract_obj()
        cv_array = self.extract_cv()

        # Conduct non-dominated sorting (considering constraints & objectives)
        if constrained:
            cons_val = cv_array
        else:
            cons_val = None

        fronts = NonDominatedSorting().do(
            obj_array, cons_val=cons_val, n_stop_if_ranked=len(self)
        )

        # Cycle through fronts
        for k, front in enumerate(fronts):
            # Calculate crowding distance of the front

            # LEFT OUT TO SPEED UP COMPUTATION
            # front_crowding_distance = calculate_crowding_distance(obj_array[front, :])

            # Save rank and crowding to the individuals
            for j, i in enumerate(front):
                self[i].rank = k
                # self[i].crowding_distance = front_crowding_distance[j]
                self[i].crowding_distance = None

    ### EVALUATE AT A GIVEN SET OF POINTS.
    def evaluate(self, var_array):
        for i in range(len(self)):
            # Assign decision variables.
            self[i].set_var(var_array[i, :])

            # Run evaluation of objectives, constraints and CV.
            self[i].eval_instance()

        # Now can find rank and crowding of each individual.
        self.eval_rank_and_crowding()
