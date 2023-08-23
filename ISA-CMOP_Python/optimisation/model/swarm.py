import numpy as np

from optimisation.model.population import Population
from optimisation.model.particle import Particle

from optimisation.operators.survival.rank_and_crowding_survival import RankAndCrowdingSurvival


class Swarm(Population):

    def __new__(cls, problem, n_individuals=0):
        obj = super(Population, cls).__new__(cls, n_individuals, dtype=cls).view(cls)
        for i in range(n_individuals):
            obj[i] = Particle(problem)

        return obj

    @classmethod
    def merge(cls, a, b):

        if isinstance(a, Swarm) and isinstance(b, Swarm):
            if len(a) == 0:
                return b
            elif len(b) == 0:
                return a
            else:
                obj = np.concatenate([a, b]).view(Swarm)
                return obj
        else:
            raise Exception('Both a and b must be Swarm instances')


    def compute_local_optima(self, problem):

        local_optima_pos_array = []
        for i in range(len(self)):
            self[i].compute_optimum(problem)
            local_optima_pos_array.append(self[i].optimum.var)

        return np.asarray(local_optima_pos_array)

    def compute_global_optimum(self, problem, n_individuals=0, survival=None):

        # Survival method
        if survival is None:
            survival = RankAndCrowdingSurvival(filter_infeasible=True)

        # Extract particle optimum positions
        optimum_pos_array = []
        optimum_obj_array = []
        optimum_cons_array = []
        for i in range(len(self)):
            if i == 0:
                optimum_pos_array = self[i].optimum.var
                optimum_obj_array = self[i].optimum.obj
                optimum_cons_array = self[i].optimum.cons
            else:
                optimum_pos_array = np.vstack((optimum_pos_array, self[i].optimum.var))
                optimum_obj_array = np.vstack((optimum_obj_array, self[i].optimum.obj))
                optimum_cons_array = np.vstack((optimum_cons_array, self[i].optimum.cons))

        # Dummy population
        dummy_pop = Swarm(problem, n_individuals)

        # Assign swarm optimum variables, objectives and constraints to dummy population
        dummy_pop.assign_var(problem, optimum_pos_array)
        dummy_pop.assign_obj(optimum_obj_array)
        dummy_pop.assign_cons(optimum_cons_array)

        # Use survival method to compute global best
        global_optimum_individual = survival.do(problem, dummy_pop, 1, None, None)

        return global_optimum_individual[0], global_optimum_individual[0].var

    def extract_velocity(self):

        velocity_array = []
        for i in range(len(self)):
            velocity_array.append(self[i].velocity)

        return np.asarray(velocity_array)

    def assign_velocity(self, velocity_array):

        for i in range(len(self)):
            self[i].velocity = velocity_array[i, :]

    def extract_personal_best(self):
        pbest_array = []
        for i in range(len(self)):
            pbest_array.append(self[i].personal_best)

        return np.asarray(pbest_array)

    def assign_personal_best(self, pbest_array):
        for i in range(len(self)):
            self[i].personal_best = pbest_array[i, :]