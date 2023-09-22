import numpy as np
import copy


class Individual(object):
    def __init__(self, problem):
        """
        Initialize an instance of the problem.

        Problem is an instance of the Setup class.
        """

        # Problem characteristics
        self.problem = problem
        self.n_var = problem.dim
        self.n_obj = problem.n_objectives
        self.n_cons = problem.n_constraints

        self.var_lower = problem.lb
        self.var_upper = problem.ub

        # Initialize arrays.
        self.var = np.zeros((1, self.n_var))
        self.obj = np.zeros((1, self.n_obj))
        self.cons = np.zeros((1, self.n_cons))
        self.cv = np.zeros((1, 1))

        # Exact/approximated pareto front
        self.pareto_front = problem.f_opt

        # Rank, crowding distance & hypervolume
        self.rank = np.nan
        self.crowding_distance = 0.0
        self.hypervolume = 0.0

        # Performance
        self.performance = []

    ### SETTERS (IN CASE WE MAKE ATTRIBUTES PRIVATE)
    def set_var(self, var):
        # var should be a row vector.
        self.var = var

    def set_obj(self, obj):
        # obj should be a row vector.
        self.obj = obj

    def set_cons(self, cons):
        # cons should be a row vector.
        self.cons = cons

    def set_cv(self, cv):
        # cv should be a scalar.
        self.cv = cv

    def set_rank(self, rank):
        # rank should be a scalar.
        self.rank = rank

    def set_crowding_distance(self, crowding_distance):
        self.crowding_distance = crowding_distance

    def set_hypervolume(self, hypervolume):
        self.hypervolume = hypervolume

    def set_performance(self, performance):
        self.performance = performance

    # ADD GETTERS!!!

    ### EVALUATION FUNCTIONS
    def eval_obj_cons(self):
        # Returns a tuple (obj, cons)
        return self.problem.obj_func_specific(self.var)

    def eval_cv(self, use_norm=True):
        # Find the constraint violation.

        if use_norm:
            cons = copy.deepcopy(self.cons)
            cons[cons <= 0] = 0  # assuming >= constraints
            return np.linalg.norm(cons)

    def eval_instance(self):
        obj, cons = self.eval_obj_cons()
        self.set_obj(obj)
        self.set_cons(cons)
        self.set_cv(self.eval_cv())

    # def is_feasible(self):
    #     """
    #     Run after evaluation of constraints/objectives.
    #     """
    #     return self.cv <= 0
