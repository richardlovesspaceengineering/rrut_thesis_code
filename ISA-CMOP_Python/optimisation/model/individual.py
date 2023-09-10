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

    ### SETTERS
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

    ### EVALUATION FUNCTIONS
    def eval_obj_cons(self):
        # Returns a tuple (obj, cons)
        return self.problem.obj_func_specific(self.var)

    def eval_obj(self):
        # evaluates objectives
        return self.eval_obj_cons()[0]

    def eval_cons(self):
        # evaluates constraints
        return self.eval_obj_cons()[1]

    def eval_cv(self, use_norm=True):
        # Find the constraint violation.

        if use_norm:
            cons = copy.deepcopy(self.cons)
            cons[cons <= 0] = 0  # assuming >= constraints
            return np.linalg.norm(cons)

    def eval_instance(self):
        self.set_obj(self.eval_obj())
        self.set_cons(self.eval_cons())
        self.set_cv(self.eval_cv())
