import numpy as np
from optimisation.setup import Setup
from optimisation.util.reference_directions import UniformReferenceDirection

"""
Implemented Directly from LYO paper & supp material Yuan2020 : 
"A constrained multi-objective evolutionary algorithm using valuable infeasible solutions"
"""


class LYOSetup(Setup):
    """
    Wrapper class for standarised LYO methods
    """
    def __init__(self, dim, n_objectives, parent_class):
        super().__init__()

        # Variable extraction
        self.alpha = parent_class.alpha
        self.beta = parent_class.beta
        self.p1, self.p2 = parent_class.p1, parent_class.p2
        self.tau = parent_class.tau
        self.n = parent_class.n
        self.gamma = parent_class.gamma
        self.q = parent_class.q
        self.a = parent_class.a
        self.b = parent_class.b
        self.l = np.sin(0.5 * np.pi * np.sqrt(self.gamma)) ** 2

        self.dim = dim
        self.n_objectives = n_objectives

    # Shape function  TODO: Check formulation
    def s(self, x):
        s = np.ones(self.n_objectives)
        for i in range(self.n_objectives-1):
            s[i] *= np.prod(x[:self.n_objectives-i-1]**self.q[i])

            if i > 0:
                s[i] *= (1 - x[self.n_objectives-i-1]) ** self.q[i]

        s[-1] = (1 - x[0]) ** self.q[-1]
        return s

    def g1(self, x, d=50.0):
        g2 = self.g2(x)
        term = d ** (0.5*(1 - self.beta)) * g2 ** self.beta * (1 + self.alpha * x[0])
        return term

    def g2(self, x):
        return 50.0 * (self.dim - self.n_objectives + 1 + np.sum((x[self.n_objectives - 1:] - 0.5) ** 2 - np.cos(20 * np.pi * (x[self.n_objectives - 1:] - 0.5))))

    # Constraint Functions
    def c0(self, obj):
        return -np.sign(self.phi_func(obj, self.a))

    def c1(self, x, obj):
        h1 = self.h1(obj)
        psi1 = self.psi1_func(obj)

        cons = self.p1 * h1 * (np.sin(np.pi * psi1 / self.tau) ** 2 - self.l)
        return cons

    def c2(self, obj):
        h1 = self.h1(obj)
        psi2 = self.psi2_func(obj)

        cons = self.p2 * h1 * (np.sin(np.pi * self.n * psi2) ** 2 - self.l)
        return cons

    def h1(self, obj):
        return np.maximum(np.sign(self.h2(obj)), 0)

    def h2(self, obj):
        return self.phi_func(obj, self.a) * self.phi_func(obj, self.a + self.b)

    @staticmethod
    def phi_func(obj, a):
        return np.sum(obj ** 2 / a ** 2) - 1

    @staticmethod
    def psi1_func(obj):
        return np.sum(obj)

    def psi2_func(self, obj):
        inner_term = np.max(obj / np.sum(obj) - 1 / self.n_objectives)
        psi2 = (self.n_objectives / (self.n_objectives - 1)) * inner_term
        return psi2


class LYO1(LYOSetup):
    """
    LYO1: "supp: A constrained multi-objective evolutionary algorithm using valuable infeasible solutions"
    """

    def __init__(self, n_obj=3):
        self.problem_name = 'LYO1'
        self.n_objectives = n_obj
        self.dim = self.n_objectives + 5
        self.n_constraints = 3

        # Problem variables
        self.alpha = 0.0
        self.beta = 1.0
        self.p1, self.p2 = 1.0, 1.0
        self.tau = 5.0
        self.n = 3.0
        self.gamma = 0.1

        # Problem arrays
        i_range = np.arange(0, self.n_objectives) + 1.0
        self.q = i_range
        self.a = 3*i_range
        self.b = np.ones(self.n_objectives)

        super().__init__(dim=self.dim, n_objectives=self.n_objectives, parent_class=self)

        # Optimum Pareto Front
        self.f_opt = self.a * generic_sphere(get_ref_dirs(self.n_objectives))
        self.var_opt = None

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub, value=0.5 * np.ones(self.dim),
                           scale=1.0, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        prob.add_con_group('con', self.n_constraints, lower=None, upper=None)  # TODO: check bounds

    def set_objectives(self, prob, **kwargs):
        for i in range(self.n_objectives):
            prob.add_obj(f"f_{i}")

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']
        obj, cons = self.obj_func_specific(x)
        # cons = None
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        g1 = self.g1(x)
        s_terms = self.s(x)
        obj = (1 + g1) * s_terms

        cons = self.cons_func_specific(x, obj)
        return obj, cons

    def cons_func_specific(self, x, obj):
        cons = np.zeros(self.n_constraints)

        cons[0] = self.c0(obj)
        cons[1] = self.c1(x, obj)
        cons[2] = self.c2(obj)
        return cons


class LYO2(LYOSetup):
    """
    LYO2: "supp: A constrained multi-objective evolutionary algorithm using valuable infeasible solutions"
    """

    def __init__(self, n_obj=3):
        self.problem_name = 'LYO2'
        self.n_objectives = n_obj
        self.dim = self.n_objectives + 5
        self.n_constraints = 3

        # Problem variables
        self.alpha = 100.0
        self.beta = 1.0
        self.p1, self.p2 = 1.0, 1.0
        self.tau = 30.0
        self.n = 3.0
        self.gamma = 0.1

        # Problem arrays
        i_range = np.arange(0, self.n_objectives) + 1.0
        self.q = 2 * i_range / (self.n_objectives + 1)
        self.a = 2 * self.n_objectives * i_range
        self.b = 2 * np.ones(self.n_objectives)

        super().__init__(dim=self.dim, n_objectives=self.n_objectives, parent_class=self)

        # Optimum Pareto Front
        self.f_opt = self.a * get_ref_dirs(self.n_objectives)
        self.var_opt = None

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub, value=0.5 * np.ones(self.dim),
                           scale=1.0, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        prob.add_con_group('con', self.n_constraints, lower=None, upper=None)  # TODO: check bounds

    def set_objectives(self, prob, **kwargs):
        for i in range(self.n_objectives):
            prob.add_obj(f"f_{i}")

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']
        obj, cons = self.obj_func_specific(x)
        # cons = None
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        g1 = self.g1(x)
        s_terms = self.s(x)
        obj = (1 + g1) * s_terms

        cons = self.cons_func_specific(x, obj)
        return obj, cons

    def cons_func_specific(self, x, obj):
        cons = np.zeros(self.n_constraints)

        cons[0] = self.c0(obj)
        cons[1] = self.c1(x, obj)
        cons[2] = self.c2(obj)
        return cons

    @staticmethod
    def phi_func(obj, a):
        return np.sum(obj / a) - 1

    @staticmethod
    def psi1_func(obj):
        return np.sum(np.sqrt(obj)) ** 2


class LYO3(LYOSetup):
    """
    LYO3: "supp: A constrained multi-objective evolutionary algorithm using valuable infeasible solutions"
    """

    def __init__(self, n_obj=3):
        self.problem_name = 'LYO3'
        self.n_objectives = n_obj
        self.dim = self.n_objectives + 9
        self.n_constraints = 3

        # Problem variables
        self.alpha = 0.0
        self.beta = 1.0
        self.p1, self.p2 = 10.0, 1.0
        self.tau = 50.0
        self.n = 6.0
        self.gamma = 0.1

        # Problem arrays
        self.q = np.ones(self.n_objectives)
        self.a = 30 * np.ones(self.n_objectives)
        self.b = np.ones(self.n_objectives)

        super().__init__(dim=self.dim, n_objectives=self.n_objectives, parent_class=self)

        # Optimum Pareto Front
        self.f_opt = self.a * concave_sphere(get_ref_dirs(self.n_objectives))
        self.var_opt = None

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub, value=0.5 * np.ones(self.dim),
                           scale=1.0, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        prob.add_con_group('con', self.n_constraints, lower=None, upper=None)  # TODO: check bounds

    def set_objectives(self, prob, **kwargs):
        for i in range(self.n_objectives):
            prob.add_obj(f"f_{i}")

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']
        obj, cons = self.obj_func_specific(x)
        # cons = None
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        g1 = self.g1(x)
        s_terms = self.s(x)
        obj = (1 + g1) * s_terms

        cons = self.cons_func_specific(x, obj)
        return obj, cons

    def cons_func_specific(self, x, obj):
        cons = np.zeros(self.n_constraints)

        cons[0] = self.c0(obj)
        cons[1] = self.c1(x, obj)
        cons[2] = self.c2(obj)
        return cons

    @staticmethod
    def phi_func(obj, a):
        return np.sum(np.sqrt(obj / a)) - 1

    @staticmethod
    def psi1_func(obj):
        return np.sqrt(np.sum(obj**2))


class LYO4(LYOSetup):
    """
    LYO4: "supp: A constrained multi-objective evolutionary algorithm using valuable infeasible solutions"
    """

    def __init__(self, n_obj=3):
        self.problem_name = 'LYO4'
        self.n_objectives = n_obj
        self.dim = self.n_objectives + 9
        self.n_constraints = 3

        # Problem variables
        self.alpha = 0.0
        self.beta = 1.0
        self.p1, self.p2 = 1.0, 10.0
        self.tau = 30.0
        self.n = 3.0
        self.gamma = 0.001
        self.k = np.maximum(4, 15 - self.n_objectives)

        # Problem arrays
        i_range = np.arange(0, self.n_objectives) + 1.0
        self.q = 0.5 * np.ones(self.n_objectives)
        self.a = 20.0 + 10 * i_range
        self.b = 0.01 * np.ones(self.n_objectives)

        super().__init__(dim=self.dim, n_objectives=self.n_objectives, parent_class=self)

        # Optimum Pareto Front
        self.f_opt = self.a * get_ref_dirs(self.n_objectives, n_points=[self.k+1, self.k+1])
        self.var_opt = None

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub, value=0.5 * np.ones(self.dim),
                           scale=1.0, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        prob.add_con_group('con', self.n_constraints, lower=None, upper=None)  # TODO: check bounds

    def set_objectives(self, prob, **kwargs):
        for i in range(self.n_objectives):
            prob.add_obj(f"f_{i}")

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']
        obj, cons = self.obj_func_specific(x)
        # cons = None
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        g1 = self.g1(x)
        s_terms = self.s(x)
        obj = (1 + g1) * s_terms

        cons = self.cons_func_specific(x, obj)
        return obj, cons

    def cons_func_specific(self, x, obj):
        cons = np.zeros(self.n_constraints)

        cons[0] = self.c0(obj)
        cons[1] = self.c1(x, obj)
        cons[2] = self.c2(obj)
        return cons

    def phi_func(self, obj, a):
        return np.floor(np.sum(np.sqrt(self.k * obj / a))) - self.k

    @staticmethod
    def psi1_func(obj):
        term1 = np.sqrt(np.sum(4 * obj ** 2))
        term2 = obj ** 2
        for i in range(len(obj)):
            term2[i] += np.sum(16 * obj[np.arange(len(obj)) != i] ** 2)
        term2 = np.sqrt(np.min(term2))
        return np.minimum(term1, term2)


class LYO5(LYOSetup):
    """
    LYO5: "supp: A constrained multi-objective evolutionary algorithm using valuable infeasible solutions"
    """

    def __init__(self, n_obj=3):
        self.problem_name = 'LYO5'
        self.n_objectives = n_obj
        self.dim = self.n_objectives + 5
        self.n_constraints = 3

        # Problem variables
        self.alpha = 0.0
        self.beta = -1.0
        self.p1, self.p2 = 1.0, 1.0
        self.tau = 0.2
        self.n = 3.0
        self.gamma = 0.5

        # Problem arrays
        i_range = np.arange(0, self.n_objectives) + 1.0
        self.q = 1 / i_range
        self.a = 10 * np.sqrt(self.n_objectives) * np.ones(self.n_objectives)
        self.b = np.ones(self.n_objectives)

        super().__init__(dim=self.dim, n_objectives=self.n_objectives, parent_class=self)

        # Optimum Pareto Front
        self.f_opt = self.a * get_ref_dirs(self.n_objectives)
        self.var_opt = None

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub, value=0.5 * np.ones(self.dim),
                           scale=1.0, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        prob.add_con_group('con', self.n_constraints, lower=None, upper=None)  # TODO: check bounds

    def set_objectives(self, prob, **kwargs):
        for i in range(self.n_objectives):
            prob.add_obj(f"f_{i}")

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']
        obj, cons = self.obj_func_specific(x)
        # cons = None
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        g1 = self.g1(x)
        s_terms = self.s(x)
        obj = (1 + g1) * s_terms

        cons = self.cons_func_specific(x, obj)
        return obj, cons

    def cons_func_specific(self, x, obj):
        cons = np.zeros(self.n_constraints)

        cons[0] = self.c0(obj)
        cons[1] = self.c1(x, obj)
        cons[2] = self.c2(obj)
        return cons

    @staticmethod
    def phi_func(obj, a):
        return np.sum(np.sqrt(obj / a)) - 1

    @staticmethod
    def psi1_func(obj):
        return np.sum(np.sqrt(obj)) ** 2


class LYO6(LYOSetup):
    """
    LYO6: "supp: A constrained multi-objective evolutionary algorithm using valuable infeasible solutions"
    """

    def __init__(self, n_obj=3):
        self.problem_name = 'LYO6'
        self.n_objectives = n_obj
        self.dim = self.n_objectives + 5
        self.n_constraints = 3

        # Problem variables
        self.alpha = 0.0
        self.beta = -1.0
        self.p1, self.p2 = 1.0, 10.0
        self.tau = 0.2
        self.n = 6.0
        self.gamma = 0.1

        # Problem arrays
        self.q = 2 * np.ones(self.n_objectives)
        self.a = 6.5 * np.ones(self.n_objectives)
        self.b = np.ones(self.n_objectives)

        super().__init__(dim=self.dim, n_objectives=self.n_objectives, parent_class=self)

        # Optimum Pareto Front
        self.f_opt = self.a * concave_sphere(get_ref_dirs(self.n_objectives))
        self.var_opt = None

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub, value=0.5 * np.ones(self.dim),
                           scale=1.0, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        prob.add_con_group('con', self.n_constraints, lower=None, upper=None)  # TODO: check bounds

    def set_objectives(self, prob, **kwargs):
        for i in range(self.n_objectives):
            prob.add_obj(f"f_{i}")

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']
        obj, cons = self.obj_func_specific(x)
        # cons = None
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        g1 = self.g1(x)
        s_terms = self.s(x)
        obj = (1 + g1) * s_terms

        cons = self.cons_func_specific(x, obj)
        return obj, cons

    def cons_func_specific(self, x, obj):
        cons = np.zeros(self.n_constraints)

        cons[0] = self.c0(obj)
        cons[1] = self.c1(x, obj)
        cons[2] = self.c2(obj)
        return cons

    @staticmethod
    def phi_func(obj, a):
        return np.sum(obj / a) - 1


class LYO7(LYOSetup):
    """
    LYO7: "supp: A constrained multi-objective evolutionary algorithm using valuable infeasible solutions"
    """

    def __init__(self, n_obj=3):
        self.problem_name = 'LYO7'
        self.n_objectives = n_obj
        self.dim = self.n_objectives + 5
        self.n_constraints = 3

        # Problem variables
        self.alpha = 10.0
        self.beta = -1.0
        self.p1, self.p2 = 10.0, 1.0
        self.tau = 1.0
        self.n = 3.0
        self.gamma = 0.2

        # Problem arrays
        self.q = np.ones(self.n_objectives)
        self.a = 50.0 * np.ones(self.n_objectives)
        self.b = np.ones(self.n_objectives)

        super().__init__(dim=self.dim, n_objectives=self.n_objectives, parent_class=self)

        # Optimum Pareto Front
        self.f_opt = self.a * generic_sphere(get_ref_dirs(self.n_objectives))
        self.var_opt = None

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub, value=0.5 * np.ones(self.dim),
                           scale=1.0, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        prob.add_con_group('con', self.n_constraints, lower=None, upper=None)  # TODO: check bounds

    def set_objectives(self, prob, **kwargs):
        for i in range(self.n_objectives):
            prob.add_obj(f"f_{i}")

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']
        obj, cons = self.obj_func_specific(x)
        # cons = None
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        g1 = self.g1(x)
        s_terms = self.s(x)
        obj = (1 + g1) * s_terms

        cons = self.cons_func_specific(x, obj)
        return obj, cons

    def cons_func_specific(self, x, obj):
        cons = np.zeros(self.n_constraints)

        cons[0] = self.c0(obj)
        cons[1] = self.c1(x, obj)
        cons[2] = self.c2(obj)
        return cons

    @staticmethod
    def psi1_func(obj):
        return np.sqrt(np.sum(obj**2))


class LYO8(LYOSetup):
    """
    LYO8: "supp: A constrained multi-objective evolutionary algorithm using valuable infeasible solutions"
    """

    def __init__(self, n_obj=3):
        self.problem_name = 'LYO8'
        self.n_objectives = n_obj
        self.dim = self.n_objectives + 5
        self.n_constraints = 3

        # Problem variables
        self.alpha = 0.0
        self.beta = -1.0
        self.p1, self.p2 = 1.0, 10.0
        self.tau = 0.2
        self.n = 6.0
        self.gamma = 0.01
        self.k = np.maximum(4, 15 - self.n_objectives)

        # Problem arrays
        i_range = np.arange(0, self.n_objectives) + 1.0
        self.q = 2 * i_range / (self.n_objectives + 1)
        self.a = 10.0 * np.ones(self.n_objectives)
        self.b = np.ones(self.n_objectives)

        super().__init__(dim=self.dim, n_objectives=self.n_objectives, parent_class=self)

        # Optimum Pareto Front
        self.f_opt = self.a * get_ref_dirs(self.n_objectives, n_points=[self.k+1, self.k+1])
        self.var_opt = None

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub, value=0.5 * np.ones(self.dim),
                           scale=1.0, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        prob.add_con_group('con', self.n_constraints, lower=None, upper=None)  # TODO: check bounds

    def set_objectives(self, prob, **kwargs):
        for i in range(self.n_objectives):
            prob.add_obj(f"f_{i}")

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']
        obj, cons = self.obj_func_specific(x)
        # cons = None
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        g1 = self.g1(x)
        s_terms = self.s(x)
        obj = (1 + g1) * s_terms

        cons = self.cons_func_specific(x, obj)
        return obj, cons

    def cons_func_specific(self, x, obj):
        cons = np.zeros(self.n_constraints)

        cons[0] = self.c0(obj)
        cons[1] = self.c1(x, obj)
        cons[2] = self.c2(obj)
        return cons

    def phi_func(self, obj, a):
        return np.floor(np.sum(np.sqrt(self.k * obj / a))) - self.k

    @staticmethod
    def psi1_func(obj):
        term1 = np.sqrt(np.sum(4 * obj ** 2))
        term2 = obj ** 2
        for i in range(len(obj)):
            term2[i] += np.sum(16 * obj[np.arange(len(obj)) != i] ** 2)
        term2 = np.sqrt(np.min(term2))
        return np.minimum(term1, term2)


# Utils ----------------------------------------------------------------------------------------------------------------


def get_ref_dirs(n_obj, n_points=None):
    if n_points is None:
        n_points = [100, 15]
    if n_obj == 2:
        ref_dirs = UniformReferenceDirection(2, n_partitions=n_points[0]).do()
    elif n_obj == 3:
        ref_dirs = UniformReferenceDirection(3, n_partitions=n_points[1]).do()
    else:
        raise Exception("Please provide reference directions for more than 3 objectives!")
    return ref_dirs


def generic_sphere(ref_dirs):
    return ref_dirs / np.tile(np.linalg.norm(ref_dirs, axis=1)[:, None], (1, ref_dirs.shape[1]))


def concave_sphere(ref_dirs):
    return ref_dirs * np.tile(np.linalg.norm(ref_dirs, axis=1)[:, None], (1, ref_dirs.shape[1]))
