import numpy as np

from optimisation.setup import Setup
from optimisation.util.reference_directions import UniformReferenceDirection

"""
Ported the standard DTLZ implementation across from Pymoo
"""


class DTLZSetup(Setup):
    """
    Wrapper class for standarised DTLZ methods
    """
    def __init__(self, dim, n_objectives):
        super().__init__()

        self.dim = dim
        self.n_objectives = n_objectives

        self.k = self.dim - self.n_objectives + 1

    def g1(self, x_m):
        return 100.0 * (len(x_m) + np.sum((x_m - 0.5) ** 2 - np.cos(20 * np.pi * (x_m - 0.5))))

    def g2(self, x_m):
        return np.sum(np.square(x_m - 0.5))

    def g3(self, x):
        contrib = 2.0 * np.power(
            x[self.n_objectives - 1:] + (x[self.n_objectives - 2:-1] - 0.5) * (
                        x[self.n_objectives - 2:-1] - 0.5) - 1.0, 2.0)
        distance = 1 + contrib.sum()
        return distance

    def func_eval(self, x, g, alpha=1):
        f = []

        for i in range(0, self.n_objectives):
            _f = (1 + g)
            _f *= np.prod(np.cos(np.power(x[:len(x) - i], alpha) * np.pi / 2.0))
            if i > 0:
                _f *= np.sin(np.power(x[len(x) - i], alpha) * np.pi / 2.0)

            f.append(_f)

        return np.array(f)


class DTLZ1(DTLZSetup):
    """
    DTLZ1: "Scalable Multi-Objective Optimization Test Problems": Deb 2002
    """

    def __init__(self, n_dim=7):
        self.problem_name = 'DTLZ1'
        self.dim = n_dim
        self.n_objectives = 3
        self.n_constraints = 0
        super().__init__(dim=n_dim, n_objectives=self.n_objectives)

        # Optimum Pareto Front
        self.f_opt = 0.5 * get_ref_dirs(self.n_objectives)
        self.var_opt = None

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub, value=np.ones(self.dim),
                           scale=1.0, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        pass

    def set_objectives(self, prob, **kwargs):
        for i in range(self.n_objectives):
            prob.add_obj(f"f_{i}")

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']
        obj = self.obj_func_specific(x)
        cons = None
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        x_, x_m = x[:self.n_objectives - 1], x[self.n_objectives - 1:]
        g = self.g1(x_m)
        obj = self.func_eval1(x_, g)

        return obj

    def cons_func_specific(self, x):
        return None

    def func_eval1(self, x, g):
        f = []

        for i in range(self.n_objectives):
            _f = 0.5 * np.prod(x[:len(x) - i]) * (1 + g)
            if i > 0:
                _f *= (1 - x[len(x) - i])
            f.append(_f)

        return np.array(f)


class DTLZ2(DTLZSetup):
    """
    DTLZ2: "Scalable Multi-Objective Optimization Test Problems": Deb 2002
    """

    def __init__(self, n_dim=10):
        self.problem_name = 'DTLZ2'
        self.dim = n_dim
        self.n_objectives = 3
        self.n_constraints = 0
        super().__init__(dim=n_dim, n_objectives=self.n_objectives)

        # Optimum Pareto Front
        self.f_opt = generic_sphere(get_ref_dirs(self.n_objectives))
        self.var_opt = None

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub, value=0.5 * np.ones(self.dim),
                           scale=1.0, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        pass

    def set_objectives(self, prob, **kwargs):
        for i in range(self.n_objectives):
            prob.add_obj(f"f_{i}")

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']
        obj = self.obj_func_specific(x)
        cons = None
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        x, x_m = x[:self.n_objectives - 1], x[self.n_objectives - 1:]
        g = self.g2(x_m)
        obj = self.func_eval(x, g, alpha=1)

        return obj

    def cons_func_specific(self, x):
        return None


class DTLZ3(DTLZSetup):
    """
    DTLZ3: "Scalable Multi-Objective Optimization Test Problems": Deb 2002
    """

    def __init__(self, n_dim=10):
        self.problem_name = 'DTLZ3'
        self.dim = n_dim
        self.n_objectives = 3
        self.n_constraints = 0
        super().__init__(dim=n_dim, n_objectives=self.n_objectives)

        # Optimum Pareto Front
        self.f_opt = generic_sphere(get_ref_dirs(self.n_objectives))
        self.var_opt = None

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub, value=0.5 * np.ones(self.dim),
                           scale=1.0, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        pass

    def set_objectives(self, prob, **kwargs):
        for i in range(self.n_objectives):
            prob.add_obj(f"f_{i}")

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']
        obj = self.obj_func_specific(x)
        cons = None
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        x, x_m = x[:self.n_objectives - 1], x[self.n_objectives - 1:]
        g = self.g1(x_m)
        obj = self.func_eval(x, g, alpha=1)

        return obj

    def cons_func_specific(self, x):
        return None


class DTLZ4(DTLZSetup):
    """
    DTLZ4: "Scalable Multi-Objective Optimization Test Problems": Deb 2002
    """

    def __init__(self, n_dim=10, alpha=100):
        self.problem_name = 'DTLZ4'
        self.alpha = alpha
        self.dim = n_dim
        self.n_objectives = 3
        self.n_constraints = 0
        super().__init__(dim=n_dim, n_objectives=self.n_objectives)

        # Optimum Pareto Front
        self.f_opt = generic_sphere(get_ref_dirs(self.n_objectives))
        self.var_opt = None

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub, value=0.5 * np.ones(self.dim),
                           scale=1.0, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        pass

    def set_objectives(self, prob, **kwargs):
        for i in range(self.n_objectives):
            prob.add_obj(f"f_{i}")

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']
        obj = self.obj_func_specific(x)
        cons = None
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        x, x_m = x[:self.n_objectives - 1], x[self.n_objectives - 1:]
        g = self.g2(x_m)
        obj = self.func_eval(x, g, alpha=self.alpha)

        return obj

    def cons_func_specific(self, x):
        return None


class DTLZ5(DTLZSetup):
    """
    DTLZ5: "Scalable Multi-Objective Optimization Test Problems": Deb 2002
    """

    def __init__(self, n_dim=10):
        self.problem_name = 'DTLZ5'
        self.dim = n_dim
        self.n_objectives = 3
        self.n_constraints = 0
        super().__init__(dim=n_dim, n_objectives=self.n_objectives)

        # Optimum Pareto Front
        self.f_opt = np.genfromtxt('./cases/DTLZ_files/dtlz5_3d.txt', delimiter='')
        self.var_opt = None

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub, value=0.5 * np.ones(self.dim),
                           scale=1.0, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        pass

    def set_objectives(self, prob, **kwargs):
        for i in range(self.n_objectives):
            prob.add_obj(f"f_{i}")

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']
        obj = self.obj_func_specific(x)
        cons = None
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        x, x_m = x[:self.n_objectives - 1], x[self.n_objectives - 1:]
        g = self.g2(x_m)

        theta = 1 / (2 * (1 + g)) * (1 + 2 * g * x)
        theta = np.hstack((x[0], theta[1:]))
        obj = self.func_eval(theta, g, alpha=1)

        return obj

    def cons_func_specific(self, x):
        return None


class DTLZ6(DTLZSetup):
    """
    DTLZ6: "Scalable Multi-Objective Optimization Test Problems": Deb 2002
    """

    def __init__(self, n_dim=10):
        self.problem_name = 'DTLZ6'
        self.dim = n_dim
        self.n_objectives = 3
        self.n_constraints = 0
        super().__init__(dim=n_dim, n_objectives=self.n_objectives)

        # Optimum Pareto Front
        self.f_opt = np.genfromtxt('./cases/DTLZ_files/dtlz6_3d.txt', delimiter='')
        self.var_opt = None

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub, value=0.5 * np.ones(self.dim),
                           scale=1.0, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        pass

    def set_objectives(self, prob, **kwargs):
        for i in range(self.n_objectives):
            prob.add_obj(f"f_{i}")

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']
        obj = self.obj_func_specific(x)
        cons = None
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        x, x_m = x[:self.n_objectives - 1], x[self.n_objectives - 1:]
        g = np.sum(np.power(x_m, 0.1))

        theta = 1 / (2 * (1 + g)) * (1 + 2 * g * x)
        theta = np.hstack((x[0], theta[1:]))
        obj = self.func_eval(theta, g, alpha=1)

        return obj

    def cons_func_specific(self, x):
        return None


class DTLZ7(DTLZSetup):
    """
    DTLZ7: "Scalable Multi-Objective Optimization Test Problems": Deb 2002
    """

    def __init__(self, n_dim=10):
        self.problem_name = 'DTLZ7'
        self.dim = n_dim
        self.n_objectives = 3
        self.n_constraints = 0
        super().__init__(dim=n_dim, n_objectives=self.n_objectives)

        # Optimum Pareto Front
        self.f_opt = np.genfromtxt('./cases/DTLZ_files/dtlz7_3d.txt', delimiter='')
        self.var_opt = None

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub, value=0.5 * np.ones(self.dim),
                           scale=1.0, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        pass

    def set_objectives(self, prob, **kwargs):
        for i in range(self.n_objectives):
            prob.add_obj(f"f_{i}")

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']
        obj = self.obj_func_specific(x)
        cons = None
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        f = []
        for i in range(0, self.n_objectives - 1):
            f.append(x[i])
        f = np.array(f)

        g = 1 + (9 / self.k) * np.sum(x[-self.k:])
        h = self.n_objectives - np.sum((f / (1 + g)) * (1 + np.sin(3 * np.pi * f)))

        obj = np.hstack((f, (1 + g) * h))
        return obj

    def cons_func_specific(self, x):
        return None


def get_ref_dirs(n_obj):
    if n_obj == 2:
        ref_dirs = UniformReferenceDirection(2, n_partitions=100).do()
    elif n_obj == 3:
        ref_dirs = UniformReferenceDirection(3, n_partitions=15).do()
    else:
        raise Exception("Please provide reference directions for more than 3 objectives!")
    return ref_dirs


def generic_sphere(ref_dirs):
    return ref_dirs / np.tile(np.linalg.norm(ref_dirs, axis=1)[:, None], (1, ref_dirs.shape[1]))
