import numpy as np
from optimisation.setup import Setup
from optimisation.util.reference_directions import UniformReferenceDirection
from optimisation.util.non_dominated_sorting import NonDominatedSorting

from itertools import combinations

"""
Ported the WFG implementation across from Pymoo
"""


class WFGSetup(Setup):
    """
    Wrapper class for standarised WFG methods
    """
    def __init__(self, dim, n_objectives, k=None, l=None, lb=None, ub=None):
        super().__init__()
        self.lb = lb
        self.ub = ub

        self.dim = dim
        self.n_objectives = n_objectives

        self.S = np.arange(2, 2 * self.n_objectives + 1, 2)
        self.A = np.ones(self.n_objectives - 1)

        # Variable inputs
        if k:
            self.k = k
        else:
            if self.n_objectives == 2:
                self.k = 4
            else:
                self.k = 2 * (n_objectives - 1)
        if l:
            self.l = l
        else:
            self.l = self.dim - self.k

        # Check provided input variables are correct
        self.validate(self.l, self.k, self.n_objectives)

    @staticmethod
    def validate(l, k, n_obj):
        if n_obj < 2:
            raise ValueError('WFG problems must have two or more objectives.')
        if not k % (n_obj - 1) == 0:
            raise ValueError('Position parameter (k) must be divisible by number of objectives minus one.')
        if k < 4:
            raise ValueError('Position parameter (k) must be greater or equal than 4.')
        if (k + l) < n_obj:
            raise ValueError('Sum of distance and position parameters must be greater than num. of objs. (k + l >= M).')

    @staticmethod
    def _post(t, a):
        x = []
        for i in range(len(t) - 1):
            x.append(np.maximum(t[-1], a[i]) * (t[i] - 0.5) + 0.5)
        x.append(t[-1])
        return np.array(x)

    @staticmethod
    def _calculate(x, s, h):
        return x[-1] + s * h

    # WFG1 Methods
    @staticmethod
    def t1_1(x, n, k):
        x[k:n] = _transformation_shift_linear(x[k:n], 0.35)
        return x

    @staticmethod
    def t2_1(x, n, k):
        x[k:n] = _transformation_bias_flat(x[k:n], 0.8, 0.75, 0.85)
        return x

    @staticmethod
    def t3_1(x, n):
        x[:n] = _transformation_bias_poly(x[:n], 0.02)
        return x

    @staticmethod
    def t4_1(x, m, n, k):
        w = np.arange(2, 2 * n + 1, 2)
        gap = k // (m - 1)
        t = np.array([])
        for m in range(1, m):
            _y = x[(m - 1) * gap: (m * gap)]
            _w = w[(m - 1) * gap: (m * gap)]
            t = np.hstack((t, _reduction_weighted_sum(_y, _w)))
        t = np.hstack((t, _reduction_weighted_sum(x[k:n], w[k:n])))
        return t

    # WFG2 Methods
    @staticmethod
    def t2_2(x, n, k):
        y = np.array([x[i] for i in range(k)]).flatten()

        l = n - k
        ind_non_sep = k + l // 2

        i = k + 1
        while i <= ind_non_sep:
            head = k + 2 * (i - k) - 2
            tail = k + 2 * (i - k)
            y = np.hstack((y, _reduction_non_sep(x[head:tail], 2)))
            i += 1

        return y

    @staticmethod
    def t3_2(x, m, n, k):
        ind_r_sum = k + (n - k) // 2
        gap = k // (m - 1)

        t = np.array([_reduction_weighted_sum_uniform(x[(m - 1) * gap: (m * gap)]) for m in range(1, m)]).flatten()
        t = np.hstack((t, _reduction_weighted_sum_uniform(x[k:ind_r_sum])))

        return t

    # WFG4 Methods
    @staticmethod
    def t1_4(x):
        return _transformation_shift_multi_modal(x, 30.0, 10.0, 0.35)

    @staticmethod
    def t2_4(x, m, k):
        gap = k // (m - 1)
        t = np.array([_reduction_weighted_sum_uniform(x[(m - 1) * gap: (m * gap)]) for m in range(1, m)]).flatten()
        t = np.hstack((t, _reduction_weighted_sum_uniform(x[k:])))
        return t

    # WFG5 Methods
    @staticmethod
    def t1_5(x):
        return _transformation_param_deceptive(x, A=0.35, B=0.001, C=0.05)

    # WFG6 Methods
    @staticmethod
    def t2_6(x, m, n, k):
        gap = k // (m - 1)
        t = np.array([_reduction_non_sep(x[(m - 1) * gap: (m * gap)], gap) for m in range(1, m)]).flatten()
        t = np.hstack((t, _reduction_non_sep(x[k:], n - k)))
        return t

    # WFG7 Methods
    @staticmethod
    def t1_7(x, k):
        for i in range(k):
            aux = _reduction_weighted_sum_uniform(x[i + 1:])
            x[i] = _transformation_param_dependent(x[i], aux)
        return x

    # WFG8 Methods
    @staticmethod
    def t1_8(x, n, k):
        ret = np.array([])
        for i in range(k, n):
            aux = _reduction_weighted_sum_uniform(x[:i])
            ret = np.hstack((ret, _transformation_param_dependent(x[i], aux, A=0.98 / 49.98, B=0.02, C=50.0)))
        return ret

    # WFG9 Methods
    @staticmethod
    def t1_9(x, n):
        ret = np.array([])
        for i in range(0, n - 1):
            aux = _reduction_weighted_sum_uniform(x[i + 1:])
            ret = np.hstack((ret, _transformation_param_dependent(x[i], aux)))
        return ret

    @staticmethod
    def t2_9(x, n, k):
        a = [_transformation_shift_deceptive(x[i], 0.35, 0.001, 0.05) for i in range(k)]
        b = [_transformation_shift_multi_modal(x[i], 30.0, 95.0, 0.35) for i in range(k, n)]
        return np.array(a + b)

    @staticmethod
    def t3_9(x, m, n, k):
        gap = k // (m - 1)
        t = np.array([_reduction_non_sep(x[(m - 1) * gap: (m * gap)], gap) for m in range(1, m)]).flatten()
        t = np.hstack((t, _reduction_non_sep(x[k:], n - k)))
        return t


class WFG1(WFGSetup):
    """
    WFG1: "A Review of Multi-objective Test Problems and a Scalable Test Problem Toolkit"
    """

    def __init__(self, n_dim=10, n_obj=3):
        self.problem_name = 'WFG1'
        self.dim = n_dim
        self.n_objectives = n_obj
        self.n_constraints = 0

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.lb = np.zeros(self.dim)
        self.ub = 2 * np.arange(1, self.dim + 1)

        super().__init__(dim=n_dim, n_objectives=self.n_objectives, lb=self.lb, ub=self.ub)

        # Optimum Pareto Front
        self.f_opt = self.exact_pareto(50)
        self.var_opt = None

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub, value=np.ones(self.dim),
                           scale=1.0, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        pass
        # prob.add_con_group('con', self.n_constraints, lower=None, upper=None)

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
        y = x / self.ub

        y = self.t1_1(y, self.dim, self.k)
        y = self.t2_1(y, self.dim, self.k)
        y = self.t3_1(y, self.dim)
        y = self.t4_1(y, self.n_objectives, self.dim, self.k)

        y = self._post(y, self.A)

        h = [_shape_convex(y[:-1], m + 1) for m in range(self.n_objectives - 1)]
        h.append(_shape_mixed(y[0], alpha=1.0, A=5.0))

        obj = self._calculate(y, self.S, np.array(h))

        return obj

    def cons_func_specific(self, x):
        pass

    def exact_pareto(self, n_points):
        if self.n_objectives == 3:
            a = np.linspace(0, np.pi / 2, n_points).reshape(-1, 1)
            x = (1 - np.cos(a)) * (1 - np.cos(a.flatten()))
            y = (1 - np.cos(a)) * (1 - np.sin(a.flatten()))
            z = 1 - a * np.ones(len(a))*2 / np.pi - np.cos(20*a*np.ones(len(a)) + np.pi / 2) / 10 / np.pi
            pf = np.vstack((2*x.flatten(), 4*y.flatten(), 6*z.flatten())).T

        elif self.n_objectives == 2:
            pf = None
        else:
            pf = None
        return pf


class WFG2(WFGSetup):
    """
    WFG2: "A Review of Multi-objective Test Problems and a Scalable Test Problem Toolkit"
    """

    def __init__(self, n_dim=10, n_obj=3):
        self.problem_name = 'WFG2'
        self.dim = n_dim
        self.n_objectives = n_obj
        self.n_constraints = 0

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.lb = np.zeros(self.dim)
        self.ub = 2 * np.arange(1, self.dim + 1)

        super().__init__(dim=n_dim, n_objectives=self.n_objectives, lb=self.lb, ub=self.ub)

        # Optimum Pareto Front
        self.f_opt = self.exact_pareto(50)
        self.var_opt = None

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub, value=np.ones(self.dim),
                           scale=1.0, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        pass
        # prob.add_con_group('con', self.n_constraints, lower=None, upper=None)

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
        y = x / self.ub

        y = self.t1_1(y, self.dim, self.k)
        y = self.t2_2(y, self.dim, self.k)
        y = self.t3_2(y, self.n_objectives, self.dim, self.k)

        y = self._post(y, self.A)

        h = [_shape_convex(y[:-1], m + 1) for m in range(self.n_objectives - 1)]
        h.append(_shape_disconnected(y[0], alpha=1.0, beta=1.0, A=5.0))

        obj = self._calculate(y, self.S, np.array(h))

        return obj

    def cons_func_specific(self, x):
        pass

    def exact_pareto(self, n_points):
        if self.n_objectives == 3:
            a = np.linspace(0, np.pi / 2, n_points).reshape(-1, 1)
            x = (1 - np.cos(a)) * (1 - np.cos(a.flatten()))
            y = (1 - np.cos(a)) * (1 - np.sin(a.flatten()))
            z = 1 - a * np.ones(len(a))*2 / np.pi - np.cos(10*a*np.ones(len(a)))**2
            obj_array = np.vstack((2*x.flatten(), 4*y.flatten(), 6*z.flatten())).T
            fronts = NonDominatedSorting().do(obj_array)
            pf = obj_array[fronts[0]]
        elif self.n_objectives == 2:
            pf = None
        else:
            pf = None
        return pf


class WFG3(WFGSetup):
    """
    WFG3: "A Review of Multi-objective Test Problems and a Scalable Test Problem Toolkit"
    """

    def __init__(self, n_dim=10, n_obj=3):
        self.problem_name = 'WFG3'
        self.dim = n_dim
        self.n_objectives = n_obj
        self.n_constraints = 0

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.lb = np.zeros(self.dim)
        self.ub = 2 * np.arange(1, self.dim + 1)

        super().__init__(dim=n_dim, n_objectives=self.n_objectives, lb=self.lb, ub=self.ub)
        self.A[1:] = 0

        # Optimum Pareto Front
        self.f_opt = self.exact_pareto(50)
        self.var_opt = None

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub, value=np.ones(self.dim),
                           scale=1.0, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        pass
        # prob.add_con_group('con', self.n_constraints, lower=None, upper=None)

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
        y = x / self.ub

        y = self.t1_1(y, self.dim, self.k)
        y = self.t2_2(y, self.dim, self.k)
        y = self.t3_2(y, self.n_objectives, self.dim, self.k)

        y = self._post(y, self.A)

        h = [_shape_linear(y[:-1], m + 1) for m in range(self.n_objectives)]

        obj = self._calculate(y, self.S, np.array(h))

        return obj

    def cons_func_specific(self, x):
        pass

    def validate(self, l, k, n_obj):
        super().validate(l, k, n_obj)
        validate_wfg2_wfg3(l)

    def exact_pareto(self, n_points):
        if self.n_objectives == 2:
            x = np.linspace(0, 1, n_points).reshape(-1, 1)
            pf = np.hstack((2*x, 4*np.flipud(x)))
        elif self.n_objectives == 3:
            x = np.linspace(0, 1, n_points).reshape(-1, 1)
            pf = np.hstack((x, 2*x, 6*np.flipud(x)))
        else:
            pf = None
        return pf


class WFG4(WFGSetup):
    """
    WFG4: "A Review of Multi-objective Test Problems and a Scalable Test Problem Toolkit"
    """

    def __init__(self, n_dim=10, n_obj=3):
        self.problem_name = 'WFG4'
        self.dim = n_dim
        self.n_objectives = n_obj
        self.n_constraints = 0

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.lb = np.zeros(self.dim)
        self.ub = 2 * np.arange(1, self.dim + 1)

        super().__init__(dim=n_dim, n_objectives=self.n_objectives, lb=self.lb, ub=self.ub)

        # Optimum Pareto Front
        self.f_opt = generic_sphere(get_ref_dirs(self.n_objectives)) * self.S
        self.var_opt = None

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub, value=np.ones(self.dim),
                           scale=1.0, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        pass
        # prob.add_con_group('con', self.n_constraints, lower=None, upper=None)

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
        y = x / self.ub

        y = self.t1_4(y)
        y = self.t2_4(y, self.n_objectives, self.k)

        y = self._post(y, self.A)

        h = [_shape_concave(y[:-1], m + 1) for m in range(self.n_objectives)]

        obj = self._calculate(y, self.S, np.array(h))

        return obj

    def cons_func_specific(self, x):
        pass


class WFG5(WFGSetup):
    """
    WFG5: "A Review of Multi-objective Test Problems and a Scalable Test Problem Toolkit"
    """

    def __init__(self, n_dim=10, n_obj=2):
        self.problem_name = 'WFG5'
        self.dim = n_dim
        self.n_objectives = n_obj
        self.n_constraints = 0

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.lb = np.zeros(self.dim)
        self.ub = 2 * np.arange(1, self.dim + 1)

        super().__init__(dim=n_dim, n_objectives=self.n_objectives, lb=self.lb, ub=self.ub)

        # Optimum Pareto Front
        self.f_opt = generic_sphere(get_ref_dirs(self.n_objectives)) * self.S
        self.var_opt = None

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub, value=np.ones(self.dim),
                           scale=1.0, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        pass
        # prob.add_con_group('con', self.n_constraints, lower=None, upper=None)

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
        y = x / self.ub

        y = self.t1_5(y)
        y = self.t2_4(y, self.n_objectives, self.k)

        y = self._post(y, self.A)

        h = [_shape_concave(y[:-1], m + 1) for m in range(self.n_objectives)]

        obj = self._calculate(y, self.S, np.array(h))

        return obj

    def cons_func_specific(self, x):
        pass


class WFG6(WFGSetup):
    """
    WFG6: "A Review of Multi-objective Test Problems and a Scalable Test Problem Toolkit"
    """

    def __init__(self, n_dim=10, n_obj=3):
        self.problem_name = 'WFG6'
        self.dim = n_dim
        self.n_objectives = n_obj
        self.n_constraints = 0

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.lb = np.zeros(self.dim)
        self.ub = 2 * np.arange(1, self.dim + 1)

        super().__init__(dim=n_dim, n_objectives=self.n_objectives, lb=self.lb, ub=self.ub)

        # Optimum Pareto Front
        self.f_opt = generic_sphere(get_ref_dirs(self.n_objectives)) * self.S
        self.var_opt = None

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub, value=np.ones(self.dim),
                           scale=1.0, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        pass
        # prob.add_con_group('con', self.n_constraints, lower=None, upper=None)

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
        y = x / self.ub

        y = self.t1_1(y, self.dim, self.k)
        y = self.t2_6(y, self.n_objectives, self.dim, self.k)

        y = self._post(y, self.A)

        h = [_shape_concave(y[:-1], m + 1) for m in range(self.n_objectives)]

        obj = self._calculate(y, self.S, np.array(h))

        return obj

    def cons_func_specific(self, x):
        pass


class WFG7(WFGSetup):
    """
    WFG7: "A Review of Multi-objective Test Problems and a Scalable Test Problem Toolkit"
    """

    def __init__(self, n_dim=10, n_obj=3):
        self.problem_name = 'WFG7'
        self.dim = n_dim
        self.n_objectives = n_obj
        self.n_constraints = 0

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.lb = np.zeros(self.dim)
        self.ub = 2 * np.arange(1, self.dim + 1)

        super().__init__(dim=n_dim, n_objectives=self.n_objectives, lb=self.lb, ub=self.ub)

        # Optimum Pareto Front
        self.f_opt = generic_sphere(get_ref_dirs(self.n_objectives)) * self.S
        self.var_opt = None

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub, value=np.ones(self.dim),
                           scale=1.0, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        pass
        # prob.add_con_group('con', self.n_constraints, lower=None, upper=None)

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
        y = x / self.ub

        y = self.t1_7(y, self.k)
        y = self.t1_1(y, self.dim, self.k)
        y = self.t2_4(y, self.n_objectives, self.k)

        y = self._post(y, self.A)

        h = [_shape_concave(y[:-1], m + 1) for m in range(self.n_objectives)]

        obj = self._calculate(y, self.S, np.array(h))

        return obj

    def cons_func_specific(self, x):
        pass


class WFG8(WFGSetup):
    """
    WFG8: "A Review of Multi-objective Test Problems and a Scalable Test Problem Toolkit"
    """

    def __init__(self, n_dim=10, n_obj=3):
        self.problem_name = 'WFG8'
        self.dim = n_dim
        self.n_objectives = n_obj
        self.n_constraints = 0

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.lb = np.zeros(self.dim)
        self.ub = 2 * np.arange(1, self.dim + 1)

        super().__init__(dim=n_dim, n_objectives=self.n_objectives, lb=self.lb, ub=self.ub)

        # Optimum Pareto Front
        self.f_opt = generic_sphere(get_ref_dirs(self.n_objectives)) * self.S
        self.var_opt = None

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub, value=np.ones(self.dim),
                           scale=1.0, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        pass
        # prob.add_con_group('con', self.n_constraints, lower=None, upper=None)

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
        y = x / self.ub

        y[self.k:self.dim] = self.t1_8(y, self.dim, self.k)
        y = self.t1_1(y, self.dim, self.k)
        y = self.t2_4(y, self.n_objectives, self.k)

        y = self._post(y, self.A)

        h = [_shape_concave(y[:-1], m + 1) for m in range(self.n_objectives)]

        obj = self._calculate(y, self.S, np.array(h))

        return obj

    def cons_func_specific(self, x):
        pass


class WFG9(WFGSetup):
    """
    WFG9: "A Review of Multi-objective Test Problems and a Scalable Test Problem Toolkit"
    """

    def __init__(self, n_dim=10, n_obj=2):
        self.problem_name = 'WFG9'
        self.dim = n_dim
        self.n_objectives = n_obj
        self.n_constraints = 0

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.lb = np.zeros(self.dim)
        self.ub = 2 * np.arange(1, self.dim + 1)

        super().__init__(dim=n_dim, n_objectives=self.n_objectives, lb=self.lb, ub=self.ub)

        # Optimum Pareto Front
        self.f_opt = generic_sphere(get_ref_dirs(self.n_objectives)) * self.S
        self.var_opt = None

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub, value=np.ones(self.dim),
                           scale=1.0, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        pass
        # prob.add_con_group('con', self.n_constraints, lower=None, upper=None)

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
        y = x / self.ub

        y[:self.dim - 1] = self.t1_9(y, self.dim)
        y = self.t2_9(y, self.dim, self.k)
        y = self.t3_9(y, self.n_objectives, self.dim, self.k)

        h = [_shape_concave(y[:-1], m + 1) for m in range(self.n_objectives)]

        obj = self._calculate(y, self.S, np.array(h))

        return obj

    def cons_func_specific(self, x):
        pass

# ---------------------------------------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------------------------------------


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


def powerset(iterable):
    for n in range(len(iterable) + 1):
        yield from combinations(iterable, n)


# ---------------------------------------------------------------------------------------------------------
# TRANSFORMATIONS
# ---------------------------------------------------------------------------------------------------------


def _transformation_shift_linear(value, shift=0.35):
    return correct_to_01(np.fabs(value - shift) / np.fabs(np.floor(shift - value) + shift))


def _transformation_shift_deceptive(y, A=0.35, B=0.005, C=0.05):
    tmp1 = np.floor(y - A + B) * (1.0 - C + (A - B) / B) / (A - B)
    tmp2 = np.floor(A + B - y) * (1.0 - C + (1.0 - A - B) / B) / (1.0 - A - B)
    ret = 1.0 + (np.fabs(y - A) - B) * (tmp1 + tmp2 + 1.0 / B)
    return correct_to_01(ret)


def _transformation_shift_multi_modal(y, A, B, C):
    tmp1 = np.fabs(y - C) / (2.0 * (np.floor(C - y) + C))
    tmp2 = (4.0 * A + 2.0) * np.pi * (0.5 - tmp1)
    ret = (1.0 + np.cos(tmp2) + 4.0 * B * np.power(tmp1, 2.0)) / (B + 2.0)
    return correct_to_01(ret)


def _transformation_bias_flat(y, a, b, c):
    ret = a + np.minimum(0, np.floor(y - b)) * (a * (b - y) / b) \
          - np.minimum(0, np.floor(c - y)) * ((1.0 - a) * (y - c) / (1.0 - c))
    return correct_to_01(ret)


def _transformation_bias_poly(y, alpha):
    return correct_to_01(y ** alpha)


def _transformation_param_dependent(y, y_deg, A=0.98 / 49.98, B=0.02, C=50.0):
    aux = A - (1.0 - 2.0 * y_deg) * np.fabs(np.floor(0.5 - y_deg) + A)
    ret = np.power(y, B + (C - B) * aux)
    return correct_to_01(ret)


def _transformation_param_deceptive(y, A=0.35, B=0.001, C=0.05):
    tmp1 = np.floor(y - A + B) * (1.0 - C + (A - B) / B) / (A - B)
    tmp2 = np.floor(A + B - y) * (1.0 - C + (1.0 - A - B) / B) / (1.0 - A - B)
    ret = 1.0 + (np.fabs(y - A) - B) * (tmp1 + tmp2 + 1.0 / B)
    return correct_to_01(ret)


# ---------------------------------------------------------------------------------------------------------
# REDUCTION
# ---------------------------------------------------------------------------------------------------------


def _reduction_weighted_sum(y, w):
    return correct_to_01(np.dot(y, w) / w.sum())


def _reduction_weighted_sum_uniform(y):
    return correct_to_01(np.array([y.mean()]))


def _reduction_non_sep(y, A):
    # n, m = len(y), 1
    n, m = 1, len(y)
    val = np.ceil(A / 2.0)

    num = np.zeros(n)
    for j in range(m):
        num += y[j]
        for k in range(A - 1):
            num += np.fabs(y[j] - y[(1 + j + k) % m])

    denom = m * val * (1.0 + 2.0 * A - 2 * val) / A

    return correct_to_01(num / denom)


# ---------------------------------------------------------------------------------------------------------
# SHAPE
# ---------------------------------------------------------------------------------------------------------


def _shape_concave(x, m):
    M = len(x)
    if m == 1:
        ret = np.prod(np.sin(0.5 * x[:M] * np.pi))
    elif 1 < m <= M:
        ret = np.prod(np.sin(0.5 * x[:M - m + 1] * np.pi))
        ret *= np.cos(0.5 * x[M - m + 1] * np.pi)
    else:
        ret = np.cos(0.5 * x[0] * np.pi)
    return correct_to_01(ret)


def _shape_convex(x, m):
    M = len(x)
    if m == 1:
        ret = np.prod(1.0 - np.cos(0.5 * x[:M] * np.pi))
    elif 1 < m <= M:
        ret = np.prod(1.0 - np.cos(0.5 * x[:M - m + 1] * np.pi))
        ret *= 1.0 - np.sin(0.5 * x[M - m + 1] * np.pi)
    else:
        ret = 1.0 - np.sin(0.5 * x[0] * np.pi)
    return correct_to_01(ret)


def _shape_linear(x, m):
    M = len(x)
    if m == 1:
        ret = np.prod(x)
    elif 1 < m <= M:
        ret = np.prod(x[:M - m + 1])
        ret *= 1.0 - x[M - m + 1]
    else:
        ret = 1.0 - x[0]
    return correct_to_01(ret)


def _shape_mixed(x, A=5.0, alpha=1.0):
    aux = 2.0 * A * np.pi
    ret = np.power(1.0 - x - (np.cos(aux * x + 0.5 * np.pi) / aux), alpha)
    return correct_to_01(ret)


def _shape_disconnected(x, alpha=1.0, beta=1.0, A=5.0):
    aux = np.cos(A * np.pi * x ** beta)
    return correct_to_01(1.0 - x ** alpha * aux ** 2)


# ---------------------------------------------------------------------------------------------------------
# UTIL
# ---------------------------------------------------------------------------------------------------------

def validate_wfg2_wfg3(l):
    if not l % 2 == 0:
        raise ValueError('In WFG2/WFG3 the distance-related parameter (l) must be divisible by 2.')


def correct_to_01(x, epsilon=1.0e-10):
    if x.ndim > 0:
        x[np.logical_and(x < 0, x >= 0 - epsilon)] = 0
        x[np.logical_and(x > 1, x <= 1 + epsilon)] = 1
    else:
        if 0 > x >= 0 - epsilon:
            x = 0
        elif 1 < x <= 1 + epsilon:
            x = 1
    return x
