import numpy as np
from optimisation.setup import Setup

"""
Ported the MW implementation across from Pymoo
"""


class MWSetup(Setup):
    """
    Wrapper class for standarised MW methods
    """
    def __init__(self, dim, n_objectives):
        super().__init__()

        self.dim = dim
        self.n_objectives = n_objectives

    def g1(self, x):
        d = self.dim
        n = d - self.n_objectives

        z = np.power(x[self.n_objectives - 1:], n)
        i = np.arange(self.n_objectives - 1, d)

        exp = 1 - np.exp(-10.0 * (z - 0.5 - i / (2 * d)) * (z - 0.5 - i / (2 * d)))
        distance = 1 + exp.sum()
        return distance

    def g2(self, x):
        d = self.dim
        n = d

        i = np.arange(self.n_objectives - 1, d)
        z = 1 - np.exp(-10.0 * (x[self.n_objectives - 1:] - i / n) * (x[self.n_objectives - 1:] - i / n))
        contrib = (0.1 / n) * z * z + 1.5 - 1.5 * np.cos(2 * np.pi * z)
        distance = 1 + contrib.sum()
        return distance

    def g3(self, x):
        contrib = 2.0 * np.power(
            x[self.n_objectives - 1:] + (x[self.n_objectives - 2:-1] - 0.5) * (
                        x[self.n_objectives - 2:-1] - 0.5) - 1.0, 2.0)
        distance = 1 + contrib.sum()
        return distance

    @staticmethod
    def LA1(A, B, C, D, theta):
        return A * np.power(np.sin(B * np.pi * np.power(theta, C)), D)

    @staticmethod
    def LA2(A, B, C, D, theta):
        return A * np.power(np.sin(B * np.power(theta, C)), D)

    @staticmethod
    def LA3(A, B, C, D, theta):
        return A * np.power(np.cos(B * np.power(theta, C)), D)


class MW1(MWSetup):
    """
    MW1: "supp:A constrained multi-objective evolutionary algorithm using valuable infeasible solutions"
    """

    def __init__(self, n_dim=15):
        self.problem_name = 'MW1'
        self.dim = n_dim
        self.n_objectives = 2
        self.n_constraints = 1
        super().__init__(dim=n_dim, n_objectives=self.n_objectives)

        # Optimum Pareto Front
        self.f_opt = np.genfromtxt('./cases/MW_files/MW1_pf.txt', delimiter='')
        self.var_opt = None

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub, value=0.5 * np.ones(self.dim),
                           scale=1.0, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        prob.add_con_group('con', self.n_constraints, lower=None, upper=None) # Todo: check bounds

    def set_objectives(self, prob, **kwargs):
        for i in range(self.n_objectives):
            prob.add_obj(f"f_{i}")

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']
        obj, cons = self.obj_func_specific(x)
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        obj = np.zeros(self.n_objectives)
        cons = np.zeros(self.n_constraints)
        g = self.g1(x)

        obj[0] = x[0]
        obj[1] = g * (1 - 0.85 * obj[0] / g)

        cons[0] = obj[0] + obj[1] - 1 - self.LA1(0.5, 2.0, 1.0, 8.0, np.sqrt(2.0) * obj[1] - np.sqrt(2.0) * obj[0])
        return obj, cons

    def cons_func_specific(self, x):
        return None


class MW2(MWSetup):
    """
    MW2: "supp:A constrained multi-objective evolutionary algorithm using valuable infeasible solutions"
    """

    def __init__(self, n_dim=15):
        self.problem_name = 'MW2'
        self.dim = n_dim
        self.n_objectives = 2
        self.n_constraints = 1
        super().__init__(dim=n_dim, n_objectives=self.n_objectives)

        # Optimum Pareto Front
        self.f_opt = np.genfromtxt('./cases/MW_files/MW2_pf.txt', delimiter='')
        self.var_opt = None

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub, value=0.5 * np.ones(self.dim),
                           scale=1.0, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        prob.add_con_group('con', self.n_constraints, lower=None, upper=None) # Todo: check bounds

    def set_objectives(self, prob, **kwargs):
        for i in range(self.n_objectives):
            prob.add_obj(f"f_{i}")

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']
        obj, cons = self.obj_func_specific(x)
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        obj = np.zeros(self.n_objectives)
        cons = np.zeros(self.n_constraints)
        g = self.g2(x)

        obj[0] = x[0]
        obj[1] = g * (1 - obj[0] / g)

        cons[0] = obj[0] + obj[1] - 1 - self.LA1(0.5, 3.0, 1.0, 8.0, np.sqrt(2.0) * obj[1] - np.sqrt(2.0) * obj[0])
        return obj, cons

    def cons_func_specific(self, x):
        return None


class MW3(MWSetup):
    """
    MW3: "supp:A constrained multi-objective evolutionary algorithm using valuable infeasible solutions"
    """

    def __init__(self, n_dim=15):
        self.problem_name = 'MW3'
        self.dim = n_dim
        self.n_objectives = 2
        self.n_constraints = 2
        super().__init__(dim=n_dim, n_objectives=self.n_objectives)

        # Optimum Pareto Front
        self.f_opt = np.genfromtxt('./cases/MW_files/MW3_pf.txt', delimiter='')
        self.var_opt = None

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub, value=0.5 * np.ones(self.dim),
                           scale=1.0, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        prob.add_con_group('con', self.n_constraints, lower=None, upper=None) # Todo: check bounds

    def set_objectives(self, prob, **kwargs):
        for i in range(self.n_objectives):
            prob.add_obj(f"f_{i}")

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']
        obj, cons = self.obj_func_specific(x)
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        obj = np.zeros(self.n_objectives)
        cons = np.zeros(self.n_constraints)
        g = self.g3(x)
        obj[0] = x[0]
        obj[1] = g * (1 - obj[0] / g)

        cons[0] = obj[0] + obj[1] - 1.05 - self.LA1(0.45, 0.75, 1.0, 6.0, np.sqrt(2.0) * obj[1] - np.sqrt(2.0) * obj[0])
        cons[1] = 0.85 - obj[0] - obj[1] + self.LA1(0.3, 0.75, 1.0, 2.0, np.sqrt(2.0) * obj[1] - np.sqrt(2.0) * obj[0])
        return obj, cons

    def cons_func_specific(self, x):
        return None


class MW4(MWSetup):
    """
    MW4: "supp:A constrained multi-objective evolutionary algorithm using valuable infeasible solutions"
    """

    def __init__(self, n_dim=15):
        self.problem_name = 'MW4'
        self.dim = n_dim
        self.n_objectives = 3
        self.n_constraints = 1
        super().__init__(dim=n_dim, n_objectives=self.n_objectives)

        # Optimum Pareto Front
        self.f_opt = np.genfromtxt('./cases/MW_files/MW4_pf.txt', delimiter='')
        self.var_opt = None

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub, value=0.5 * np.ones(self.dim),
                           scale=1.0, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        prob.add_con_group('con', self.n_constraints, lower=None, upper=None) # Todo: check bounds

    def set_objectives(self, prob, **kwargs):
        for i in range(self.n_objectives):
            prob.add_obj(f"f_{i}")

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']
        obj, cons = self.obj_func_specific(x)
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        g = self.g1(x)
        obj = g.reshape((-1, 1)) * np.ones((x.shape[0], self.n_objectives))
        obj[1:] *= x[(self.n_objectives - 2):-1]
        obj[0:-1] *= np.flip(np.cumprod(1 - x[:(self.n_objectives - 1)]))

        cons = obj.sum() - 1 - self.LA1(0.4, 2.5, 1.0, 8.0, obj[-1] - obj[:-1].sum())
        return obj, cons

    def cons_func_specific(self, x):
        return None


class MW5(MWSetup):
    """
    MW5: "supp:A constrained multi-objective evolutionary algorithm using valuable infeasible solutions"
    """

    def __init__(self, n_dim=15):
        self.problem_name = 'MW5'
        self.dim = n_dim
        self.n_objectives = 2
        self.n_constraints = 3
        super().__init__(dim=n_dim, n_objectives=self.n_objectives)

        # Optimum Pareto Front
        self.f_opt = np.genfromtxt('./cases/MW_files/MW5_pf.txt', delimiter='')
        self.var_opt = None

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub, value=0.5 * np.ones(self.dim),
                           scale=1.0, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        prob.add_con_group('con', self.n_constraints, lower=None, upper=None) # Todo: check bounds

    def set_objectives(self, prob, **kwargs):
        for i in range(self.n_objectives):
            prob.add_obj(f"f_{i}")

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']
        obj, cons = self.obj_func_specific(x)
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        obj = np.zeros(self.n_objectives)
        cons = np.zeros(self.n_constraints)
        g = self.g1(x)
        obj[0] = g * x[0]
        obj[1] = g * np.sqrt(1.0 - np.power(obj[0] / g, 2.0))

        with np.errstate(divide='ignore'):
            atan = np.arctan(obj[1] / obj[0])

        cons[0] = obj[0] ** 2 + obj[1] ** 2 - np.power(1.7 - self.LA2(0.2, 2.0, 1.0, 1.0, atan), 2.0)
        t = 0.5 * np.pi - 2 * np.abs(atan - 0.25 * np.pi)
        cons[1] = np.power(1 + self.LA2(0.5, 6.0, 3.0, 1.0, t), 2.0) - obj[0] ** 2 - obj[1] ** 2
        cons[2] = np.power(1 - self.LA2(0.45, 6.0, 3.0, 1.0, t), 2.0) - obj[0] ** 2 - obj[1] ** 2
        return obj, cons

    def cons_func_specific(self, x):
        return None


class MW6(MWSetup):
    """
    MW6: "supp:A constrained multi-objective evolutionary algorithm using valuable infeasible solutions"
    """

    def __init__(self, n_dim=15):
        self.problem_name = 'MW6'
        self.dim = n_dim
        self.n_objectives = 2
        self.n_constraints = 1
        super().__init__(dim=n_dim, n_objectives=self.n_objectives)

        # Optimum Pareto Front
        self.f_opt = np.genfromtxt('./cases/MW_files/MW6_pf.txt', delimiter='')
        self.var_opt = None

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.lb = np.zeros(self.dim)
        self.ub = 1.1*np.ones(self.dim)

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub, value=0.5 * np.ones(self.dim),
                           scale=1.0, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        prob.add_con_group('con', self.n_constraints, lower=None, upper=None) # Todo: check bounds

    def set_objectives(self, prob, **kwargs):
        for i in range(self.n_objectives):
            prob.add_obj(f"f_{i}")

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']
        obj, cons = self.obj_func_specific(x)
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        obj = np.zeros(self.n_objectives)
        cons = np.zeros(self.n_constraints)
        g = self.g2(x)
        obj[0] = g * x[0]
        obj[1] = g * np.sqrt(1.1 * 1.1 - np.power(obj[0] / g, 2.0))

        with np.errstate(divide='ignore'):
            atan = np.arctan(obj[1] / obj[0])

        cons[0] = obj[0] ** 2 / np.power(1.0 + self.LA3(0.15, 6.0, 4.0, 10.0, atan), 2.0) + obj[1] ** 2 / np.power(
            1.0 + self.LA3(0.75, 6.0, 4.0, 10.0, atan), 2.0) - 1
        return obj, cons

    def cons_func_specific(self, x):
        return None


class MW7(MWSetup):
    """
    MW7: "supp:A constrained multi-objective evolutionary algorithm using valuable infeasible solutions"
    """

    def __init__(self, n_dim=15):
        self.problem_name = 'MW7'
        self.dim = n_dim
        self.n_objectives = 2
        self.n_constraints = 2
        super().__init__(dim=n_dim, n_objectives=self.n_objectives)

        # Optimum Pareto Front
        self.f_opt = np.genfromtxt('./cases/MW_files/MW7_pf.txt', delimiter='')
        self.var_opt = None

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub, value=0.5 * np.ones(self.dim),
                           scale=1.0, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        prob.add_con_group('con', self.n_constraints, lower=None, upper=None) # Todo: check bounds

    def set_objectives(self, prob, **kwargs):
        for i in range(self.n_objectives):
            prob.add_obj(f"f_{i}")

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']
        obj, cons = self.obj_func_specific(x)
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        obj = np.zeros(self.n_objectives)
        cons = np.zeros(self.n_constraints)
        g = self.g3(x)
        obj[0] = g * x[0]
        obj[1] = g * np.sqrt(1 - np.power(obj[0] / g, 2))

        with np.errstate(divide='ignore'):
            atan = np.arctan(obj[1] / obj[0])

        cons[0] = obj[0] ** 2 + obj[1] ** 2 - np.power(1.2 + np.abs(self.LA2(0.4, 4.0, 1.0, 16.0, atan)), 2.0)
        cons[1] = np.power(1.15 - self.LA2(0.2, 4.0, 1.0, 8.0, atan), 2.0) - obj[0] ** 2 - obj[1] ** 2
        return obj, cons

    def cons_func_specific(self, x):
        return None


class MW8(MWSetup):
    """
    MW8: "supp:A constrained multi-objective evolutionary algorithm using valuable infeasible solutions"
    """

    def __init__(self, n_dim=15):
        self.problem_name = 'MW8'
        self.dim = n_dim
        self.n_objectives = 3
        self.n_constraints = 1
        super().__init__(dim=n_dim, n_objectives=self.n_objectives)

        # Optimum Pareto Front
        self.f_opt = np.genfromtxt('./cases/MW_files/MW8_pf.txt', delimiter='')
        self.var_opt = None

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub, value=0.5 * np.ones(self.dim),
                           scale=1.0, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        prob.add_con_group('con', self.n_constraints, lower=None, upper=None) # Todo: check bounds

    def set_objectives(self, prob, **kwargs):
        for i in range(self.n_objectives):
            prob.add_obj(f"f_{i}")

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']
        obj, cons = self.obj_func_specific(x)
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        cons = np.zeros(self.n_constraints)
        g = self.g2(x)
        obj = g.reshape((-1, 1)) * np.ones((x.shape[0], self.n_objectives))
        obj[1:] *= np.sin(0.5 * np.pi * x[(self.n_objectives - 2):-1])
        cos = np.cos(0.5 * np.pi * x[:(self.n_objectives - 1)])
        obj[0:-1] *= np.flip(np.cumprod(cos))

        f_squared = (obj ** 2).sum()
        cons[0] = f_squared - (1.25 - self.LA2(0.5, 6.0, 1.0, 2.0, np.arcsin(obj[-1] / np.sqrt(f_squared)))) * (
                1.25 - self.LA2(0.5, 6.0, 1.0, 2.0, np.arcsin(obj[-1] / np.sqrt(f_squared))))
        return obj, cons

    def cons_func_specific(self, x):
        return None


class MW9(MWSetup):
    """
    MW9: "supp:A constrained multi-objective evolutionary algorithm using valuable infeasible solutions"
    """

    def __init__(self, n_dim=15):
        self.problem_name = 'MW9'
        self.dim = n_dim
        self.n_objectives = 2
        self.n_constraints = 1
        super().__init__(dim=n_dim, n_objectives=self.n_objectives)

        # Optimum Pareto Front
        self.f_opt = np.genfromtxt('./cases/MW_files/MW9_pf.txt', delimiter='')
        self.var_opt = None

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub, value=0.5 * np.ones(self.dim),
                           scale=1.0, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        prob.add_con_group('con', self.n_constraints, lower=None, upper=None) # Todo: check bounds

    def set_objectives(self, prob, **kwargs):
        for i in range(self.n_objectives):
            prob.add_obj(f"f_{i}")

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']
        obj, cons = self.obj_func_specific(x)
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        obj = np.zeros(self.n_objectives)
        cons = np.zeros(self.n_constraints)
        g = self.g1(x)
        obj[0] = g * x[0]
        obj[1] = g * (1.0 - np.power(obj[0] / g, 0.6))

        t1 = (1 - 0.64 * obj[0] * obj[0] - obj[1]) * (1 - 0.36 * obj[0] * obj[0] - obj[1])
        t2 = (1.35 * 1.35 - (obj[0] + 0.35) * (obj[0] + 0.35) - obj[1]) * (1.15 * 1.15 - (obj[0] + 0.15) * (obj[0] + 0.15) - obj[1])
        cons[0] = np.minimum(t1, t2)
        return obj, cons

    def cons_func_specific(self, x):
        return None


class MW10(MWSetup):
    """
    MW10: "supp:A constrained multi-objective evolutionary algorithm using valuable infeasible solutions"
    """

    def __init__(self, n_dim=15):
        self.problem_name = 'MW10'
        self.dim = n_dim
        self.n_objectives = 2
        self.n_constraints = 3
        super().__init__(dim=n_dim, n_objectives=self.n_objectives)

        # Optimum Pareto Front
        self.f_opt = np.genfromtxt('./cases/MW_files/MW10_pf.txt', delimiter='')
        self.var_opt = None

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub, value=0.5 * np.ones(self.dim),
                           scale=1.0, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        prob.add_con_group('con', self.n_constraints, lower=None, upper=None) # Todo: check bounds

    def set_objectives(self, prob, **kwargs):
        for i in range(self.n_objectives):
            prob.add_obj(f"f_{i}")

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']
        obj, cons = self.obj_func_specific(x)
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        obj = np.zeros(self.n_objectives)
        cons = np.zeros(self.n_constraints)
        g = self.g2(x)
        obj[0] = g * np.power(x[0], self.dim)
        obj[1] = g * (1.0 - np.power(obj[0] / g, 2.0))

        cons[0] = -1.0 * (2.0 - 4.0 * obj[0] * obj[0] - obj[1]) * (2.0 - 8.0 * obj[0] * obj[0] - obj[1])
        cons[1] = (2.0 - 2.0 * obj[0] * obj[0] - obj[1]) * (2.0 - 16.0 * obj[0] * obj[0] - obj[1])
        cons[2] = (1.0 - obj[0] * obj[0] - obj[1]) * (1.2 - 1.2 * obj[0] * obj[0] - obj[1])
        return obj, cons

    def cons_func_specific(self, x):
        return None


class MW11(MWSetup):
    """
    MW11: "supp:A constrained multi-objective evolutionary algorithm using valuable infeasible solutions"
    """

    def __init__(self, n_dim=15):
        self.problem_name = 'MW11'
        self.dim = n_dim
        self.n_objectives = 2
        self.n_constraints = 4
        super().__init__(dim=n_dim, n_objectives=self.n_objectives)

        # Optimum Pareto Front
        self.f_opt = np.genfromtxt('./cases/MW_files/MW11_pf.txt', delimiter='')
        self.var_opt = None
        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.lb = np.zeros(self.dim)
        self.ub = np.sqrt(2) * np.ones(self.dim)

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub, value=0.5 * np.ones(self.dim),
                           scale=1.0, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        prob.add_con_group('con', self.n_constraints, lower=None, upper=None) # Todo: check bounds

    def set_objectives(self, prob, **kwargs):
        for i in range(self.n_objectives):
            prob.add_obj(f"f_{i}")

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']
        obj, cons = self.obj_func_specific(x)
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        obj = np.zeros(self.n_objectives)
        cons = np.zeros(self.n_constraints)
        g = self.g3(x)
        obj[0] = g * x[0]
        obj[1] = g * np.sqrt(2.0 - np.power(obj[0] / g, 2.0))

        cons[0] = -1.0 * (3.0 - obj[0] * obj[0] - obj[1]) * (3.0 - 2.0 * obj[0] * obj[0] - obj[1])
        cons[1] = (3.0 - 0.625 * obj[0] * obj[0] - obj[1]) * (3.0 - 7.0 * obj[0] * obj[0] - obj[1])
        cons[2] = -1.0 * (1.62 - 0.18 * obj[0] * obj[0] - obj[1]) * (1.125 - 0.125 * obj[0] * obj[0] - obj[1])
        cons[3] = (2.07 - 0.23 * obj[0] * obj[0] - obj[1]) * (0.63 - 0.07 * obj[0] * obj[0] - obj[1])
        return obj, cons

    def cons_func_specific(self, x):
        return None


class MW12(MWSetup):
    """
    MW12: "supp:A constrained multi-objective evolutionary algorithm using valuable infeasible solutions"
    """

    def __init__(self, n_dim=15):
        self.problem_name = 'MW12'
        self.dim = n_dim
        self.n_objectives = 2
        self.n_constraints = 2
        super().__init__(dim=n_dim, n_objectives=self.n_objectives)

        # Optimum Pareto Front
        self.f_opt = np.genfromtxt('./cases/MW_files/MW12_pf.txt', delimiter='')
        self.var_opt = None

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub, value=0.5 * np.ones(self.dim),
                           scale=1.0, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        prob.add_con_group('con', self.n_constraints, lower=None, upper=None)  # Todo: check bounds

    def set_objectives(self, prob, **kwargs):
        for i in range(self.n_objectives):
            prob.add_obj(f"f_{i}")

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']
        obj, cons = self.obj_func_specific(x)
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        obj = np.zeros(self.n_objectives)
        cons = np.zeros(self.n_constraints)
        g = self.g1(x)
        obj[0] = g * x[0]
        obj[1] = g * (0.85 - 0.8 * (obj[0] / g) - 0.08 * np.abs(np.sin(3.2 * np.pi * (obj[0] / g))))

        cons[0] = -1.0 * (1 - 0.625 * obj[0] - obj[1] + 0.08 * np.sin(2 * np.pi * (obj[1] - obj[0] / 1.6))) * (
                1.4 - 0.875 * obj[0] - obj[1] + 0.08 * np.sin(2 * np.pi * (obj[1] / 1.4 - obj[0] / 1.6)))
        cons[1] = (1 - 0.8 * obj[0] - obj[1] + 0.08 * np.sin(2 * np.pi * (obj[1] - obj[0] / 1.5))) * (
                1.8 - 1.125 * obj[0] - obj[1] + 0.08 * np.sin(2 * np.pi * (obj[1] / 1.8 - obj[0] / 1.6)))
        return obj, cons

    def cons_func_specific(self, x):
        return None


class MW13(MWSetup):
    """
    MW13: "supp:A constrained multi-objective evolutionary algorithm using valuable infeasible solutions"
    """

    def __init__(self, n_dim=15):
        self.problem_name = 'MW13'
        self.dim = n_dim
        self.n_objectives = 2
        self.n_constraints = 2
        super().__init__(dim=n_dim, n_objectives=self.n_objectives)

        # Optimum Pareto Front
        self.f_opt = np.genfromtxt('./cases/MW_files/MW13_pf.txt', delimiter='')
        self.var_opt = None

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.lb = np.zeros(self.dim)
        self.ub = 1.5 * np.ones(self.dim)

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub, value=0.5 * np.ones(self.dim),
                           scale=1.0, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        prob.add_con_group('con', self.n_constraints, lower=None, upper=None)  # Todo: check bounds

    def set_objectives(self, prob, **kwargs):
        for i in range(self.n_objectives):
            prob.add_obj(f"f_{i}")

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']
        obj, cons = self.obj_func_specific(x)
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        obj = np.zeros(self.n_objectives)
        cons = np.zeros(self.n_constraints)
        g = self.g2(x)
        obj[0] = g * x[0]
        obj[1] = g * (5.0 - np.exp(obj[0] / g) - np.abs(0.5 * np.sin(3 * np.pi * obj[0] / g)))

        cons[0] = -1.0 * (5.0 - (1 + obj[0] + 0.5 * obj[0] * obj[0]) - 0.5 * np.sin(3 * np.pi * obj[0]) - obj[1]) * (
                5.0 - (1 + 0.7 * obj[0]) - 0.5 * np.sin(3 * np.pi * obj[0]) - obj[1])
        cons[1] = (5.0 - np.exp(obj[0]) - 0.5 * np.sin(3 * np.pi * obj[0]) - obj[1]) * (
                5.0 - (1 + 0.4 * obj[0]) - 0.5 * np.sin(3 * np.pi * obj[0]) - obj[1])
        return obj, cons

    def cons_func_specific(self, x):
        return None


class MW14(MWSetup):
    """
    MW14: "supp:A constrained multi-objective evolutionary algorithm using valuable infeasible solutions"
    """

    def __init__(self, n_dim=15):
        self.problem_name = 'MW14'
        self.dim = n_dim
        self.n_objectives = 3
        self.n_constraints = 1
        super().__init__(dim=n_dim, n_objectives=self.n_objectives)

        # Optimum Pareto Front
        self.f_opt = np.genfromtxt('./cases/MW_files/MW14_pf.txt', delimiter='')
        self.var_opt = None

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.lb = np.zeros(self.dim)
        self.ub = 1.5 * np.ones(self.dim)

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub, value=0.5 * np.ones(self.dim),
                           scale=1.0, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        prob.add_con_group('con', self.n_constraints, lower=None, upper=None)  # Todo: check bounds

    def set_objectives(self, prob, **kwargs):
        for i in range(self.n_objectives):
            prob.add_obj(f"f_{i}")

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']
        obj, cons = self.obj_func_specific(x)
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        cons = np.zeros(self.n_constraints)
        g = self.g3(x)
        obj = np.zeros(self.n_objectives)
        obj[:-1] = x[:(self.n_objectives - 1)]
        LA1 = self.LA1(1.5, 1.1, 2.0, 1.0, obj[:-1])
        inter = (6 - np.exp(obj[:-1]) - LA1).sum()
        obj[-1] = g / (self.n_objectives - 1) * inter

        alpha = 6.1 - 1 - obj[:-1] - 0.5 * obj[:-1] * obj[:-1] - LA1
        cons[0] = obj[-1] - 1 / (self.n_objectives - 1) * alpha.sum()
        return obj, cons

    def cons_func_specific(self, x):
        return None