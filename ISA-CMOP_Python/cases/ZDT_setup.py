import numpy as np

from optimisation.setup import Setup


class ZDT1(Setup):
    """
    ZDT1: Convex Front
    https://dl.acm.org/doi/pdf/10.1162/106365600568202
    """

    def __init__(self, n_dim=30):
        super().__init__()
        self.problem_name = 'ZDT1'
        self.dim = n_dim
        self.n_objectives = 2
        self.n_constraints = 0

        # Optimum Pareto Front
        pf1 = np.linspace(0.0, 1.0, 100)  # 100 Points in PF
        pf2 = 1.0 - pf1 ** 0.5
        self.f_opt = np.hstack((np.reshape(pf1, (len(pf1), 1)), np.reshape(pf2, (len(pf2), 1))))
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
        prob.add_obj('f_1')
        prob.add_obj('f_2')

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']
        obj = self.obj_func_specific(x)
        cons = None
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        obj = np.zeros(self.n_objectives)
        obj[0] = x[0]
        c = np.sum(x[1:], axis=0)
        g = 1.0 + (9.0 / (len(x) - 1.0)) * c
        h = 1.0 - (obj[0] / g) ** 0.5
        obj[1] = g * h

        return obj

    def cons_func_specific(self, x):
        return None


class ZDT2(Setup):
    """
    ZDT2: Concave Front
    https://dl.acm.org/doi/pdf/10.1162/106365600568202
    """

    def __init__(self, n_dim=30):
        super().__init__()
        self.problem_name = 'ZDT2'
        self.dim = n_dim
        self.n_objectives = 2
        self.n_constraints = 0

        # Optimum Pareto Front
        pf1 = np.linspace(0.0, 1.0, 100)  # 100 Points in PF
        pf2 = 1.0 - pf1 ** 2
        self.f_opt = np.hstack((np.reshape(pf1, (len(pf1), 1)), np.reshape(pf2, (len(pf2), 1))))
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
        prob.add_obj('f_1')
        prob.add_obj('f_2')

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']
        obj = self.obj_func_specific(x)
        cons = None
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        obj = np.zeros(self.n_objectives)
        obj[0] = x[0]
        c = np.sum(x[1:], axis=0)
        g = 1.0 + (9.0 / (self.dim - 1.0)) * c
        h = 1.0 - (obj[0] / g) ** 2.0
        obj[1] = g * h

        return obj

    def cons_func_specific(self, x):
        return None


class ZDT3(Setup):
    """
    ZDT3: Discontinous Front
    https://dl.acm.org/doi/pdf/10.1162/106365600568202
    """

    def __init__(self, n_dim=30):
        super().__init__()
        self.problem_name = 'ZDT3'
        self.dim = n_dim
        self.n_objectives = 2
        self.n_constraints = 0

        # Optimum Pareto Front
        self.f_opt = None
        regions = [[0, 0.0830015349],
                   [0.182228780, 0.2577623634],
                   [0.4093136748, 0.4538821041],
                   [0.6183967944, 0.6525117038],
                   [0.8233317983, 0.8518328654]]
        for seg in regions:
            pf1 = np.linspace(seg[0], seg[1], int(100 / len(regions)))  # 100 Points in PF
            pf2 = 1.0 - pf1 ** 0.5 - pf1 * np.sin(10 * np.pi * pf1)
            if self.f_opt is None:
                self.f_opt = np.hstack((np.reshape(pf1, (len(pf1), 1)), np.reshape(pf2, (len(pf2), 1))))
            else:
                self.f_opt = np.vstack(
                    (self.f_opt, np.hstack((np.reshape(pf1, (len(pf1), 1)), np.reshape(pf2, (len(pf2), 1))))))
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
        prob.add_obj('f_1')
        prob.add_obj('f_2')

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']
        obj = self.obj_func_specific(x)
        cons = None
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        obj = np.zeros(self.n_objectives)
        obj[0] = x[0]
        c = np.sum(x[1:])
        g = 1.0 + (9.0 / (self.dim - 1.0)) * c
        h = 1.0 - (obj[0] / g)**0.5 - (obj[0] / g) * np.sin(10 * np.pi * obj[0])
        obj[1] = g * h

        return obj

    def cons_func_specific(self, x):
        return None


class ZDT4(Setup):
    """
    ZDT4: 21^9 Local Fronts
    https://dl.acm.org/doi/pdf/10.1162/106365600568202
    """

    def __init__(self, n_dim=10):
        super().__init__()
        self.problem_name = 'ZDT4'
        self.dim = n_dim
        self.n_objectives = 2
        self.n_constraints = 0

        # Optimum Pareto Front
        pf1 = np.linspace(0, 1, 100)  # 100 Points in PF
        pf2 = 1.0 - pf1 ** 0.5
        self.f_opt = np.hstack((np.reshape(pf1, (len(pf1), 1)), np.reshape(pf2, (len(pf2), 1))))
        self.var_opt = None

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)
        for i in range(1, self.dim):
            self.lb[i] = -5.0
            self.ub[i] = 5.0

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub, value=0.5 * np.ones(self.dim),
                           scale=1.0, f_opt=self.f_opt)

    def set_constraints(self, prob, **kwargs):
        pass

    def set_objectives(self, prob, **kwargs):
        prob.add_obj('f_1')
        prob.add_obj('f_2')

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']
        obj = self.obj_func_specific(x)
        cons = None
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        obj = np.zeros(self.n_objectives)
        obj[0] = np.round(x[0], 10)
        g = 1.0 + 10 * (self.dim - 1) + np.sum(x[1:]**2 - 10 * np.cos(4 * np.pi * x[1:]))
        h = 1.0 - np.sqrt(obj[0] / g)
        obj[1] = g * h

        return obj

    def cons_func_specific(self, x):
        return None


class ZDT6(Setup):
    """
    ZDT6: Nonuniform Front
    https://dl.acm.org/doi/pdf/10.1162/106365600568202
    """

    def __init__(self, n_dim=10):
        super().__init__()
        self.problem_name = 'ZDT6'
        self.dim = n_dim
        self.n_objectives = 2
        self.n_constraints = 0

        # Optimum Pareto Front
        pf1 = np.linspace(0.280775, 1.0, 100)  # 100 Points in PF
        pf2 = 1.0 - pf1 ** 2
        self.f_opt = np.hstack((np.reshape(pf1, (len(pf1), 1)), np.reshape(pf2, (len(pf2), 1))))
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
        prob.add_obj('f_1')
        prob.add_obj('f_2')

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']
        obj = self.obj_func_specific(x)
        cons = None
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        obj = np.zeros(self.n_objectives)
        obj[0] = 1 - np.exp(-4 * x[0]) * np.sin(6 * np.pi * x[0]) ** 6
        c = np.sum(x[1:], axis=0)
        g = 1.0 + 9 * (np.sum(x[1:]) / (self.dim - 1)) ** 0.25
        h = 1.0 - (obj[0] / g) ** 2
        obj[1] = g * h

        return obj

    def cons_func_specific(self, x):
        return None