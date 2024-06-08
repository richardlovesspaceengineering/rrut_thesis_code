import numpy as np
from optimisation.setup import Setup

"""
Ported the DASCMOP implementation across from Pymoo : https://www.egr.msu.edu/~kdeb/papers/c2019006.pdf
"""

DIFFICULTIES = [
    (0.25, 0., 0.), (0., 0.25, 0.), (0., 0., 0.25), (0.25, 0.25, 0.25),
    (0.5, 0., 0.), (0., 0.5, 0.), (0., 0., 0.5), (0.5, 0.5, 0.5),
    (0.75, 0., 0.), (0., 0.75, 0.), (0., 0., 0.75), (0.75, 0.75, 0.75),
    (0., 1.0, 0.), (0.5, 1.0, 0.), (0., 1.0, 0.5), (0.5, 1.0, 0.5)
]


class DASCMOPSetup(Setup):
    """
    Wrapper class for standarised DASCMOP methods
    """
    def __init__(self, dim, n_objectives, difficulty):
        super().__init__()

        self.dim = dim
        self.n_objectives = n_objectives

        # Validate difficulty input
        if isinstance(difficulty, int):
            self.difficulty = difficulty
            if not (1 <= difficulty <= len(DIFFICULTIES)):
                raise Exception(f"Difficulty must be 1 <= difficulty <= {len(DIFFICULTIES)}, but it is {difficulty}!")
            vals = DIFFICULTIES[difficulty - 1]
        else:
            self.difficulty = -1
            vals = difficulty

        self.eta, self.zeta, self.gamma = vals

    def g1(self, x):
        contrib = (x[self.n_objectives - 1:] - np.sin(0.5 * np.pi * x[0:1])) ** 2
        return contrib.sum()

    def g2(self, x):
        z = x[self.n_objectives - 1:] - 0.5
        contrib = z ** 2 - np.cos(20 * np.pi * z)
        return (self.dim - self.n_objectives + 1) + contrib.sum()

    def g3(self, x):
        j = np.arange(self.n_objectives - 1, self.dim) + 1
        contrib = (x[self.n_objectives - 1:] - np.cos(0.25 * j / self.dim * np.pi * (x[0:1] + x[1:2]))) ** 2
        return contrib.sum()


class DASCMOP1(DASCMOPSetup):
    """
    DASCMOP1: "Difficulty Adjustable and Scalable Constrained Multi-objective Test Problem Toolkit"
    """

    def __init__(self, n_dim=30, difficulty=1):
        self.problem_name = 'DASCMOP1'
        self.difficulty = difficulty
        self.dim = n_dim
        self.n_objectives = 2
        self.n_constraints = 11
        super().__init__(dim=n_dim, n_objectives=self.n_objectives, difficulty=self.difficulty)

        # Optimum Pareto Front
        self.f_opt = np.genfromtxt(f"./cases/DASCMOP_files/{self.problem_name.lower()}_{self.difficulty}.pf", delimiter='')
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
        g = self.g1(x)

        obj[0] = x[0:1] + g
        obj[1] = 1.0 - x[0:1] ** 2 + g

        cons = self.cons_func_specific(x, obj, g)
        return obj, cons

    def cons_func_specific(self, x, obj, g):
        a = 20.
        b = 2. * self.eta - 1.
        d = 0.5 if self.zeta != 0 else 0.
        if self.zeta > 0:
            e = d - np.log(self.zeta)
        else:
            e = 1e30
        r = 0.5 * self.gamma

        p_k = np.array([[0., 1.0, 0., 1.0, 2.0, 0., 1.0, 2.0, 3.0]])
        q_k = np.array([[1.5, 0.5, 2.5, 1.5, 0.5, 3.5, 2.5, 1.5, 0.5]])

        a_k2 = 0.3
        b_k2 = 1.2
        theta_k = -0.25 * np.pi

        c = np.zeros(2 + p_k.shape[1])

        c[0] = np.sin(a * np.pi * x[0]) - b
        if self.zeta == 1.:
            c[1:2] = 1e-4 - np.abs(e - g)
        else:
            c[1:2] = (e - g) * (g - d)

        c[2:] = (((obj[0] - p_k) * np.cos(theta_k) - (obj[1] - q_k) * np.sin(theta_k)) ** 2 / a_k2
                    + ((obj[0] - p_k) * np.sin(theta_k) + (obj[1] - q_k) * np.cos(theta_k)) ** 2 / b_k2
                    - r)

        return -1 * c


class DASCMOP2(DASCMOP1):
    """
    DASCMOP2: "Difficulty Adjustable and Scalable Constrained Multi-objective Test Problem Toolkit"
    """

    def __init__(self, n_dim=30, difficulty=1):
        super().__init__(n_dim=n_dim, difficulty=difficulty)
        self.problem_name = 'DASCMOP2'

        # Optimum Pareto Front
        self.f_opt = np.genfromtxt(f"./cases/DASCMOP_files/{self.problem_name.lower()}_{self.difficulty}.pf",
                                   delimiter='')

    def obj_func_specific(self, x):
        obj = np.zeros(self.n_objectives)
        g = self.g1(x)

        obj[0] = x[0:1] + g
        obj[1] = 1.0 - np.sqrt(x[0:1]) + g

        cons = self.cons_func_specific(x, obj, g)
        return obj, cons


class DASCMOP3(DASCMOP1):
    """
    DASCMOP3: "Difficulty Adjustable and Scalable Constrained Multi-objective Test Problem Toolkit"
    """

    def __init__(self, n_dim=30, difficulty=1):
        super().__init__(n_dim=n_dim, difficulty=difficulty)
        self.problem_name = 'DASCMOP3'

        # Optimum Pareto Front
        self.f_opt = np.genfromtxt(f"./cases/DASCMOP_files/{self.problem_name.lower()}_{self.difficulty}.pf",
                                   delimiter='')

    def obj_func_specific(self, x):
        obj = np.zeros(self.n_objectives)
        g = self.g1(x)

        obj[0] = x[0:1] + g
        obj[1] = 1.0 - np.sqrt(x[0:1]) + 0.5 * np.abs(np.sin(5 * np.pi * x[0:1])) + g

        cons = self.cons_func_specific(x, obj, g)
        return obj, cons


class DASCMOP4(DASCMOP1):
    """
    DASCMOP4: "Difficulty Adjustable and Scalable Constrained Multi-objective Test Problem Toolkit"
    """

    def __init__(self, n_dim=30, difficulty=1):
        super().__init__(n_dim=n_dim, difficulty=difficulty)
        self.problem_name = 'DASCMOP4'

        # Optimum Pareto Front
        self.f_opt = np.genfromtxt(f"./cases/DASCMOP_files/{self.problem_name.lower()}_{self.difficulty}.pf",
                                   delimiter='')

    def obj_func_specific(self, x):
        obj = np.zeros(self.n_objectives)
        g = self.g2(x)

        obj[0] = x[0:1] + g
        obj[1] = 1.0 - x[0:1] ** 2 + g

        cons = self.cons_func_specific(x, obj, g)
        return obj, cons


class DASCMOP5(DASCMOP1):
    """
    DASCMOP5: "Difficulty Adjustable and Scalable Constrained Multi-objective Test Problem Toolkit"
    """

    def __init__(self, n_dim=30, difficulty=1):
        super().__init__(n_dim=n_dim, difficulty=difficulty)
        self.problem_name = 'DASCMOP5'

        # Optimum Pareto Front
        self.f_opt = np.genfromtxt(f"./cases/DASCMOP_files/{self.problem_name.lower()}_{self.difficulty}.pf",
                                   delimiter='')

    def obj_func_specific(self, x):
        obj = np.zeros(self.n_objectives)
        g = self.g2(x)

        obj[0] = x[0:1] + g
        obj[1] = 1.0 - np.sqrt(x[0:1]) + g

        cons = self.cons_func_specific(x, obj, g)
        return obj, cons


class DASCMOP6(DASCMOP1):
    """
    DASCMOP6: "Difficulty Adjustable and Scalable Constrained Multi-objective Test Problem Toolkit"
    """

    def __init__(self, n_dim=30, difficulty=1):
        super().__init__(n_dim=n_dim, difficulty=difficulty)
        self.problem_name = 'DASCMOP6'

        # Optimum Pareto Front
        self.f_opt = np.genfromtxt(f"./cases/DASCMOP_files/{self.problem_name.lower()}_{self.difficulty}.pf",
                                   delimiter='')

    def obj_func_specific(self, x):
        obj = np.zeros(self.n_objectives)
        g = self.g2(x)

        obj[0] = x[0:1] + g
        obj[1] = 1.0 - np.sqrt(x[0:1]) + 0.5 * np.abs(np.sin(5 * np.pi * x[0:1])) + g

        cons = self.cons_func_specific(x, obj, g)
        return obj, cons


class DASCMOP7(DASCMOPSetup):
    """
    DASCMOP7: "Difficulty Adjustable and Scalable Constrained Multi-objective Test Problem Toolkit"
    """

    def __init__(self, n_dim=30, difficulty=1):
        self.problem_name = 'DASCMOP7'
        self.difficulty = difficulty
        self.dim = n_dim
        self.n_objectives = 3
        self.n_constraints = 7
        super().__init__(dim=n_dim, n_objectives=self.n_objectives, difficulty=self.difficulty)

        # Optimum Pareto Front
        self.f_opt = np.genfromtxt(f"./cases/DASCMOP_files/{self.problem_name.lower()}_{self.difficulty}.pf", delimiter='')
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
        g = self.g2(x)

        obj[0] = x[0:1] * x[1:2] + g
        obj[1] = x[1:2] * (1.0 - x[0:1]) + g
        obj[2] = 1 - x[1:2] + g

        cons = self.cons_func_specific(x, obj, g)
        return obj, cons

    def cons_func_specific(self, x, obj, g):
        a = 20.
        b = 2. * self.eta - 1.
        d = 0.5 if self.zeta != 0 else 0
        if self.zeta > 0:
            e = d - np.log(self.zeta)
        else:
            e = 1e30
        r = 0.5 * self.gamma

        x_k = np.array([[1.0, 0., 0., 1.0 / np.sqrt(3.0)]])
        y_k = np.array([[0., 1.0, 0., 1.0 / np.sqrt(3.0)]])
        z_k = np.array([[0., 0., 1.0, 1.0 / np.sqrt(3.0)]])

        c = np.zeros(3 + x_k.shape[1])

        c[0] = np.sin(a * np.pi * x[0]) - b
        c[1] = np.cos(a * np.pi * x[1]) - b
        if self.zeta == 1:
            c[2:3] = 1e-4 - np.abs(e - g)
        else:
            c[2:3] = (e - g) * (g - d)

        c[3:] = (obj[0] - x_k) ** 2 + (obj[1] - y_k) ** 2 + (obj[2] - z_k) ** 2 - r ** 2
        return -1 * c


class DASCMOP8(DASCMOP7):
    """
    DASCMOP8: "Difficulty Adjustable and Scalable Constrained Multi-objective Test Problem Toolkit"
    """

    def __init__(self, n_dim=30, difficulty=1):
        super().__init__(n_dim=n_dim, difficulty=difficulty)
        self.problem_name = 'DASCMOP8'

        # Optimum Pareto Front
        self.f_opt = np.genfromtxt(f"./cases/DASCMOP_files/{self.problem_name.lower()}_{self.difficulty}.pf",
                                   delimiter='')

    def obj_func_specific(self, x):
        obj = np.zeros(self.n_objectives)
        g = self.g2(x)

        obj[0] = np.cos(0.5 * np.pi * x[0:1]) * np.cos(0.5 * np.pi * x[1:2]) + g
        obj[1] = np.cos(0.5 * np.pi * x[0:1]) * np.sin(0.5 * np.pi * x[1:2]) + g
        obj[2] = np.sin(0.5 * np.pi * x[0:1]) + g

        cons = self.cons_func_specific(x, obj, g)
        return obj, cons


class DASCMOP9(DASCMOP7):
    """
    DASCMOP9: "Difficulty Adjustable and Scalable Constrained Multi-objective Test Problem Toolkit"
    """

    def __init__(self, n_dim=30, difficulty=1):
        super().__init__(n_dim=n_dim, difficulty=difficulty)
        self.problem_name = 'DASCMOP9'

        # Optimum Pareto Front
        self.f_opt = np.genfromtxt(f"./cases/DASCMOP_files/{self.problem_name.lower()}_{self.difficulty}.pf",
                                   delimiter='')

    def obj_func_specific(self, x):
        obj = np.zeros(self.n_objectives)
        g = self.g3(x)

        obj[0] = np.cos(0.5 * np.pi * x[0:1]) * np.cos(0.5 * np.pi * x[1:2]) + g
        obj[1] = np.cos(0.5 * np.pi * x[0:1]) * np.sin(0.5 * np.pi * x[1:2]) + g
        obj[2] = np.sin(0.5 * np.pi * x[0:1]) + g

        cons = self.cons_func_specific(x, obj, g)
        return obj, cons