import numpy as np
from optimisation.setup import Setup

"""
Ported the LIRCMOP implementation across from jMetalPy : 
"An Improved Epsilon Constraint-handling Method in MOEA/D for CMOPs with Large Infeasible Regions"
"""


class LIRCMOPSetup(Setup):
    """
    Wrapper class for standarised LIRCMOP methods
    """

    def __init__(self, dim, n_objectives):
        super().__init__()

        self.dim = dim
        self.n_objectives = n_objectives

    # LIRCMOP13-14
    @staticmethod
    def g(obj):
        return np.sum(obj**2)

    # LIRCMOP1-4
    @staticmethod
    def g1(x):
        return np.sum((x[2::2] - np.sin(0.5 * np.pi * x[0])) ** 2)

    # LIRCMOP1-4
    @staticmethod
    def g2(x):
        return np.sum((x[1::2] - np.cos(0.5 * np.pi * x[0])) ** 2)

    # LIRCMOP5-12
    def g3(self, x):
        const = 0.5 * np.pi * np.arange(3, self.dim, 2) / self.dim
        g = np.sum((x[2::2] - np.sin(const * x[0])) ** 2)
        return g

    # LIRCMOP5-12
    def g4(self, x):
        const = 0.5 * np.pi * np.arange(2, self.dim + 1, 2) / self.dim
        g = np.sum((x[1::2] - np.cos(const * x[0])) ** 2)
        return g

    # LIRCMOP13-14
    def g5(self, x):
        return np.sum((x[2::2] - 0.5) ** 2)

    def evaluate(self, var):
        obj = self.obj_func_specific(var)
        cons = self.cons_func_specific(var)
        return obj, cons

    def exact_pareto(self, n_points):
        pass

    def _calc_pareto_front(self):
        return super()._calc_pareto_front(
            f"./cases/LIRCMOP_files/{self.problem_name}.pf"
        )


class LIRCMOP1(LIRCMOPSetup):
    """
    LIRCMOP1: 2 obj, 2 cons
    """

    def __init__(self, n_dim=30, n_constraints=2):
        self.problem_name = "LIRCMOP1"
        self.dim = n_dim
        self.n_objectives = 2
        self.n_constraints = n_constraints
        super().__init__(dim=n_dim, n_objectives=self.n_objectives)

        self.f_opt = None
        self.var_opt = None

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)

    def set_variables(self, prob, **kwargs):
        prob.add_var_group(
            "x_vars",
            self.dim,
            "c",
            lower=self.lb,
            upper=self.ub,
            value=0.5 * np.ones(self.dim),
            scale=1.0,
            f_opt=self.f_opt,
        )

    def set_constraints(self, prob, **kwargs):
        prob.add_con_group(
            "con", self.n_constraints, lower=None, upper=None
        )  # Todo: check bounds

    def set_objectives(self, prob, **kwargs):
        for i in range(self.n_objectives):
            prob.add_obj(f"f_{i}")

    def obj_func(self, x_dict, **kwargs):
        x = x_dict["x_vars"]
        obj = self.obj_func_specific(x)
        cons = self.cons_func_specific(x)
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        obj = np.zeros(self.n_objectives)

        obj[0] = x[0] + self.g1(x)
        obj[1] = 1 - x[0] ** 2 + self.g2(x)

        return obj

    def cons_func_specific(self, x, a=0.51, b=0.5):
        cons = np.zeros(self.n_constraints)

        cons[0] = (a - self.g1(x)) * (self.g1(x) - b)
        cons[1] = (a - self.g2(x)) * (self.g2(x) - b)

        return -cons

    def exact_pareto(self, n_points):
        pf = np.zeros((n_points, 2))
        pf[:, 0] = np.linspace(0, 1, n_points)
        pf[:, 1] = 1 - pf[:, 0] ** 2
        pf += 0.5

        return pf


class LIRCMOP2(LIRCMOP1):
    """
    LIRCMOP2: 2 obj, 2 cons
    """

    def __init__(self, n_dim=30, n_constraints=2):
        super().__init__(n_dim=n_dim, n_constraints=n_constraints)
        self.problem_name = "LIRCMOP2"

        # Optimum Pareto Front
        self.f_opt = None

    def obj_func_specific(self, x):
        obj = np.zeros(self.n_objectives)

        obj[0] = x[0] + self.g1(x)
        obj[1] = 1 - np.sqrt(x[0]) + self.g2(x)

        return obj

    def exact_pareto(self, n_points):
        pf = np.zeros((n_points, 2))
        pf[:, 0] = np.linspace(0, 1, n_points)
        pf[:, 1] = 1 - np.sqrt(pf[:, 0])
        pf += 0.5

        return pf


class LIRCMOP3(LIRCMOP1):
    """
    LIRCMOP3: 2 obj, 3 cons
    """

    def __init__(self, n_dim=30):
        super().__init__(n_dim=n_dim, n_constraints=3)
        self.problem_name = "LIRCMOP3"
        self.f_opt = None

    def cons_func_specific(self, x, a=0.51, b=0.5, c=20.0):
        cons = np.zeros(self.n_constraints)

        cons[0] = (a - self.g1(x)) * (self.g1(x) - b)
        cons[1] = (a - self.g2(x)) * (self.g2(x) - b)
        cons[2] = np.sin(c * np.pi * x[0]) - 0.5

        return -cons


class LIRCMOP4(LIRCMOP2):
    """
    LIRCMOP4: 2 obj, 3 cons
    """

    def __init__(self, n_dim=30):
        super().__init__(n_dim=n_dim, n_constraints=3)
        self.problem_name = "LIRCMOP4"
        self.f_opt = None

    def cons_func_specific(self, x, a=0.51, b=0.5, c=20.0):
        cons = np.zeros(self.n_constraints)

        cons[0] = (a - self.g1(x)) * (self.g1(x) - b)
        cons[1] = (a - self.g2(x)) * (self.g2(x) - b)
        cons[2] = np.sin(c * np.pi * x[0]) - 0.5

        return -cons


class LIRCMOP5(LIRCMOPSetup):
    """
    LIRCMOP5: 2 obj, 2 cons
    """

    def __init__(self, n_dim=30, n_constraints=2):
        self.problem_name = "LIRCMOP5"
        self.dim = n_dim
        self.n_objectives = 2
        self.n_constraints = n_constraints
        super().__init__(dim=n_dim, n_objectives=self.n_objectives)

        self.f_opt = None
        self.var_opt = None

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)

    def set_variables(self, prob, **kwargs):
        prob.add_var_group(
            "x_vars",
            self.dim,
            "c",
            lower=self.lb,
            upper=self.ub,
            value=0.5 * np.ones(self.dim),
            scale=1.0,
            f_opt=self.f_opt,
        )

    def set_constraints(self, prob, **kwargs):
        prob.add_con_group(
            "con", self.n_constraints, lower=None, upper=None
        )  # Todo: check bounds

    def set_objectives(self, prob, **kwargs):
        for i in range(self.n_objectives):
            prob.add_obj(f"f_{i}")

    def obj_func(self, x_dict, **kwargs):
        x = x_dict["x_vars"]
        obj, cons = self.obj_func_specific(x)
        # cons = self.cons_func_specific(x)
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        obj = np.zeros(self.n_objectives)

        obj[0] = x[0] + 10 * self.g3(x) + 0.7057
        obj[1] = 1 - np.sqrt(x[0]) + 10 * self.g4(x) + 0.7057

        return obj

    def cons_func_specific(self, obj, r=0.1, theta=-0.25 * np.pi):
        a_array = [2.0, 2.0]
        b_array = [4.0, 8.0]
        x_offset = [1.6, 2.5]
        y_offset = [1.6, 2.5]

        f1 = obj[0]
        f2 = obj[1]

        cons = (
            (
                ((f1 - x_offset) * np.cos(theta) - (f2 - y_offset) * np.sin(theta))
                / a_array
            )
            ** 2
            + (
                ((f1 - x_offset) * np.sin(theta) + (f2 - y_offset) * np.cos(theta))
                / b_array
            )
            ** 2
            - r
        )

        return -cons

    def exact_pareto(self, n_points):
        pf = np.zeros((n_points, 2))
        pf[:, 0] = np.linspace(0, 1, n_points)
        pf[:, 1] = 1 - pf[:, 0] ** 2
        pf += 0.5

        return pf


class LIRCMOP6(LIRCMOP5):
    """
    LIRCMOP6: 2 obj, 2 cons
    """

    def __init__(self, n_dim=30, n_constraints=2):
        super().__init__(n_dim=n_dim, n_constraints=n_constraints)
        self.problem_name = "LIRCMOP6"
        self.f_opt = None

    def obj_func_specific(self, x):
        obj = np.zeros(self.n_objectives)

        obj[0] = x[0] + 10 * self.g3(x) + 0.7057
        obj[1] = 1 - x[0] ** 2 + 10 * self.g4(x) + 0.7057

        return obj

    def cons_func_specific(self, obj, r=0.1, theta=-0.25 * np.pi):
        a_array = [2.0, 2.0]
        b_array = [8.0, 8.0]
        x_offset = [1.8, 2.8]
        y_offset = [1.8, 2.8]

        f1 = obj[0]
        f2 = obj[1]

        cons = (
            (
                ((f1 - x_offset) * np.cos(theta) - (f2 - y_offset) * np.sin(theta))
                / a_array
            )
            ** 2
            + (
                ((f1 - x_offset) * np.sin(theta) + (f2 - y_offset) * np.cos(theta))
                / b_array
            )
            ** 2
            - r
        )

        return -cons


class LIRCMOP7(LIRCMOP5):
    """
    LIRCMOP7: 2 obj, 3 cons
    """

    def __init__(self, n_dim=30):
        super().__init__(n_dim=n_dim, n_constraints=3)
        self.problem_name = "LIRCMOP7"
        self.f_opt = None

    def cons_func_specific(self, obj, r=0.1, theta=-0.25 * np.pi):
        a_array = [2.0, 2.5, 2.5]
        b_array = [6.0, 12.0, 10.0]
        x_offset = [1.2, 2.25, 3.5]
        y_offset = [1.2, 2.25, 3.5]

        f1 = obj[0]
        f2 = obj[1]

        cons = (
            (
                ((f1 - x_offset) * np.cos(theta) - (f2 - y_offset) * np.sin(theta))
                / a_array
            )
            ** 2
            + (
                ((f1 - x_offset) * np.sin(theta) + (f2 - y_offset) * np.cos(theta))
                / b_array
            )
            ** 2
            - r
        )

        return -cons


class LIRCMOP8(LIRCMOP6):
    """
    LIRCMOP8: 2 obj, 3 cons
    """

    def __init__(self, n_dim=30, n_constraints=3):
        super().__init__(n_dim=n_dim, n_constraints=n_constraints)
        self.problem_name = "LIRCMOP8"
        self.f_opt = None

    def cons_func_specific(self, obj, r=0.1, theta=-0.25 * np.pi):
        a_array = [2.0, 2.5, 2.5]
        b_array = [6.0, 12.0, 10.0]
        x_offset = [1.2, 2.25, 3.5]
        y_offset = [1.2, 2.25, 3.5]

        f1 = obj[0]
        f2 = obj[1]

        cons = (
            (
                ((f1 - x_offset) * np.cos(theta) - (f2 - y_offset) * np.sin(theta))
                / a_array
            )
            ** 2
            + (
                ((f1 - x_offset) * np.sin(theta) + (f2 - y_offset) * np.cos(theta))
                / b_array
            )
            ** 2
            - r
        )

        return -cons


class LIRCMOP9(LIRCMOP8):
    """
    LIRCMOP9: 2 obj, 2 cons
    """

    def __init__(self, n_dim=30, n_constraints=2):
        super().__init__(n_dim=n_dim, n_constraints=n_constraints)
        self.problem_name = "LIRCMOP9"
        self.f_opt = None

    def obj_func_specific(self, x):
        obj = np.zeros(self.n_objectives)

        obj[0] = 1.7057 * x[0] * (10 * self.g3(x) + 1)
        obj[1] = 1.7957 * (1 - x[0] ** 2) * (10 * self.g4(x) + 1)

        return obj

    def cons_func_specific(self, obj, r=0.1, theta=-0.25 * np.pi, n=4.0):
        cons = np.zeros(self.n_constraints)

        x_offset = 1.40
        y_offset = 1.40
        a = 1.5
        b = 6.0
        alpha = -theta

        f1 = obj[0]
        f2 = obj[1]

        cons[0] = (
            f1 * np.sin(alpha)
            + f2 * np.cos(alpha)
            - np.sin(n * np.pi * (f1 * np.cos(alpha) - f2 * np.sin(alpha)))
            - 2
        )
        cons[1] = (
            (((f1 - x_offset) * np.cos(theta) - (f2 - y_offset) * np.sin(theta)) / a)
            ** 2
            + (((f1 - x_offset) * np.sin(theta) + (f2 - y_offset) * np.cos(theta)) / b)
            ** 2
            - r
        )

        return -cons


class LIRCMOP10(LIRCMOP9):
    """
    LIRCMOP10: 2 obj, 2 cons
    """

    def __init__(self, n_dim=30, n_constraints=2):
        super().__init__(n_dim=n_dim, n_constraints=n_constraints)
        self.problem_name = "LIRCMOP10"
        self.f_opt = None

    def obj_func(self, x_dict, **kwargs):
        x = x_dict["x_vars"]
        obj, cons = self.obj_func_specific(x)
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        obj = np.zeros(self.n_objectives)

        obj[0] = 1.7057 * x[0] * (10 * self.g3(x) + 1)
        obj[1] = 1.7957 * (1 - np.sqrt(x[0])) * (10 * self.g4(x) + 1)

        return obj

    def cons_func_specific(self, obj, r=0.1, theta=-0.25 * np.pi, n=4.0):
        cons = np.zeros(self.n_constraints)

        x_offset = 1.1
        y_offset = 1.2
        a = 2.0
        b = 4.0
        alpha = -theta

        f1 = obj[0]
        f2 = obj[1]

        cons[0] = (
            f1 * np.sin(alpha)
            + f2 * np.cos(alpha)
            - np.sin(n * np.pi * (f1 * np.cos(alpha) - f2 * np.sin(alpha)))
            - 2
        )
        cons[1] = (
            (((f1 - x_offset) * np.cos(theta) - (f2 - y_offset) * np.sin(theta)) / a)
            ** 2
            + (((f1 - x_offset) * np.sin(theta) + (f2 - y_offset) * np.cos(theta)) / b)
            ** 2
            - r
        )

        return -cons


class LIRCMOP11(LIRCMOP10):
    """
    LIRCMOP11: 2 obj, 2 cons
    """

    def __init__(self, n_dim=30, n_constraints=2):
        super().__init__(n_dim=n_dim, n_constraints=n_constraints)
        self.problem_name = "LIRCMOP11"
        self.f_opt = None

    def cons_func_specific(self, obj, r=0.1, theta=-0.25 * np.pi, n=4.0):
        cons = np.zeros(self.n_constraints)

        x_offset = 1.2
        y_offset = 1.2
        a = 1.5
        b = 5.0
        alpha = -theta

        f1 = obj[0]
        f2 = obj[1]

        cons[0] = (
            f1 * np.sin(alpha)
            + f2 * np.cos(alpha)
            - np.sin(n * np.pi * (f1 * np.cos(alpha) - f2 * np.sin(alpha)))
            - 2
        )
        cons[1] = (
            (((f1 - x_offset) * np.cos(theta) - (f2 - y_offset) * np.sin(theta)) / a)
            ** 2
            + (((f1 - x_offset) * np.sin(theta) + (f2 - y_offset) * np.cos(theta)) / b)
            ** 2
            - r
        )

        return -cons


class LIRCMOP12(LIRCMOP9):
    """
    LIRCMOP12: 2 obj, 2 cons
    """

    def __init__(self, n_dim=30, n_constraints=2):
        super().__init__(n_dim=n_dim, n_constraints=n_constraints)
        self.problem_name = "LIRCMOP12"
        self.f_opt = None

    def cons_func_specific(self, obj, r=0.1, theta=-0.25 * np.pi, n=4.0):
        cons = np.zeros(self.n_constraints)

        x_offset = 1.6
        y_offset = 1.6
        a = 1.5
        b = 6.0
        alpha = -theta

        f1 = obj[0]
        f2 = obj[1]

        cons[0] = (
            f1 * np.sin(alpha)
            + f2 * np.cos(alpha)
            - np.sin(n * np.pi * (f1 * np.cos(alpha) - f2 * np.sin(alpha)))
            - 2
        )
        cons[1] = (
            (((f1 - x_offset) * np.cos(theta) - (f2 - y_offset) * np.sin(theta)) / a)
            ** 2
            + (((f1 - x_offset) * np.sin(theta) + (f2 - y_offset) * np.cos(theta)) / b)
            ** 2
            - r
        )

        return -cons


class LIRCMOP13(LIRCMOPSetup):
    """
    LIRCMOP13: 3 obj, 2 cons
    """

    def __init__(self, n_dim=30, n_constraints=2):
        self.problem_name = "LIRCMOP13"
        self.dim = n_dim
        self.n_objectives = 3
        self.n_constraints = n_constraints
        super().__init__(dim=n_dim, n_objectives=self.n_objectives)

        self.f_opt = None
        self.var_opt = None

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)

    def set_variables(self, prob, **kwargs):
        prob.add_var_group(
            "x_vars",
            self.dim,
            "c",
            lower=self.lb,
            upper=self.ub,
            value=0.5 * np.ones(self.dim),
            scale=1.0,
            f_opt=self.f_opt,
        )

    def set_constraints(self, prob, **kwargs):
        prob.add_con_group(
            "con", self.n_constraints, lower=None, upper=None
        )  # Todo: check bounds

    def set_objectives(self, prob, **kwargs):
        for i in range(self.n_objectives):
            prob.add_obj(f"f_{i}")

    def obj_func(self, x_dict, **kwargs):
        x = x_dict["x_vars"]
        obj, cons = self.obj_func_specific(x)
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        obj = np.zeros(self.n_objectives)

        obj[0] = (
            (1.7057 + self.g5(x))
            * np.cos(0.5 * np.pi * x[0])
            * np.cos(0.5 * np.pi + x[1])
        )
        obj[1] = (
            (1.7057 + self.g5(x))
            * np.cos(0.5 * np.pi * x[0])
            * np.sin(0.5 * np.pi + x[1])
        )
        obj[2] = (1.7057 + self.g5(x)) * np.sin(0.5 * np.pi + x[0])

        return obj

    def cons_func_specific(self, obj):
        cons = np.zeros(self.n_constraints)

        g = self.g(obj)
        cons[0] = (g - 9.0) * (g - 4.0)
        cons[1] = (g - 3.61) * (g - 3.24)

        return -cons

    def exact_pareto(self, n_points):
        pf = np.zeros((n_points, 2))
        pf[:, 0] = np.linspace(0, 1, n_points)
        pf[:, 1] = 1 - pf[:, 0] ** 2
        pf += 0.5

        return pf


class LIRCMOP14(LIRCMOP13):
    """
    LIRCMOP14: 3 obj, 3 cons
    """

    def __init__(self, n_dim=30, n_constraints=3):
        super().__init__(n_dim=n_dim, n_constraints=n_constraints)
        self.problem_name = "LIRCMOP14"
        self.f_opt = None

    def cons_func_specific(self, obj):
        cons = np.zeros(self.n_constraints)

        g = self.g(obj)
        cons[0] = (g - 9.0) * (g - 4.0)
        cons[1] = (g - 3.61) * (g - 3.24)
        cons[2] = (g - 3.0625) * (g - 2.56)

        return -cons

    def exact_pareto(self, n_points):
        pf = np.zeros((n_points, 2))
        pf[:, 0] = np.linspace(0, 1, n_points)
        pf[:, 1] = 1 - pf[:, 0] ** 2
        pf += 0.5

        return pf
