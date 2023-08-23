import numpy
import numpy as np
from optimisation.setup import Setup
from optimisation.util.reference_directions import UniformReferenceDirection
from optimisation.util.non_dominated_sorting import NonDominatedSorting

"""
Ported the LSMOP implementation across from PlatEMO
"Test Problems for Large-Scale Multiobjective and Many-Objective Optimization"
"""


class LSMOPSetup(Setup):
    """
    Wrapper class for standarised LSMOP methods
    """
    def __init__(self, dim, n_objectives):
        super().__init__()

        self.dim = dim
        self.n_objectives = n_objectives

        # Number of subcomponents in each variable group
        self.nk = 5

        # Number of variables in each subcomponent
        c0 = np.array([0.1])
        self.c = self.chaos_func(c0)
        for i in range(self.n_objectives-1):
            self.c = np.hstack((self.c, self.chaos_func(self.c[-1])))

        self.kd = np.floor(self.c / np.sum(self.c) * (self.dim - self.n_objectives + 1) / self.nk)
        self.nd = np.hstack((np.array([0.]), np.cumsum(self.kd * self.nk)))
        self.kd = self.kd.astype(int)
        self.nd = self.nd.astype(int)

    def linear_linkage(self, x, x0):
        i_range = np.arange(0, len(x)) + 1.0
        La = 1.0 + i_range / len(x)
        x_lin = La * x - x0 * 10.0
        return x_lin

    def nonlinear_linkage(self, x, x0):
        i_range = np.arange(0, len(x)) + 1.0
        La = 1.0 + np.cos(0.5 * np.pi * i_range / len(x))
        x_nonlin = La * x - x0 * 10.0
        return x_nonlin

    def g_func(self, x, func_args):
        func = []
        for idx, item in enumerate(func_args):
            if 'n1' in item:
                func.append(self.n1)
            elif 'n2' in item:
                func.append(self.n2)
            elif 'n3' in item:
                func.append(self.n3)
            elif 'n4' in item:
                func.append(self.n4)
            elif 'n5' in item:
                func.append(self.n5)
            elif 'n6' in item:
                func.append(self.n6)

        # Even base function
        G = np.zeros(self.n_objectives)
        for i in range(0, self.n_objectives, 2):
            for j in range(self.nk):
                start = self.nd[i] + self.n_objectives - 1 + j*self.kd[i] + 1
                end = self.nd[i] + self.n_objectives - 1 + (j+1)*self.kd[i]
                G[i] += func[0](x[start:end])

        # Odd base function
        for i in range(1, self.n_objectives, 2):
            for j in range(self.nk):
                start = self.nd[i] + self.n_objectives - 1 + j*self.kd[i] + 1
                end = self.nd[i] + self.n_objectives - 1 + (j+1)*self.kd[i]
                G[i] += func[1](x[start:end])

        # Normalise
        return G / self.kd / self.nk

    def s_lin(self, x):
        s = np.ones(self.n_objectives)
        for i in range(self.n_objectives-1):
            s[i] *= np.prod(x[:self.n_objectives-i-1])

            if i > 0:
                s[i] *= (1 - x[self.n_objectives-i-1])

        s[-1] = (1 - x[0])
        return s

    def s_nonlin(self, x):
        s = np.ones(self.n_objectives)
        for i in range(self.n_objectives-1):
            s[i] *= np.prod(np.cos(0.5 * np.pi * x[:self.n_objectives-i-1]))

            if i > 0:
                s[i] *= np.sin(0.5 * np.pi * x[self.n_objectives-i-1])

        s[-1] = np.sin(0.5 * np.pi * x[0])
        return s

    def s_full(self, x, g):
        obj = np.zeros(self.n_objectives)

        obj[:self.n_objectives-1] = x[:self.n_objectives-1]

        obj[-1] = (2 + g) * (self.n_objectives - np.sum(x[:self.n_objectives-1] *
                             (1 + np.sin(3.0 * np.pi * x[:self.n_objectives-1])) / (2 + g)))

        return obj

    @staticmethod
    def chaos_func(c):
        return 3.8 * c * (1 - c)

    # Sphere Function
    @staticmethod
    def n1(x):
        f = np.sum(x**2)
        return f

    # Schwefel Function
    @staticmethod
    def n2(x):
        f = np.max(np.abs(x))
        return f

    # Rosenbrock Function
    @staticmethod
    def n3(x):
        d = len(x)
        f = 0.0
        for i in range(d-1):
            f += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1.0 - x[i]) ** 2
        return f

    # Rastrigin Function
    @staticmethod
    def n4(x):
        f = np.sum(x ** 2 - 10 * np.cos(2.0 * np.pi * x) + 10)
        return f

    # Griewank Function
    @staticmethod
    def n5(x):
        term = 1
        for i in range(len(x)):
            term *= np.cos(x[i] / np.sqrt(i+1))

        f = np.sum(x ** 2) / 4000 - term + 1
        return f

    # Ackley Function
    @staticmethod
    def n6(x):
        fit = np.sum(x ** 2)
        d = len(x)
        f = -20.0 * np.exp(-0.2 * np.sqrt(fit / d)) - np.exp(np.sum(np.cos(2.0 * np.pi * x)) / d) + 20.0 + np.exp(1)
        return f


class LSMOP1(LSMOPSetup):
    """
    LSMOP1: "supp:A constrained multi-objective evolutionary algorithm using valuable infeasible solutions"
    """

    def __init__(self, n_obj=2):
        self.problem_name = 'LSMOP1'
        self.n_objectives = n_obj
        self.dim = 100 * self.n_objectives
        self.n_constraints = 0
        super().__init__(dim=self.dim, n_objectives=self.n_objectives)

        # Optimum Pareto Front
        self.f_opt = self.exact_pareto(50)
        self.var_opt = None

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)
        self.ub[self.n_objectives-1:] *= 10

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub, value=0.5 * np.ones(self.dim),
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
        # Variable Linkages
        x[self.n_objectives:self.dim] = self.linear_linkage(x[self.n_objectives:self.dim], x[0])

        # Convergence Function
        g = self.g_func(x, ['n1', 'n1'])

        f_prod = self.s_lin(x[:self.n_objectives])
        obj = (1 + g) * f_prod

        return obj

    def cons_func_specific(self, x):
        return None

    def exact_pareto(self, n_points=15):
        if self.n_objectives == 2:
            pf = get_ref_dirs(self.n_objectives)
        elif self.n_objectives == 3:
            vec = np.linspace(0, 1, n_points)
            pf = np.vstack((vec, (1-vec), (1-vec))).T
        else:
            pf = None
        return pf


class LSMOP2(LSMOP1):
    """
    LSMOP2: "supp:A constrained multi-objective evolutionary algorithm using valuable infeasible solutions"
    """

    def __init__(self, n_obj=2):
        super().__init__(n_obj=n_obj)
        self.problem_name = 'LSMOP2'

    def obj_func_specific(self, x):
        # Variable Linkages
        x[self.n_objectives:self.dim] = self.linear_linkage(x[self.n_objectives:self.dim], x[0])

        # Convergence Function
        g = self.g_func(x, ['n5', 'n2'])

        f_prod = self.s_lin(x[:self.n_objectives])
        obj = (1 + g) * f_prod

        return obj


class LSMOP3(LSMOP1):
    """
    LSMOP3: "supp:A constrained multi-objective evolutionary algorithm using valuable infeasible solutions"
    """

    def __init__(self, n_obj=2):
        super().__init__(n_obj=n_obj)
        self.problem_name = 'LSMOP3'

    def obj_func_specific(self, x):
        # Variable Linkages
        x[self.n_objectives:self.dim] = self.linear_linkage(x[self.n_objectives:self.dim], x[0])

        # Convergence Function
        g = self.g_func(x, ['n4', 'n3'])

        f_prod = self.s_lin(x[:self.n_objectives])
        obj = (1 + g) * f_prod

        return obj


class LSMOP4(LSMOP1):
    """
    LSMOP4: "supp:A constrained multi-objective evolutionary algorithm using valuable infeasible solutions"
    """

    def __init__(self, n_obj=2):
        super().__init__(n_obj=n_obj)
        self.problem_name = 'LSMOP4'

    def obj_func_specific(self, x):
        # Variable Linkages
        x[self.n_objectives:self.dim] = self.linear_linkage(x[self.n_objectives:self.dim], x[0])

        # Convergence Function
        g = self.g_func(x, ['n6', 'n5'])

        f_prod = self.s_lin(x[:self.n_objectives])
        obj = (1 + g) * f_prod

        return obj


class LSMOP5(LSMOP1):
    """
    LSMOP5: "supp:A constrained multi-objective evolutionary algorithm using valuable infeasible solutions"
    """

    def __init__(self, n_obj=2):
        super().__init__(n_obj=n_obj)
        self.problem_name = 'LSMOP5'

    def obj_func_specific(self, x):
        # Variable Linkages
        x[self.n_objectives:self.dim] = self.nonlinear_linkage(x[self.n_objectives:self.dim], x[0])

        # Convergence Function
        g = self.g_func(x, ['n1', 'n1'])
        g_add = np.hstack((g[1:], 0.0))

        f_prod = self.s_nonlin(x[:self.n_objectives])
        obj = (1 + g + g_add) * f_prod

        return obj

    def exact_pareto(self, n_points=15):
        if self.n_objectives == 2:
            pf = generic_sphere(get_ref_dirs(self.n_objectives))
        elif self.n_objectives == 3:
            vec = np.linspace(0, 1, n_points)
            pf = np.vstack((np.sin(vec) * np.cos(vec), np.sin(vec) ** 2, np.cos(vec))).T
        else:
            pf = None
        return pf


class LSMOP6(LSMOP5):
    """
    LSMOP6: "supp:A constrained multi-objective evolutionary algorithm using valuable infeasible solutions"
    """

    def __init__(self, n_obj=2):
        super().__init__(n_obj=n_obj)
        self.problem_name = 'LSMOP6'

    def obj_func_specific(self, x):
        # Variable Linkages
        x[self.n_objectives:self.dim] = self.nonlinear_linkage(x[self.n_objectives:self.dim], x[0])

        # Convergence Function
        g = self.g_func(x, ['n3', 'n2'])
        g_add = np.hstack((g[1:], 0.0))

        f_prod = self.s_nonlin(x[:self.n_objectives])
        obj = (1 + g + g_add) * f_prod

        return obj


class LSMOP7(LSMOP5):
    """
    LSMOP7: "supp:A constrained multi-objective evolutionary algorithm using valuable infeasible solutions"
    """

    def __init__(self, n_obj=2):
        super().__init__(n_obj=n_obj)
        self.problem_name = 'LSMOP7'

    def obj_func_specific(self, x):
        # Variable Linkages
        x[self.n_objectives:self.dim] = self.nonlinear_linkage(x[self.n_objectives:self.dim], x[0])

        # Convergence Function
        g = self.g_func(x, ['n6', 'n3'])
        g_add = np.hstack((g[1:], 0.0))

        f_prod = self.s_nonlin(x[:self.n_objectives])
        obj = (1 + g + g_add) * f_prod

        return obj


class LSMOP8(LSMOP5):
    """
    LSMOP8: "supp:A constrained multi-objective evolutionary algorithm using valuable infeasible solutions"
    """

    def __init__(self, n_obj=2):
        super().__init__(n_obj=n_obj)
        self.problem_name = 'LSMOP8'

    def obj_func_specific(self, x):
        # Variable Linkages
        x[self.n_objectives:self.dim] = self.nonlinear_linkage(x[self.n_objectives:self.dim], x[0])

        # Convergence Function
        g = self.g_func(x, ['n5', 'n1'])
        g_add = np.hstack((g[1:], 0.0))

        f_prod = self.s_nonlin(x[:self.n_objectives])
        obj = (1 + g + g_add) * f_prod

        return obj


class LSMOP9(LSMOP5):
    """
    LSMOP9: "supp:A constrained multi-objective evolutionary algorithm using valuable infeasible solutions"
    """

    def __init__(self, n_obj=2):
        super().__init__(n_obj=n_obj)
        self.problem_name = 'LSMOP9'

    def obj_func_specific(self, x):
        # Variable Linkages
        x[self.n_objectives:self.dim] = self.nonlinear_linkage(x[self.n_objectives:self.dim], x[0])

        # Convergence Function
        g = self.g_func(x, ['n1', 'n6'])

        obj = self.s_full(x[:self.n_objectives], np.sum(g))
        return obj

    def exact_pareto(self, n_points=15):
        if self.n_objectives == 2:
            x = np.linspace(0.0, 1.0, n_points)
            y = 2.0 * (2.0 - x / 2.0 * (1 + np.sin(3.0 * np.pi * x)))
            obj_array = np.vstack((x, y)).T
            fronts = NonDominatedSorting().do(obj_array)
            pf = obj_array[fronts[0]]
        elif self.n_objectives == 3:
            x = np.linspace(0.0, 1.0, n_points)
            y = np.linspace(0.0, 1.0, n_points)
            z = 2.0 * (2.0 - x / 2.0 * (1 + np.sin(3.0 * np.pi * x)))
            obj_array = np.hstack((x, y, z)).T
            fronts = NonDominatedSorting().do(obj_array)
            pf = obj_array[fronts[0]]
        else:
            pf = None
        return pf


# Utils ----------------------------------------------------------------------------------------------------------------


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