import numpy as np
import sys

from optimisation.setup import Setup


class CExpSetup(Setup):
    def __init__(self):
        super().__init__()
        self.problem_name = 'CEXP'
        self.n_objectives = 2
        self.dim = 2
        self.n_constraints = 2

        self.ub = np.zeros(self.dim)
        self.lb = np.zeros(self.dim)
        self.lb[0] = 0.1
        self.lb[1] = 1.0
        self.ub[0] = 1.0
        self.ub[1] = 5.0


        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub,
                           value=[0.5, 3.0], scale=1.0)

    def set_constraints(self, prob, **kwargs):
        prob.add_con_group('con', self.n_constraints, lower=0.0, upper=None)

    def set_objectives(self, prob, **kwargs):
        for i in range(self.n_objectives):
            prob.add_obj(f"f_{i}")

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']

        obj = self.obj_func_specific(x)
        cons = self.cons_func_specific(x)
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        obj = np.zeros(self.n_objectives)
        obj[0] = x[0]
        obj[1] = (1.0 + x[1]) / x[0]

        return obj

    def cons_func_specific(self, x):
        cons = np.zeros(self.n_constraints)
        cons[0] = x[1] + 9 * x[0] - 6
        cons[1] = -1 * x[1] + 9 * x[0] - 1

        cons *= -1

        return cons


class TNKSetup(Setup):
    def __init__(self):
        super().__init__()
        self.problem_name = 'TNK'
        self.n_objectives = 2
        self.dim = 2
        self.n_constraints = 2

        self.ub = np.zeros(self.dim)
        self.lb = np.zeros(self.dim)
        self.lb[0] = 0.0
        self.lb[1] = 0.0
        self.ub[0] = np.pi
        self.ub[1] = np.pi


        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub,
                           value=[np.pi/2, np.pi/2], scale=1.0)

    def set_constraints(self, prob, **kwargs):
        prob.add_con_group('con', self.n_constraints, lower=0.0, upper=None)

    def set_objectives(self, prob, **kwargs):
        for i in range(self.n_objectives):
            prob.add_obj(f"f_{i}")

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']

        obj = self.obj_func_specific(x)
        cons = self.cons_func_specific(x)
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        obj = np.zeros(self.n_objectives)
        obj[0] = x[0]
        obj[1] = x[1]

        return obj

    def cons_func_specific(self, x):
        cons = np.zeros(self.n_constraints)
        cons[0] = x[0]**2 + x[1]**2 - 1 - 0.1*np.cos(16*np.arctan(x[1]/x[0]))
        cons[1] = -1*((x[0]-0.5)**2 + (x[1]-0.5)**2 - 0.5)

        cons *= -1

        return cons

class BNHSetup(Setup):
    """
    problem MOP-C1 Binh from CoelloBook page 188
    """
    def __init__(self):
        super().__init__()
        self.problem_name = 'BNH'
        self.n_objectives = 2
        self.dim = 2
        self.n_constraints = 2

        self.ub = np.zeros(self.dim)
        self.lb = np.zeros(self.dim)
        self.lb[0] = 0.0
        self.lb[1] = 0.0
        self.ub[0] = 5.0
        self.ub[1] = 3.0


        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub,
                           value=[2.5, 1.5], scale=1.0)

    def set_constraints(self, prob, **kwargs):
        prob.add_con_group('con', self.n_constraints, lower=0.0, upper=None)

    def set_objectives(self, prob, **kwargs):
        for i in range(self.n_objectives):
            prob.add_obj(f"f_{i}")

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']

        obj = self.obj_func_specific(x)
        cons = self.cons_func_specific(x)
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        obj = np.zeros(self.n_objectives)
        obj[0] = 4*x[0]**2+4*x[1]**2
        obj[1] = (x[0]-5)**2 + (x[1]-5)**2

        return obj

    def cons_func_specific(self, x):
        cons = np.zeros(self.n_constraints)
        cons[0] = -1*((x[0]-5)**2 + x[1]-25)
        cons[1] = (x[0]-8)**2 + (x[1]-3)**2 - 7.7

        cons *= -1

        return cons


class SRNSetup(Setup):
    def __init__(self):
        super().__init__()
        self.problem_name = 'SRN'
        self.n_objectives = 2
        self.dim = 2
        self.n_constraints = 2

        self.ub = np.zeros(self.dim)
        self.lb = np.zeros(self.dim)
        self.lb[0] = -20.0
        self.lb[1] = -20.0
        self.ub[0] = 20.0
        self.ub[1] = 20.0


        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub,
                           value=[0.0, 0.0], scale=1.0)

    def set_constraints(self, prob, **kwargs):
        prob.add_con_group('con', self.n_constraints, lower=0.0, upper=None)

    def set_objectives(self, prob, **kwargs):
        for i in range(self.n_objectives):
            prob.add_obj(f"f_{i}")

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']

        obj = self.obj_func_specific(x)
        cons = self.cons_func_specific(x)
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        obj = np.zeros(self.n_objectives)
        obj[0] = 2 + (x[0]-2)**2 + (x[1]-1)**2
        obj[1] = 9*x[0] - (x[1]-1)**2

        return obj

    def cons_func_specific(self, x):
        cons = np.zeros(self.n_constraints)
        cons[0] = x[0]**2 + x[1]**2 - 225
        cons[1] = x[0]-3*x[1]+10

        return cons

class NBPSetup(Setup):
    """
    Book Forrester (Engineering Design via Surrogate Modelling p.198 - Nowacki Beam Problem
    """
    def __init__(self):
        super().__init__()
        self.problem_name = 'NBP'
        self.n_objectives = 2
        self.dim = 2
        self.n_constraints = 5

        self.ub = np.zeros(self.dim)
        self.lb = np.zeros(self.dim)
        self.lb[0] = 20.0
        self.lb[1] = 10.0
        self.ub[0] = 250.0
        self.ub[1] = 50.0


        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub,
                           value=[125, 30], scale=1.0)

    def set_constraints(self, prob, **kwargs):
        prob.add_con_group('con', self.n_constraints, lower=0.0, upper=None)

    def set_objectives(self, prob, **kwargs):
        for i in range(self.n_objectives):
            prob.add_obj(f"f_{i}")

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']

        obj = self.obj_func_specific(x)
        cons = self.cons_func_specific(x)
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        obj = np.zeros(self.n_objectives)
        h = x[0]
        b = x[1]

        l = 1500
        F = 5000
        E = 216620  # GPa
        v = 0.27
        G = 86650

        A = b * h  # cross sectional area of the beam
        sigma = 6 * F * l / (b * h ** 2)  # bending stress

        obj[0] = A
        obj[1] = sigma

        return obj

    def cons_func_specific(self, x):
        h = x[0]
        b = x[1]

        l = 1500
        F = 5000
        E = 216620  # GPa
        v = 0.27
        G = 86650

        sigma = 6 * F * l / (b * h ** 2)  # bending stress

        delta = (F * (l ** 3)) / (3 * E * ((b * h ** 3) / 12))  # maximum tip deflection
        tau = 3 * F / (2 * b * h)  # maximum allowable shear stress
        hb = h / b  # height to breadth ratio

        F_crit = -1 * (4 / (l ** 2)) * (
                    G * (((b * (h ** 3)) / 12) + (((b ** 3) * h) / 12)) * E * (((b ** 3) * h) / 12) / (
                        1 - (v ** 2))) ** 0.5  # failure force of buckling

        cons = np.zeros(self.n_constraints)
        cons[0] = delta - 5 # tip deflection of beam
        cons[1] = sigma - 240 # yield stress of material (mild steel)
        cons[2] = tau - 120 # shear stress less than half of yield stress
        cons[3] = hb - 10 # height to breadth ratio of beam
        cons[4] = F_crit + 2*F # twist buckling failure

        return cons

class BiCop1Setup(Setup):
    """Datta2016"""
    def __init__(self):
        super().__init__()
        self.problem_name = 'BICOP1'
        self.n_objectives = 2
        self.dim = 10
        self.n_constraints = 1

        self.ub = np.ones(self.dim)
        self.lb = np.zeros(self.dim)

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub,
                           value=[0.5*np.ones(self.dim)], scale=1.0)

    def set_constraints(self, prob, **kwargs):
        prob.add_con_group('con', self.n_constraints, lower=0.0, upper=None)

    def set_objectives(self, prob, **kwargs):
        for i in range(self.n_objectives):
            prob.add_obj(f"f_{i}")

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']

        obj = self.obj_func_specific(x)
        cons = self.cons_func_specific(x)
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        obj = np.zeros(self.n_objectives)
        x = np.array(x)
        g1 = 1 + 9 * np.sum(x[1:] / 9)
        obj[0] = x[0]*g1
        obj[1] = g1 - np.sqrt(obj[0]/g1)

        return obj

    def cons_func_specific(self, x):
        cons = np.zeros(self.n_constraints)
        x = np.array(x)
        cons[0] = 1 + 9 * np.sum(x[1:] / 9)

        cons *= -1

        return cons

class BiCop2Setup(Setup):
    """Datta2016"""
    def __init__(self):
        super().__init__()
        self.problem_name = 'BICOP2'
        self.n_objectives = 2
        self.dim = 10
        self.n_constraints = 2

        self.ub = np.ones(self.dim)
        self.lb = np.zeros(self.dim)

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub,
                           value=[0.5*np.ones(self.dim)], scale=1.0)

    def set_constraints(self, prob, **kwargs):
        prob.add_con_group('con', self.n_constraints, lower=0.0, upper=None)

    def set_objectives(self, prob, **kwargs):
        for i in range(self.n_objectives):
            prob.add_obj(f"f_{i}")

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']

        obj = self.obj_func_specific(x)
        cons = self.cons_func_specific(x)
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        obj = np.zeros(self.n_objectives)
        x = np.array(x)
        gx = np.sum(x[1:] - np.sin(0.5 * np.pi * x[0])) ** 2

        obj[0] = x[0] + gx
        obj[1] = 1 - x[0] ** 2 + gx

        return obj

    def cons_func_specific(self, x):
        cons = np.zeros(self.n_constraints)
        x = np.array(x)
        a = 0.1
        b = 0.9

        gx = np.sum(x[1:] - np.sin(0.5 * np.pi * x[0])) ** 2

        cons[0] = gx - a
        cons[1] = b - gx

        cons *= -1

        return cons

class TriCopSetup(Setup):
    """Datta2016"""
    def __init__(self):
        super().__init__()
        self.problem_name = 'TriCop'
        self.n_objectives = 3
        self.dim = 2
        self.n_constraints = 3

        self.ub = 4*np.ones(self.dim)
        self.lb = -4*np.ones(self.dim)

        self.int_var = np.array([])
        self.cont_var = np.arange(0, self.dim)

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.dim, 'c', lower=self.lb, upper=self.ub,
                           value=[0.0*np.ones(self.dim)], scale=1.0)

    def set_constraints(self, prob, **kwargs):
        prob.add_con_group('con', self.n_constraints, lower=0.0, upper=None)

    def set_objectives(self, prob, **kwargs):
        for i in range(self.n_objectives):
            prob.add_obj(f"f_{i}")

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']

        obj = self.obj_func_specific(x)
        cons = self.cons_func_specific(x)
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        obj = np.zeros(self.n_objectives)

        obj[0] = 0.5*(x[0]-2)**2 + 1/13*(x[1]+1)**2 + 3
        obj[1] = 1/175*(x[0]+x[1]-3)**2 + 1/17*(2*x[1]-x[0])**2 - 13
        obj[2] = 1/8*(3*x[0]-2*x[1]+4)**2 + 1/27*(x[0]-x[1]+1)**2 + 15

        return obj

    def cons_func_specific(self, x):
        cons = np.zeros(self.n_constraints)

        cons[0] = 4 - 4*x[0] - x[1]
        cons[1] = x[0] + 1
        cons[2] = -x[0] + x[1] + 2

        cons *= -1

        return cons
