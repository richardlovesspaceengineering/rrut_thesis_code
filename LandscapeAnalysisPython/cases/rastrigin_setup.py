import numpy as np

from optimisation.setup import Setup


class RastriginSetup(Setup):

    def __init__(self, n=2, A=10.0):
        super().__init__()
        self.n = n
        self.A = A

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.n, 'c',
                           lower=-5.0*np.ones(self.n), upper=5.0*np.ones(self.n),
                           value=0.0*np.ones(self.n), scale=1.0)

    def set_constraints(self, prob, **kwargs):
        pass

    def set_objectives(self, prob, **kwargs):
        prob.add_obj('f_1')

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']

        obj = self.obj_func_specific(x)
        cons = self.cons_func_specific(x)
        performance = None

        return obj, cons, performance


    def obj_func_specific(self,x):

        z = x**2.0 - self.A*np.cos(2.0*np.pi*x)
        obj = np.array([self.A*self.n + np.sum(z)])

        return obj

    def cons_func_specific(self,x):
        return None
