import numpy as np

from optimisation.setup import Setup


class RosenbrockSetup(Setup):

    def __init__(self, n_var=2, a=1.0, b=100.0):
        super().__init__()
        self.n_var = n_var
        self.a = a
        self.b = b

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.n_var, 'c',
                           lower=-100.0*np.ones(self.n_var), upper=100.0*np.ones(self.n_var),
                           value=0.0*np.ones(self.n_var), scale=1.0)

    def set_constraints(self, prob, **kwargs):
        pass

    def set_objectives(self, prob, **kwargs):
        prob.add_obj('f_1')

    def obj_func(self, x_dict, **kwargs):

        x = x_dict['x_vars']
        obj = self.obj_func_specific(x)
        cons = None
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        obj = 0.0
        for i in range(len(x) - 1):
            obj += self.a * (1.0 - x[i]) ** 2.0 + self.b * (x[i + 1] - x[i] ** 2.0) ** 2.0

        return obj

    def cons_func_specific(self,x):
        return None