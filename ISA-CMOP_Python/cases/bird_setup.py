import numpy as np

from optimisation.setup import Setup


class BirdSetup(Setup):

    def __init__(self, n_var=2):
        super().__init__()
        self.n_var = n_var


    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.n_var, 'c',
                           lower=-2.0*np.pi*np.ones(self.n_var), upper=2.0*np.pi*np.ones(self.n_var),
                           value=0.0*np.ones(self.n_var), scale=1.0)

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
        x1 = x[0]
        x2 = x[1]

        obj = np.sin(x1) * np.exp((1 - np.cos(x2))**2) + np.cos(x2)* np.exp((1 - np.sin(x1))**2) + (x1 - x2)** 2

        return obj

    def cons_func_specific(self, x):
        cons = None
        return cons
