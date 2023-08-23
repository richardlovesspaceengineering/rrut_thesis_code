import numpy as np

from optimisation.setup import Setup

"""https://arxiv.org/pdf/1308.4008.pdf"""

class UrsumWavesSetup(Setup):

    def __init__(self, n_var=2):
        super().__init__()
        self.n_var = n_var


    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', 1, 'c',
                           lower=-0.9, upper=1.2,
                           value=0.0, scale=1.0)
        prob.add_var_group('x_vars', 1, 'c',
                           lower=-1.2, upper=1.2,
                           value=0.0, scale=1.0)
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

    def obj_func_specific(self, x):
        x1 = x[0]
        x2 = x[1]

        obj = -0.9*x1**2 + (x2**2 - 4.5*x2**2)*x1*x2 + 4.7*np.cos(3*x1-x2**2*(2+x1))*np.sin(2.5*np.pi*x1)
        return obj

    def cons_func_specific(self, x):
        cons = None
        return cons


