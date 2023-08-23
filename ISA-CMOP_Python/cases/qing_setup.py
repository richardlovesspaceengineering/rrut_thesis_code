import numpy as np

from optimisation.setup import Setup

"""https://arxiv.org/pdf/1308.4008.pdf"""

class QingSetup(Setup):

    def __init__(self, n_var=2):
        super().__init__()
        self.n_var = n_var


    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.n_var, 'c',
                           lower=-500.0, upper=1.0,
                           value=500.0, scale=1.0)

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
        obj = 0
        for i in range(len(x)):
            xi = x[i]
            obj+=(xi**2-i)**2
        return obj


    def cons_func_specific(self, x):
        cons = None
        return cons


