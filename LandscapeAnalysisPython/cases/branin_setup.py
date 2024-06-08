import numpy as np

from optimisation.setup import Setup


class BraninSetup(Setup):

    def __init__(self, n_var=2):
        super().__init__()
        self.n_var = n_var


    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.n_var, 'c',
                           lower=-5.0*np.pi*np.ones(self.n_var), upper=15.0*np.pi*np.ones(self.n_var),
                           value=0.0*np.ones(self.n_var), scale=1.0)

    def set_constraints(self, prob, **kwargs):
        pass

    def set_objectives(self, prob, **kwargs):
        prob.add_obj('f_1')

    def obj_func(self, x_dict, **kwargs):

        x = x_dict['x_vars']
        x1 = x[0]
        x2 = x[1]

        obj =(x2 - 5.1 * (x1**2) / (4 * np.pi ** 2) + 5 * x1 / np.pi - 6)** 2 + \
             10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10

        cons = None
        performance = None

        return obj, cons, performance
