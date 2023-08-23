import numpy as np

from optimisation.setup import Setup


class HimmelblauSetup(Setup):

    def __init__(self, n_var=2):
        super().__init__()
        self.n_var = n_var


    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.n_var, 'c',
                           lower=-5.0*np.ones(self.n_var), upper=5.0*np.ones(self.n_var),
                           value=0.0*np.ones(self.n_var), scale=1.0)

    def set_constraints(self, prob, **kwargs):
        pass

    def set_objectives(self, prob, **kwargs):
        prob.add_obj('f_1')

    def obj_func(self, x_dict, **kwargs):

        x = x_dict['x_vars']
        x1 = x[0]
        y1 = x[1]

        obj = (x1**2 + y1 - 11)**2 + (x1 + y1**2 - 7)**2


        cons = None
        performance = None

        return obj, cons, performance
