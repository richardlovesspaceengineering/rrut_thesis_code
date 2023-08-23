import numpy as np

from optimisation.setup import Setup


class GoldsteinSetup(Setup):

    def __init__(self, n_var=2):
        super().__init__()
        self.n_var = n_var


    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.n_var, 'c',
                           lower=-2.0*np.ones(self.n_var), upper=2.0*np.ones(self.n_var),
                           value=0.0*np.ones(self.n_var), scale=1.0)

    def set_constraints(self, prob, **kwargs):
        pass

    def set_objectives(self, prob, **kwargs):
        prob.add_obj('f_1')

    def obj_func(self, x_dict, **kwargs):

        x = x_dict['x_vars']
        x1 = x[0]
        x2 = x[1]

        obj = ((1 + (x1+x2+1)**2 * (19 - 14*x1 + 3*x1**2 -14*x2 + 6*x1*x2 + 3 *x2**2))*
               (30 + (2*x1 - 3*x2)**2 * (18 - 32 * x1 + 12 * x1 ** 2 + 48 * x2 - 36 * x1 * x2 + 27 * x2 ** 2)))

        cons = None
        performance = None

        return obj, cons, performance
