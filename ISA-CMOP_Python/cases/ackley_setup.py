import numpy as np

from optimisation.setup import Setup


class AckleySetup(Setup):

    def __init__(self, n_var=2, a=20.0, b=0.2, c=2.0*np.pi):
        super().__init__()
        self.n_var = n_var
        self.a = a
        self.b = b
        self.c = c

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', self.n_var, 'c',
                           lower=-32.768*np.ones(self.n_var), upper=32.768*np.ones(self.n_var),
                           value=0.0*np.ones(self.n_var), scale=1.0)

    def set_constraints(self, prob, **kwargs):
        pass

    def set_objectives(self, prob, **kwargs):
        prob.add_obj('f_1')

    def obj_func(self, x_dict, **kwargs):

        x = x_dict['x_vars']

        term_1 = -self.a * np.exp(-self.b * np.sqrt((1.0/self.n_var)*np.sum(x*x)))
        term_2 = -np.exp((1.0/self.n_var)*np.sum(np.cos(self.c*x)))
        obj = term_1 + term_2 + np.e + self.a

        cons = None
        performance = None

        return obj, cons, performance
