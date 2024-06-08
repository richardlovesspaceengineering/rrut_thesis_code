import numpy as np

from optimisation.setup import Setup


class BukinN6Setup(Setup):

    def __init__(self, n_var=2):
        super().__init__()
        self.n_var = n_var


    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', 1, 'c',
                           lower=-15, upper=-5.0,
                           value=0.0, scale=1.0)
        prob.add_var_group('x_vars', 1, 'c',
                           lower=-3.0, upper=3.0,
                           value=0.0, scale=1.0)

    def set_constraints(self, prob, **kwargs):
        pass

    def set_objectives(self, prob, **kwargs):
        prob.add_obj('f_1')

    def obj_func(self, x_dict, **kwargs):

        x = x_dict['x_vars']
        x1 = x[0]
        x2 = x[1]

        obj = 100*(np.abs(x2-0.01*x1**2))**0.5 + 0.01*np.abs(x1+10)

        cons = None
        performance = None

        return obj, cons, performance
