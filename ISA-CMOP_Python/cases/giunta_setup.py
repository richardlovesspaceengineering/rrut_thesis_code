import numpy as np

from optimisation.setup import Setup


class GiuntaSetup(Setup):

    def __init__(self, n_var=2):
        super().__init__()
        self.n_var = n_var


    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', 1, 'c',
                           lower=-1.0, upper=1.0,
                           value=0.0, scale=1.0)
        prob.add_var_group('x_vars', 1, 'c',
                           lower=-1.0, upper=1.0,
                           value=0.0, scale=1.0)

    def set_constraints(self, prob, **kwargs):
        pass

    def set_objectives(self, prob, **kwargs):
        prob.add_obj('f_1')

    def obj_func(self, x_dict, **kwargs):

        x = x_dict['x_vars']
        x1 = x[0]
        x2 = x[1]

        obj = 0.6
        xi=x1
        arg = 16/15*xi - 1
        obj += np.sin(arg) + np.sin(arg)**2 + np.sin(4*arg)/50
        xi = x2
        arg = 16 / 15 * xi - 1
        obj += np.sin(arg) + np.sin(arg) ** 2 + np.sin(4 * arg) / 50


        cons = None
        performance = None

        return obj, cons, performance
