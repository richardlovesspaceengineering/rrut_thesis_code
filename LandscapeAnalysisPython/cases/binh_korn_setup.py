import numpy as np

from optimisation.setup import Setup


class BinhKornSetup(Setup):

    def __init__(self):
        super().__init__()

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', 2, 'c', lower=[0.0, 0.0], upper=[5.0, 3.0], value=[2.0, 2.0], scale=1.0)

    def set_constraints(self, prob, **kwargs):
        prob.add_con_group('con', 2, lower=None, upper=0.0)

    def set_objectives(self, prob, **kwargs):
        prob.add_obj('f_1')
        prob.add_obj('f_2')

    def obj_func(self, x_dict, **kwargs):

        x = x_dict['x_vars']
        obj = self.obj_func_specific(x)
        cons = self.cons_func_specific(x)
        performance = None

        return obj, cons, performance

    def obj_func_specific(self,x):

        obj = np.zeros(2)
        obj[0] = 4.0*x[0]**2.0 + 4.0*x[1]**2.0
        obj[1] = (x[0] - 5.0)**2.0 + (x[1] - 5.0)**2.0
        return obj

    def cons_func_specific(self,x):

        cons = np.zeros(2)
        cons[0] = (x[0] - 5.0)**2.0 + x[1]**2.0 - 25.0
        cons[1] = 7.7 - (x[0] - 8.0)**2.0 - (x[1] + 3.0)**2.0

        return cons


