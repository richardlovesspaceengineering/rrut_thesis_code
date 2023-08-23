import numpy as np

from optimisation.setup import Setup


class ZDT2Setup(Setup):

    def __init__(self):
        super().__init__()

    def set_variables(self, prob, **kwargs):
        n = 30
        prob.add_var_group('x_vars', n, 'c', lower=np.zeros(n), upper=np.ones(n), value=0.5*np.ones(n), scale=1.0)

    def set_constraints(self, prob, **kwargs):
        pass

    def set_objectives(self, prob, **kwargs):
        prob.add_obj('f_1')
        prob.add_obj('f_2')

    def obj_func(self, x_dict, **kwargs):
        x = x_dict['x_vars']
        obj = self.obj_func_specific(x)
        cons = self.cons_func_specific(x)
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):

        f_1 = x[0]
        c = np.sum(x[1:], axis=0)
        g = 1.0 + (9.0/(len(x) - 1.0))*c
        f_2 = g*(1.0 - np.power((f_1/g), 2.0))

        obj = np.array([f_1, f_2])

        return obj

    def cons_func_specific(self, x):
        cons = None
        return cons


