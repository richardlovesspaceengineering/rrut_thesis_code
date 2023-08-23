import numpy as np

from optimisation.setup import Setup


class CRE23Setup(Setup):
    """ T.
    Reference:
    T. Ray and K. M. Liew, "A swarm metaphor for multiobjective design optimization,"
    Eng. opt., vol. 34, no. 2, pp. 141–153, 2002.
    Implementation based on Tanabe 2020
    Ryoji Tanabe, Hisao Ishibuchi, "An Easy-to-use Real-world Multi-objective Problem Suite"
    """

    def __init__(self):
        super().__init__()

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', 4, 'c',
                           lower=[55.0, 75.0, 1000.0, 11.0],
                           upper=[80.0, 110.0, 3000.0, 20.0],
                           value=[65.0, 90.0, 2000.0, 15.0], scale=1.0)

    def set_constraints(self, prob, **kwargs):
        prob.add_con_group('con', 4, lower=None, upper=0.0)

    def set_objectives(self, prob, **kwargs):
        prob.add_obj('f_1')
        prob.add_obj('f_2')

    def obj_func(self, x_dict, **kwargs):

        x = x_dict['x_vars']

        # objective functions
        obj = np.zeros(2)
        obj[0] = 4.9 * 1e-5 * (x[1] * x[1] - x[0] * x[0]) * (x[3] - 1.0)
        obj[1] = ((9.82 * 1e6) * (x[1] * x[1] - x[0] * x[0])) / (x[2] * x[3] *
                                                        (x[1] * x[1] * x[1] - x[0] * x[0] * x[0]))

        # Constraint functions
        cons = np.zeros(4)
        cons[0] = (x[1] - x[0]) - 20.0
        cons[1] = 0.4 - (x[2] / (3.14 * (x[1] * x[1] - x[0] * x[0])))
        cons[2] = 1.0 - (2.22 * 1e-3 * x[2] * (x[1] * x[1] * x[1] - x[0] * x[0] * x[0])) / \
                  (((x[1] * x[1] - x[0] * x[0]))** 2)
        cons[3] = (2.66 * 1e-2 * x[2] * x[3] * (x[1] * x[1] * x[1] - x[0] * x[0] * x[0])) / \
                  (x[1] * x[1] - x[0] * x[0]) - 900.0

        # Calculate the constraint violation values
        cons[cons >= 0.0] = 0.0
        cons[cons < 0] = -cons[cons < 0]

        performance = None

        return obj, cons, performance

