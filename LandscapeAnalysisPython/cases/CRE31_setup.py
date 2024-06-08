import numpy as np

from optimisation.setup import Setup


class CRE31Setup(Setup):
    """ This is the car side impact problem
    Reference:
    Himanshu Jain, Kalyanmoy Deb: An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point Based
    Nondominated Sorting Approach, Part II: Handling Constraints and Extending to an Adaptive Approach.
    IEEE Trans. Evolutionary Computation 18(4): 602-622 (2014)
    Implementation based on Tanabe 2020
    Ryoji Tanabe, Hisao Ishibuchi, "An Easy-to-use Real-world Multi-objective Problem Suite"
    """

    def __init__(self):
        super().__init__()

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', 7, 'c',
                           lower=[0.5, 0.45, 0.5, 0.5, 0.875, 0.4, 0.4],
                           upper=[1.5, 1.35, 1.5, 1.5, 2.625, 1.2, 1.2],
                           value=[1.0, 1.00, 1.0, 1.0, 1.000, 1.0, 1.0], scale=1.0)

    def set_constraints(self, prob, **kwargs):
        prob.add_con_group('con', 10, lower=None, upper=0.0)

    def set_objectives(self, prob, **kwargs):
        prob.add_obj('f_1')
        prob.add_obj('f_2')
        prob.add_obj('f_3')

    def obj_func(self, x_dict, **kwargs):

        x = x_dict['x_vars']

        # objective functions
        obj = np.zeros(3)
        obj[0] = 1.98 + 4.9 * x[0] + 6.67 * x[1] + 6.98 * x[2] + 4.01 * x[3] + 1.78 * x[4] + 0.00001 * x[5] + 2.73 * x[6]
        obj[1] = 4.72 - 0.5 * x[3] - 0.19 * x[1] * x[2]

        v_mbp = 10.58 - 0.674 * x[0] * x[1] - 0.67275 * x[1]
        v_fd = 16.45 - 0.489 * x[2] * x[6] - 0.843 * x[4] * x[5]
        obj[2] = 0.5 * (v_mbp + v_fd)

        # Constraint functions
        cons = np.zeros(10)
        cons[0] = 1 - (1.16 - 0.3717 * x[1] * x[3] - 0.0092928 * x[2])
        cons[1] = 0.32 - (0.261 - 0.0159 * x[0] * x[1] - 0.06486 * x[0] - 0.019 * x[1] * x[6] + 0.0144 * x[2] * x[4] +
                          0.0154464 * x[5])
        cons[2] = 0.32 - (0.214 + 0.00817 * x[4] - 0.045195 * x[0] - 0.0135168 * x[0] + 0.03099 * x[1] * x[5] -
                          0.018 * x[1] * x[6] + 0.007176 * x[2] + 0.023232 * x[2] - 0.00364 * x[4] * x[5] -
                          0.018 * x[1] * x[1])
        cons[3] = 0.32 - (0.74 - 0.61 * x[1] - 0.031296 * x[2] - 0.031872 * x[6] + 0.227 * x[1] * x[1])
        cons[4] = 32 - (28.98 + 3.818 * x[2] - 4.2 * x[0] * x[1] + 1.27296 * x[5] - 2.68065 * x[6])
        cons[5] = 32 - (33.86 + 2.95 * x[2] - 5.057 * x[0] * x[1] - 3.795 * x[1] - 3.4431 * x[6] + 1.45728)
        cons[6] = 32 - (46.36 - 9.9 * x[1] - 4.4505 * x[0])
        cons[7] = 4 - obj[1]
        cons[8] = 9.9 - v_mbp
        cons[9] = 15.7 - v_fd

        # Calculate the constraint violation values
        cons[cons >= 0.0] = 0.0
        cons[cons < 0] = -cons[cons < 0]

        performance = None

        return obj, cons, performance

