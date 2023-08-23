import numpy as np

from optimisation.setup import Setup


class CRE22Setup(Setup):
    """ Should be welded beam design problem - check in original Tanabe paper
    Reference:
    T. Ray and K. M. Liew, "A swarm metaphor for multiobjective design optimization,"
    Eng. opt., vol. 34, no. 2, pp. 141–153, 2002.
    Implementation based on Tanabe 2020
    Ryoji Tanabe, Hisao Ishibuchi, "An Easy-to-use Real-world Multi-objective Problem Suite"
    """

    def __init__(self):
        super().__init__()

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', 4, 'c', lower=[0.125, 0.1, 0.1, 0.125], upper=[5.0, 10.0, 10.0, 5.0],
                           value=[2, 2, 2, 2], scale=1.0)

    def set_constraints(self, prob, **kwargs):
        prob.add_con_group('con', 4, lower=None, upper=0.0)

    def set_objectives(self, prob, **kwargs):
        prob.add_obj('f_1')
        prob.add_obj('f_2')

    def obj_func(self, x_dict, **kwargs):

        x = x_dict['x_vars']

        # Problem constants
        P = 6000
        L = 14
        E = 30 * 1e6
        G = 12 * 1e6
        tauMax = 13600
        sigmaMax = 30000

        # objective functions
        obj = np.zeros(2)
        obj[0] = (1.10471 * x[0] * x[0] * x[1]) + (0.04811 * x[2] * x[3]) * (14.0 + x[1])
        obj[1] = (4 * P * L * L * L) / (E * x[3] * x[2] * x[2] * x[2])

        # Constraint functions
        M = P * (L + (x[1] / 2))
        R = np.sqrt((((x[1] * x[1]) / 4.0) + ((x[0] + x[2]) / 2.0) ** 2))
        tmp_var = ((x[1] * x[1]) / 12.0) + (((x[0] + x[2]) / 2.0) ** 2)
        J = 2 * np.sqrt(2) * x[0] * x[1] * tmp_var

        tau_dash_dash = (M * R) / J
        tau_dash = P / (np.sqrt(2) * x[0] * x[1])
        tmp_var = tau_dash * tau_dash + ((2 * tau_dash * tau_dash_dash * x[1]) / (2 * R)) + \
                  (tau_dash_dash * tau_dash_dash)
        tau = np.sqrt(tmp_var)
        sigma = (6 * P * L) / (x[3] * x[2] * x[2])
        tmp_var = 4.013 * E * np.sqrt((x[2] * x[2] * x[3] * x[3] * x[3] * x[3] * x[3] * x[3]) / 36.0) / (L * L)
        tmp_var2 = (x[2] / (2 * L)) * np.sqrt(E / (4 * G))
        PC = tmp_var * (1 - tmp_var2)

        cons = np.zeros(4)
        cons[0] = tauMax - tau
        cons[1] = sigmaMax - sigma
        cons[2] = x[3] - x[0]
        cons[3] = PC - P

        # Calculate the constraint violation values
        cons[cons >= 0.0] = 0.0
        cons[cons < 0] = -cons[cons < 0]

        performance = None

        return obj, cons, performance

