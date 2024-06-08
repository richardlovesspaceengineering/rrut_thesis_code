import numpy as np

from optimisation.setup import Setup


class CRE21Setup(Setup):
    """ This is the two bar truss design problem
    Reference:
    C. A. C. Coello and G. T. Pulido, "Multiobjective structural optimization using a microgenetic algorithm,"
    Stru. and Multi. Opt., vol. 30, no. 5, pp. 388-403, 2005.
    Implementation based on Tanabe 2020
    Ryoji Tanabe, Hisao Ishibuchi, "An Easy-to-use Real-world Multi-objective Problem Suite"
    """

    def __init__(self):
        super().__init__()

    def set_variables(self, prob, **kwargs):
        prob.add_var_group('x_vars', 3, 'c', lower=[0.00001, 0.00001, 1.0], upper=[100.0, 100.0, 3.0], value=[50.0, 50.0, 2.0], scale=1.0)

    def set_constraints(self, prob, **kwargs):
        prob.add_con_group('con', 3, lower=None, upper=0.0)

    def set_objectives(self, prob, **kwargs):
        prob.add_obj('f_1')
        prob.add_obj('f_2')

    def obj_func(self, x_dict, **kwargs):

        x = x_dict['x_vars']

        # objective functions
        obj = np.zeros(2)
        obj[0] = x[0] * np.sqrt(16.0 + (x[2] * x[2])) + x[1] * np.sqrt(1.0 + x[2] * x[2])
        obj[1] = (20.0 * np.sqrt(16.0 + (x[2] * x[2]))) / (x[0] * x[2])

        # Constraint functions
        cons = np.zeros(3)
        cons[0] = 0.1 - obj[0]
        cons[1] = 100000.0 - obj[1]
        cons[2] = 100000 - ((80.0 * np.sqrt(1.0 + x[2] * x[2])) / (x[2] * x[1]))

        # Calculate the constraint violation values
        cons[cons >= 0.0] = 0.0
        cons[cons < 0] = -cons[cons < 0]

        performance = None

        return obj, cons, performance

