import numpy as np


class Setup:

    def __init__(self):

        """
        This case implements the abstract parent class on which any case setup instance can be based
        """

        self.name = ''

    def do(self, prob, **kwargs):

        self.set_variables(prob, **kwargs)
        self.set_constraints(prob, **kwargs)
        self.set_objectives(prob, **kwargs)
        self.set_pareto(prob, **kwargs)

        prob.finalise()

    def set_variables(self, prob, **kwargs):
        pass

    def set_constraints(self, prob, **kwargs):
        pass

    def set_objectives(self, prob, **kwargs):
        pass

    def obj_func(self, x_dict, **kwargs):
        pass

    def set_pareto(self, prob, pareto_set, **kwargs):
        pass



