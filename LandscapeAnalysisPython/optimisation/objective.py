import numpy as np


class Objective(object):

    def __init__(self, name, scale=1.0):

        """
        Objective class
        :param name: Objective name
        :param scale: Objective scale
        """

        self.name = name
        self.scale = scale
        self.value = 0.0
        self.optimum = 0.0


