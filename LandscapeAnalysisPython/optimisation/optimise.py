import numpy as np
import copy


def minimise(problem,
             algorithm,
             termination=None,
             **kwargs):

    """
    :param problem: Optimisation problem instance
    :param algorithm: Optimisation algorithm instance
    :param termination: Optimisation termination instance (TODO)
    :param kwargs:
    :return:
    """

    # Set up problem
    if algorithm.problem is None:
        algorithm.setup(problem, **kwargs)

    # Run optimisation
    algorithm.solve()


