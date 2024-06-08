import numpy as np

inf = 1e20


class Constraint(object):

    def __init__(self, name, n_con, lower, upper):

        """
        Constraint class
        :param name: Constraint name
        :param n_con: Number of constraints
        :param lower: Lower bound
        :param upper: Upper bound
        """

        self.name = name
        self.n_con = n_con

        # Process the lower argument
        if lower is None:
            lower = [None for i in range(self.n_con)]
        elif np.isscalar(lower):
            lower = lower*np.ones(self.n_con)
        elif len(lower) == self.n_con:
            pass  # Some iterable object
        else:
            raise Exception('The "lower" argument to addCon or addConGroup is '
                            'invalid. It must be None, a scalar, or a '
                            'list/array or length ncon=%d.' % n_con)

        # Process the upper argument
        if upper is None:
            upper = [None for i in range(self.n_con)]
        elif np.isscalar(upper):
            upper = upper*np.ones(self.n_con)
        elif len(upper) == self.n_con:
            pass  # Some iterable object
        else:
            raise Exception('The "upper" argument to addCon or addConGroup is '
                            'invalid. It must be None, a scalar, or a '
                            'list/array or length ncon=%d.' % n_con)

        # Setting parameters
        self.lower = lower
        self.upper = upper
        # The current value of the constraint
        self.value = np.zeros(self.n_con)

