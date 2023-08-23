import numpy as np


class Selection:

    def __init__(self):
        super().__init__()

    def do(self, pop, n_select, n_parents=2, **kwargs):

        return self._do(pop, n_select, n_parents, **kwargs)

    def _do(self, pop, n_select, n_parents, **kwargs):
        pass

