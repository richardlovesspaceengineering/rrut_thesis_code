from optimisation.operators.decomposition.chebychev import Chebychev
from optimisation.operators.decomposition.pbi import PBI
from optimisation.operators.decomposition.asf import ASF


def get_decomposition(name, **kwargs):

    if name == 'chebychev':
        decomposition = Chebychev(**kwargs)
    elif name == 'pbi':
        decomposition = PBI(**kwargs)
    elif name == 'asf':
        decomposition = ASF(**kwargs)
    else:
        decomposition = None

    return decomposition
