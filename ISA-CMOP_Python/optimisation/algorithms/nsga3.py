from optimisation.algorithms.evolutionary_algorithm import EvolutionaryAlgorithm

from optimisation.operators.sampling.random_sampling import RandomSampling
from optimisation.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from optimisation.operators.selection.tournament_selection import TournamentSelection, comp_by_cv_then_random
from optimisation.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from optimisation.operators.mutation.polynomial_mutation import PolynomialMutation
from optimisation.operators.survival.reference_direction_survival import ReferenceDirectionSurvival
from optimisation.model.duplicate import DefaultDuplicateElimination


class NSGA3(EvolutionaryAlgorithm):

    def __init__(self,
                 ref_dirs=None,
                 n_population=100,
                 sampling=RandomSampling(),
                 selection=TournamentSelection(comp_func=comp_by_cv_then_random),
                 crossover=SimulatedBinaryCrossover(eta=30, prob=1.0),
                 mutation=PolynomialMutation(eta=20, prob=None),
                 eliminate_duplicates=DefaultDuplicateElimination(),
                 **kwargs):

        self.ref_dirs = ref_dirs

        # Have to define here given the need to pass ref_dirs
        if 'survival' in kwargs:
            survival = kwargs['survival']
            del kwargs['survival']
        else:
            survival = ReferenceDirectionSurvival(self.ref_dirs, filter_infeasible=True)

        super().__init__(n_population=n_population,
                         sampling=sampling,
                         selection=selection,
                         crossover=crossover,
                         mutation=mutation,
                         survival=survival,
                         eliminate_duplicates=eliminate_duplicates,
                         **kwargs)

