from optimisation.algorithms.evolutionary_algorithm import EvolutionaryAlgorithm

from optimisation.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from optimisation.operators.selection.tournament_selection import TournamentSelection, binary_tournament
from optimisation.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from optimisation.operators.mutation.polynomial_mutation import PolynomialMutation
from optimisation.operators.survival.rank_and_crowding_survival import RankAndCrowdingSurvival
from optimisation.model.duplicate import DefaultDuplicateElimination


class NSGA2(EvolutionaryAlgorithm):

    def __init__(self,
                 n_population=100,
                 sampling=LatinHypercubeSampling(),
                 selection=TournamentSelection(comp_func=binary_tournament),
                 crossover=SimulatedBinaryCrossover(eta=15, prob=0.9),
                 mutation=PolynomialMutation(eta=20, prob=None),
                 survival=RankAndCrowdingSurvival(filter_infeasible=True),
                 eliminate_duplicates=DefaultDuplicateElimination(),
                 **kwargs):

        super().__init__(n_population=n_population,
                         sampling=sampling,
                         selection=selection,
                         crossover=crossover,
                         mutation=mutation,
                         survival=survival,
                         eliminate_duplicates=eliminate_duplicates,
                         **kwargs)
