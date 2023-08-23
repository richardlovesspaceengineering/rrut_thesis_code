from optimisation.algorithms.evolutionary_algorithm import EvolutionaryAlgorithm

from optimisation.model.population import Population
from optimisation.operators.sampling.random_sampling import RandomSampling
from optimisation.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from optimisation.operators.selection.restricted_selection import RestrictedSelection, comp_by_cv_dom_then_random
from optimisation.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from optimisation.operators.mutation.polynomial_mutation import PolynomialMutation
from optimisation.operators.survival.CADASurvival import CADASurvival
from optimisation.model.duplicate import DefaultDuplicateElimination


class CTAEA(EvolutionaryAlgorithm):

    def __init__(self,
                 ref_dirs,
                 n_population=None,
                 sampling=RandomSampling(),
                 selection=RestrictedSelection(comp_func=comp_by_cv_dom_then_random),
                 crossover=SimulatedBinaryCrossover(n_offspring=1, eta=30, prob=1.0),
                 mutation=PolynomialMutation(eta=20, prob=None),
                 eliminate_duplicates=DefaultDuplicateElimination(),
                 **kwargs):

        self.ref_dirs = ref_dirs
        if n_population is None:
            n_population = len(ref_dirs)

        if 'survival' in kwargs:
            survival = kwargs['survival']
            del kwargs['survival']
        else:
            survival = CADASurvival(ref_dirs)

        # Diversity archives
        self.da = None

        super().__init__(n_population=n_population,
                         sampling=sampling,
                         selection=selection,
                         crossover=crossover,
                         mutation=mutation,
                         survival=survival,
                         n_offspring=n_population,
                         eliminate_duplicates=eliminate_duplicates,
                         **kwargs)

    def _initialise(self):

        # Initialise diversity archives
        da = Population(self.problem)

        # Instantiate population
        self.population = Population(self.problem, self.n_population)

        # Population initialisation
        if self.hot_start:
            # Initialise population using hot-start
            self.hot_start_initialisation()
        else:
            # Compute sampling
            self.sampling.do(self.n_population, self.problem.x_lower, self.problem.x_upper)

            # Assign sampled design variables to population
            self.population.assign_var(self.problem, self.sampling.x)
            if self.x_init:
                self.population[0].set_var(self.problem, self.problem.x_value)
            if self.x_init_additional and self.problem.x_value_additional is not None:
                for i in range(len(self.problem.x_value_additional)):
                    self.population[i+1].set_var(self.problem, self.problem.x_value_additional[i, :])
            # # Assign sampled design variables to population
            # for i in range(self.n_population):
            #     self.population[i].set_var(self.sampling.x[i, :], self.problem)
            # if self.x_init:
            #     self.population[0].set_var(self.problem.x_value, self.problem)

        # Evaluate initial population
        self.population = self.evaluator.do(self.problem.obj_func, self.problem, self.population)

        # Dummy survival call to ensure population is ranked prior to mating selection
        if self.survival:
            population, da = self.survival.do(self.problem, self.population, da, self.n_population)
            self.population = population
            self.da = da

        # Calculate maximum constraint values across the population
        self.max_abs_con_vals = self.evaluator.calc_max_abs_cons(self.population, self.problem)

    def _next(self):

        # Conduct mating using the total population
        hybrid_population = Population.merge(self.population, self.da)
        self.offspring = self.mating.do(self.problem, hybrid_population, self.n_offspring)

        # Evaluate offspring
        self.offspring = self.evaluator.do(self.problem.obj_func,self.problem, self.offspring, self.max_abs_con_vals)

        # Merge the offspring with the current population
        self.population = Population.merge(self.population, self.offspring)

        # Conduct survival selection
        self.population, self.da = self.survival.do(self.problem, self.population, self.da, self.n_population)

        # Update maximum constraint values across the population
        self.max_abs_con_vals = self.evaluator.calc_max_abs_cons(self.population, self.problem)

        # Updating population normalised constraint function values
        if self.problem.n_con > 0:
            self.population = self.evaluator.sum_normalised_cons(self.population, self.problem, max_abs_con_vals=self.max_abs_con_vals)

