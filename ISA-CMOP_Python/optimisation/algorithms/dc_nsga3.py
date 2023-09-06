import copy

import numpy as np

from optimisation.algorithms.evolutionary_algorithm import EvolutionaryAlgorithm

from optimisation.operators.sampling.random_sampling import RandomSampling
from optimisation.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
from optimisation.operators.selection.tournament_selection import TournamentSelection, comp_by_cv_then_random
from optimisation.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from optimisation.operators.mutation.polynomial_mutation import PolynomialMutation
from optimisation.operators.survival.reference_direction_survival import ReferenceDirectionSurvival
from optimisation.model.duplicate import DefaultDuplicateElimination

from optimisation.model.population import Population


class DC_NSGA3(EvolutionaryAlgorithm):

    def __init__(self,
                 ref_dirs=None,
                 n_population=100,
                 sampling=LatinHypercubeSampling(),
                 selection=TournamentSelection(comp_func=comp_by_cv_then_random),
                 crossover=SimulatedBinaryCrossover(eta=30, prob=1.0),
                 mutation=PolynomialMutation(eta=20, prob=None),
                 eliminate_duplicates=DefaultDuplicateElimination(),
                 **kwargs):

        self.ref_dirs = ref_dirs

        if 'save_results' in kwargs:
            self.save_results = kwargs['save_results']

        if 'save_name' in kwargs:
            self.save_name = kwargs['save_name']

        # Have to define here given the need to pass ref_dirs
        if 'survival' in kwargs:
            survival = kwargs['survival']
            del kwargs['survival']
        else:
            survival = ReferenceDirectionSurvival(self.ref_dirs, filter_infeasible=False)

        super().__init__(n_population=n_population,
                         sampling=sampling,
                         selection=selection,
                         crossover=crossover,
                         mutation=mutation,
                         survival=survival,
                         eliminate_duplicates=eliminate_duplicates,
                         **kwargs)

    def _initialise(self):
        # Initialise from the evolutionary algorithm class -------------------------------------------------------------
        # Instantiate population
        self.population = Population(self.problem, self.n_population)

        # Population initialisation
        if self.hot_start:
            # Initialise population using hot-start
            self.hot_start_initialisation()
        else:

            # Initialise surrogate modelling strategy
            if self.surrogate is not None:
                self.surrogate.initialise(self.problem, self.sampling)

            # Compute sampling
            self.sampling.do(self.n_population, self.problem.x_lower, self.problem.x_upper) #, seed=self.surrogate.sampling_seed)

            # Assign sampled design variables to population
            self.population.assign_var(self.problem, self.sampling.x)
            if self.x_init:
                self.population[0].set_var(self.problem, self.problem.x_value)
            if self.x_init_additional and self.problem.x_value_additional is not None:
                for i in range(len(self.problem.x_value_additional)):
                    self.population[i+1].set_var(self.problem, self.problem.x_value_additional[i, :])

        # Evaluate initial population
        if self.surrogate is not None:
            self.population = self.evaluator.do(self.surrogate.obj_func, self.problem, self.population)
        else:
            self.population = self.evaluator.do(self.problem.obj_func, self.problem, self.population)

        # Calculate maximum constraint values across the population
        self.max_abs_con_vals = self.evaluator.calc_max_abs_cons(self.population, self.problem)

        # Assign rank and crowding to population
        self.population.assign_rank_and_crowding()

        # --------------------------------------------------------------------------------------------------------------

        # DCNSGA-III params
        self.cp = 5.0
        self.ideal, self.nadir = np.inf, -np.inf

        # Calculate initial epsilon values
        self.eps0 = self.get_max_cons_viol(self.population)
        self.eps = None

    def _next(self):
        # Reduce the dynamic constraint boundary
        self.eps = self.reduce_boundary(self.eps0, self.n_gen, self.max_gen, self.cp)

        # Transform population by epsilon-feasibility
        eps_transformed_pop = self.transform_cons_to_eps_feas(self.population, self.eps)

        # Generate offspring
        self.offspring = self.mating.do(self.problem, eps_transformed_pop, self.n_offspring)

        # Evaluate offspring
        if self.surrogate is not None:
            self.offspring = self.evaluator.do(self.surrogate.obj_func, self.problem, self.offspring, self.max_abs_con_vals)
        else:
            self.offspring = self.evaluator.do(self.problem.obj_func, self.problem, self.offspring, self.max_abs_con_vals)

        # Update normalisation bounds
        merged_feas_population = self.select_merged_eps_feasible_pop(self.offspring, self.population, self.eps)
        self.update_norm_bounds(merged_feas_population)

        # Merge the offspring with the current population
        self.population = Population.merge(self.population, self.offspring)

        # Conduct environmental control
        self.population = self.environmental_control(merged_feas_population, n_survive=self.n_population)

        print(np.min(self.population.extract_cons_sum()))

    def environmental_control(self, population, n_survive):
        # Formulate merged population with additional cv objective and eps-feasibility constraints
        pop = self.transform_population(population)           # Note: This method modifies problem

        # Survive n_population individuals via reference vector selection with eps-feasibility
        if len(pop) > n_survive:
            survivors = self.survival._do(self.problem, pop, n_survive, gen=self.n_gen, max_gen=self.max_gen)
            survived_population = population[survivors]
        else:
            # Append least CV individuals to achieve population size
            cv = self.calc_cv_obj(self.population, self.eps0)
            survivors = np.argpartition(cv, n_survive)[:n_survive]
            survived_population = self.population[survivors]

        # Un-modify problem for original number of objectives
        self.problem.n_obj -= 1

        return survived_population

    def transform_population(self, population):
        vars_arr = population.extract_var()
        obj_arr = population.extract_obj()
        cv_obj = self.calc_cv_obj(population, self.eps0)

        # Create new objectives and transformed constraint
        new_obj_arr = np.hstack((obj_arr, cv_obj[:, None]))
        new_cons_arr = copy.deepcopy(population.extract_cons()) - self.eps
        new_cons_arr[new_cons_arr <= 0.0] = 0.0

        # Modify problem for additional objective
        self.problem.n_obj += 1

        # Create new population
        transformed_pop = Population(self.problem, len(new_obj_arr))
        transformed_pop.assign_var(self.problem, vars_arr)
        transformed_pop.assign_obj(new_obj_arr)
        transformed_pop.assign_cons(new_cons_arr)

        return transformed_pop

    def update_norm_bounds(self, population):
        obj_array = population.extract_obj()

        # Find lower and upper bounds
        f_min = np.min(obj_array, axis=0)
        f_max = np.max(obj_array, axis=0)

        # Update the ideal and nadir points
        # self.ideal = np.minimum(f_min, self.ideal) # TODO: Might conflict with changing eps-feasibility
        self.ideal = f_min
        self.nadir = f_max

    @staticmethod
    def select_merged_eps_feasible_pop(offspring, population, eps):
        # Extract constraints
        offs_cons_arr = copy.deepcopy(offspring.extract_cons())
        pop_cons_arr = copy.deepcopy(population.extract_cons())

        # Select Epsilon- feasible only
        offs_cons_arr[offs_cons_arr <= 0.0] = 0.0
        offs_cv = np.sum(offs_cons_arr - eps, axis=1)
        offs_feas_mask = offs_cv <= 0.0

        pop_cons_arr[pop_cons_arr <= 0.0] = 0.0
        pop_cv = np.sum(pop_cons_arr - eps, axis=1)
        pop_feas_mask = pop_cv <= 0.0

        # Create new merged population
        merged_population = Population.merge(offspring[offs_feas_mask], population[pop_feas_mask])

        if len(merged_population) == 0:
            breakpoint()

        return merged_population

    @staticmethod
    def reduce_boundary(eps0, n_gen, max_gen, cp, delta=1e-8):
        A = eps0 + delta
        base_num = np.log((eps0 + delta) / delta)
        B = max_gen / np.power(base_num, 1 / cp) + 1e-15
        eps = A * np.exp(-(n_gen / B) ** cp) - delta
        eps[eps < 0.0] = 0.0

        return eps

    @staticmethod
    def get_max_cons_viol(population):
        cons_arr = copy.deepcopy(population.extract_cons())
        cons_arr[cons_arr <= 0.0] = 0.0
        max_cons_viol = np.amax(cons_arr, axis=0)
        max_cons_viol[max_cons_viol == 0] = 1.0

        return max_cons_viol

    @staticmethod
    def transform_cons_to_eps_feas(population, eps):
        pop = copy.deepcopy(population)
        transformed_cons = pop.extract_cons() - eps
        transformed_cons[transformed_cons <= 0.0] = 0.0
        pop.assign_cons(transformed_cons)

        return pop

    @staticmethod
    def calc_cv_obj(population, eps0):
        cons_arr = np.atleast_2d(population.extract_cons())
        n_cons = cons_arr.shape[1]
        cv_obj = (1 / n_cons) * np.sum(cons_arr / eps0, axis=1)

        return cv_obj
