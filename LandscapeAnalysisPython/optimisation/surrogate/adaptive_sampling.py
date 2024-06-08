import numpy as np

from collections import OrderedDict
from mpi4py import MPI

from optimisation.setup import Setup
from optimisation.model.problem import Problem
from optimisation.algorithms.nsga2 import NSGA2
from optimisation.optimise import minimise

from optimisation.operators.survival.rank_and_crowding_survival import RankAndCrowdingSurvival
from optimisation.operators.survival.rank_and_hypervolume_survival import RankAndHypervolumeSurvival


class AdaptiveSampling(object):

    def __init__(self, n_population=100, n_gen=25, acquisition_criteria='idw', **kwargs):

        self.n_population = n_population
        self.n_gen = n_gen

        self.acquisition_criteria = acquisition_criteria

    def generate_evals(self, models, n_pts, alpha=0.5, parent_prob=None, cons_models=None, use_constraints=False, n_processors=1):

        """ Use NSGA-II without constraints compute a Pareto front of LCB across each objective (constraints are ignored
        if use_constraints=False)
        NOTE: Constraints are ignored because to improve the fit of each surrogate, infeasible regions of those
        surrogates may need more data to improve the fit overall """

         # Setup instance
        setup = AdaptiveSamplingSetup(models=models, alpha=self.alpha, cons_models=cons_models,
                                      use_constraints=use_constraints, weight=weight)

        # Problem instance
        if self.acquisition_criteria == 'lcb':
            opt_prob = Problem('adaptive_sampling', obj_func=setup.lcb_func, map_internally=False,
                               n_processors=n_processors)
        elif self.acquisition_criteria == 'normalised_lcb':
            opt_prob = Problem('adaptive_sampling', obj_func=setup.normalised_lcb_func, map_internally=False,
                               n_processors=n_processors)
        elif self.acquisition_criteria == 'idw':
            opt_prob = Problem('adaptive_sampling', obj_func=setup.idw_func, map_internally=False,
                               n_processors=n_processors)
        elif self.acquisition_criteria == 'ei':
            opt_prob = Problem('adaptive_sampling', obj_func=setup.ei_func, map_internally=False,
                               n_processors=1)
        elif self.acquisition_criteria == 'mwee':
            opt_prob = Problem('adaptive_sampling', obj_func=setup.wee_func, map_internally=False,
                               n_processors=1)
        elif self.acquisition_criteria == 'wb2s':
            opt_prob = Problem('adaptive_sampling', obj_func=setup.wb2s_func, map_internally=False,
                               n_processors=1)
        elif self.acquisition_criteria == 'wei':
            opt_prob = Problem('adaptive_sampling', obj_func=setup.wei_func, map_internally=False,
                               n_processors=1)
        else:
            opt_prob = None
            Exception('Please specify valid acquisition criterion')

        # Set variables, constraints & objectives
        setup.do(opt_prob, parent_prob=parent_prob)

        # NSGA2 algorithm instance
        algorithm = NSGA2(n_population=self.n_population,
                          max_gen=self.n_gen,
                          survival=RankAndCrowdingSurvival(),
                          # survival=RankAndHypervolumeSurvival(),
                          plot=False,
                          print=False)

        # Run optimisation
        minimise(opt_prob,
                 algorithm,
                 seed=None,
                 hot_start=False,
                 x_init=False,
                 save_history=False)
        MPI.COMM_WORLD.barrier()

        # Extract non-dominated individuals from final population
        final_pop = algorithm.population
        nd_pop = final_pop[final_pop.extract_rank() == 0]

        # Todo: Implement selection if non-dominated front is not of required size

        # Select individuals from non-dominated population to evaluate as infill points
        selected = nd_pop[:n_pts]

        # Return variables
        return np.atleast_2d(selected.extract_var())


class AdaptiveSamplingSetup(Setup):

    def __init__(self, models, alpha=0.5, delta=2, ksi=0.01, weight=0, wei_weight=0.186,  use_iqr=False, cons_models=None, use_constraints=False):
        super().__init__()
        self.models = models
        self.cons_models = cons_models

        self.alpha = alpha
        self.delta = delta
        self.ksi = ksi
        self.weight = weight
        self.wei_weight = wei_weight

        self.use_iqr = use_iqr
        self.use_constraints = use_constraints

    def set_variables(self, prob, parent_prob=None, **kwargs):

        # Copy variables from parent problem
        var_dict = extract_var_groups(parent_prob.variables)
        for i, key in enumerate(var_dict):
            prob.add_var_group(key, len(var_dict[key][0]), parent_prob.variables[key][0].type,
                               lower=var_dict[key][1], upper=var_dict[key][2],
                               value=var_dict[key][0], scale=parent_prob.variables[key][0].scale)

    def set_constraints(self, prob, parent_prob=None, **kwargs):

        if self.use_constraints:
            # Copy constraints from parent problem
            for key in parent_prob.constraints:
                prob.add_con(key)
        else:
            pass

    def set_objectives(self, prob, parent_prob=None, **kwargs):

        # Copy objectives from parent problem
        for key in parent_prob.objectives:
            prob.add_obj(key)

    def lcb_func(self, x_dict, **kwargs):

        # Form design vector from input dict
        x = x_dict['x_vars']
        x = x.astype(np.float32)

        # Calculating objective function values
        obj = np.zeros(len(self.models))
        for i, model in enumerate(self.models):
            obj[i] = model.predict_lcb(x, alpha=self.alpha, use_iqr=self.use_iqr)

        if self.cons_models is not None and self.use_constraints:
            cons = np.ones(len(self.cons_models))
            for i, model in enumerate(self.cons_models):
                cons[i] = model.predict(x)
        else:
            cons = None

        performance = None

        return obj, cons, performance

    def idw_func(self, x_dict, **kwargs):

        # Form design vector from input dict
        x = x_dict['x_vars']
        x = x.astype(np.float32)

        # Calculating objective function values
        obj = np.zeros(len(self.models))
        for i, model in enumerate(self.models):
            obj[i] = model.predict_idw(x, delta=self.delta)

        if self.cons_models is not None and self.use_constraints:
            cons = np.ones(len(self.cons_models))
            for i, model in enumerate(self.cons_models):
                cons[i] = model.predict(x)
        else:
            cons = None

        performance = None

        return obj, cons, performance

    def ei_func(self, x_dict, **kwargs):

        # Form design vector from input dict
        x = x_dict['x_vars']
        x = x.astype(np.float32)

        # Calculating objective function values
        obj = np.zeros(len(self.models))
        for i, model in enumerate(self.models):
            obj[i] = model.predict_ei(x, ksi=self.ksi)

        if self.cons_models is not None and self.use_constraints:
            cons = np.ones(len(self.cons_models))
            for i, model in enumerate(self.cons_models):
                cons[i] = model.predict(x)
        else:
            cons = None

        performance = None

        return obj, cons, performance

    def normalised_lcb_func(self, x_dict):

        # Form design vector from input dict
        x = x_dict['x_vars']
        x = x.astype(np.float32)

        # Calculating objective function values
        obj = np.zeros(len(self.models))
        for i, model in enumerate(self.models):
            obj[i] = model.predict_normalised_lcb(x, alpha=self.alpha)

        if self.cons_models is not None and self.use_constraints:
            cons = np.ones(len(self.cons_models))
            for i, model in enumerate(self.cons_models):
                cons[i] = model.predict(x)
        else:
            cons = None

        performance = None

        return obj, cons, performance

    def wee_func(self, x_dict):
        # Form design vector from input dict
        x = x_dict['x_vars']
        x = x.astype(np.float32)

        # Calculating objective function values
        obj = np.zeros(len(self.models))
        for i, model in enumerate(self.models):
            obj[i] = model.predict_wee(x, weight=self.weight)

        if self.cons_models is not None and self.use_constraints:
            cons = np.ones(len(self.cons_models))
            for i, model in enumerate(self.cons_models):
                cons[i] = model.predict(x)
        else:
            cons = None

        performance = None

        return obj, cons, performance

    def wb2s_func(self, x_dict):
        # Form design vector from input dict
        x = x_dict['x_vars']
        x = x.astype(np.float32)

        # Calculating objective function values
        obj = np.zeros(len(self.models))
        for i, model in enumerate(self.models):
            obj[i] = model.predict_wb2s(x, weight=self.scale)

        if self.cons_models is not None and self.use_constraints:
            cons = np.ones(len(self.cons_models))
            for i, model in enumerate(self.cons_models):
                cons[i] = model.predict(x)
        else:
            cons = None

        performance = None

        return obj, cons, performance



def extract_var_groups(vars):

    var_dict = OrderedDict()

    for i, key in enumerate(vars.keys()):

        # Extract variable values
        var_arr = np.zeros(len(vars[key]))
        lower_arr = np.zeros(len(vars[key]))
        upper_arr = np.zeros(len(vars[key]))
        for j in range(len(vars[key])):
            var_arr[j] = vars[key][j].value
            lower_arr[j] = vars[key][j].lower
            upper_arr[j] = vars[key][j].upper

        # Add variable to dict
        var_dict[key] = (var_arr, lower_arr, upper_arr)

        # De-scale variables
        if vars[key][0].type == 'c':
            var_dict[key] = var_dict[key]/vars[key][0].scale
        elif vars[key][0].type == 'i':
            var_dict[key] = round(var_dict[key]/vars[key][0].scale)
        elif vars[key][0].type == 'd':
            idx = np.round(var_dict[key]/vars[key][0].scale, 0).astype(int)
            var_dict[key] = np.asarray(vars[key][0].choices)[idx].tolist()

    return var_dict
