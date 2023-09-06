import copy

import numpy as np
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, STATUS_FAIL

from optimisation.surrogate.models.rbf import RBF
from sklearn.svm import SVR
from smt.surrogate_models import KPLS
from optimisation.operators.survival.rank_and_crowding_survival import RankAndCrowdingSurvival

from optimisation.surrogate.models.rbf import RadialBasisFunctions
from optimisation.surrogate.models.svr import SupportVectorRegression
from optimisation.surrogate.models.KPLS import KPLSRegression


class SurrogateTuner(object):
    """
    Conducts hyperopt tuning of the width hyperparameter in a RBF surrogate
    """
    def __init__(self,
                 n_dim,
                 lb,
                 ub,
                 problem,
                 c=0.5,
                 p_type='linear',
                 kernel_type='gaussian',
                 method='mse',
                 width_range=(0.001, 100),
                 max_evals=100,
                 verbose=False,
                 attempts=5,
                 train_test_split=0.15):

        # Store inputs internally
        self.n_dim = n_dim
        self.lb = lb
        self.ub = ub
        self.problem = problem
        self.c = c
        self.p_type = p_type
        self.kernel_type = kernel_type
        self.method = method
        self.width_range = width_range
        self.max_evals = max_evals
        self.verbose = verbose
        self.attempts = attempts
        self.train_test_split = train_test_split
        self.w_min = self.width_range[0]
        self.w_max = self.width_range[1]
        self.kernel_list = ['gaussian', 'multiquadratic', 'inv_multiquadratic', 'tps',
                            'matern_52', 'matern_32', 'matern_12', 'exp', 'logistic', 'cauchy', 'mod_cubic']
        self.svr_kernel_list = ['linear', 'sigmoid', 'poly', 'rbf']

        # Pre-allocate variables
        self.obj_cntr = None
        self.models = None
        self.evaluator = None
        self.model = None
        self.x_total = None
        self.y_total = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def do(self, problem, population, n_obj, fronts):
        population = copy.deepcopy(population)

        # # Split up population (test is first front)
        # test_split = fronts[0]
        # train_split = fronts[1]
        # for i in range(2, len(fronts)):
        #     train_split = np.hstack((train_split, fronts[i]))
        # train_pop = population[train_split]
        # test_pop = population[test_split]

        # Extract n_test and n_train individuals via RankandCrowding selection
        n_survive = int(len(population) * self.train_test_split)
        test_pop = RankAndCrowdingSurvival().do(problem, population, n_survive, None, None)
        matching_idx = np.isin(population.extract_obj(), test_pop.extract_obj())
        train_indices = ~(matching_idx[:, 0] == True)
        train_pop = population[train_indices]
        # print('n_train: ', len(train_pop), 'n_test: ', len(test_pop))

        # Extract total set from population
        self.x_total = np.atleast_2d(population.extract_var())
        # self.x_total = self._normalise(self.x_total)
        self.y_total = np.atleast_2d(population.extract_obj())

        # Extract test set from population
        self.x_train = np.atleast_2d(train_pop.extract_var())
        self.x_train = self._normalise(self.x_train)
        self.y_train = np.atleast_2d(train_pop.extract_obj())

        # Extract test set from population
        self.x_test = np.atleast_2d(test_pop.extract_var())
        self.x_test = self._normalise(self.x_test)
        self.y_test = np.atleast_2d(test_pop.extract_obj())

        # Determine evaluation metric
        if 'mse' in self.method:
            evaluator = self.calc_mse
        elif 'sep' in self.method:
            evaluator = self.calc_sep
        else:
            raise Exception(f"The provided metric {self.method} is not recognised!")
        self.evaluator = evaluator

        # Optimise width hyperparameter of each model on test population
        surrogates = []
        for obj_cntr in range(n_obj):
            # Conduct hyperopt routine
            model = self._do(obj_cntr)
            surrogates.append(model)

        return surrogates

    def _do(self, obj_cntr):
        self.obj_cntr = obj_cntr

        # Search space
        # space = {'c': hp.uniform('c', self.w_min, self.w_max)}
        # space = {'c': hp.uniform('c', self.w_min, self.w_max),
        #          'kernel': hp.choice('kernel', self.kernel_list)}
        space = hp.choice('regressors', [
            {
                'model': 'rbf',
                'params': {
                    'c': hp.uniform('c', self.w_min, self.w_max),
                    'kernel': hp.choice('kernel', self.kernel_list)
                }
            },
            {
                'model': 'kpls',
                'params': {
                    'theta0': hp.uniform('theta0', 1e-4, 20)
                }
            },
            {
                'model': 'svr',
                'params': {
                    'gamma': hp.uniform('gamma', 0, 20),
                    'kern': hp.choice('kern', ['linear', 'sigmoid', 'poly', 'rbf']),
                    'epsilon': hp.uniform('epsilon', 0.0, 1.0)
                }
            }
        ])

        # Hyperopt optimiser
        trials = Trials()
        opt_params = fmin(fn=self.objective,
                          space=space,
                          algo=tpe.suggest,
                          max_evals=self.max_evals,
                          trials=trials,
                          verbose=self.verbose)
        
        # Construct optimised model with full population
        model_choice = opt_params['regressors']
        if model_choice == 0:
            opt_model = RadialBasisFunctions(self.n_dim, l_b=self.problem.x_lower, u_b=self.problem.x_upper,
                                             c=opt_params['c'], p_type=self.p_type,
                                             kernel_type=self.kernel_list[opt_params['kernel']])
            opt_model.add_points(self.x_total, self.y_total[:, self.obj_cntr].flatten())
            opt_model.train()
            print(f"Opt RBF: {obj_cntr}, C: {opt_params['c']}, Kernel: {self.kernel_list[opt_params['kernel']]}")

        elif model_choice == 1:
            opt_model = KPLSRegression(self.n_dim, l_b=self.problem.x_lower, u_b=self.problem.x_upper,
                                       theta0=opt_params['theta0'])
            opt_model.add_points(self.x_total, self.y_total[:, self.obj_cntr].flatten())
            opt_model.train()
            print(f"Opt KPLS: {obj_cntr}, Theta0: {opt_params['theta0']}")

        elif model_choice == 2:
            opt_model = SupportVectorRegression(self.n_dim, l_b=self.problem.x_lower, u_b=self.problem.x_upper,
                                                c=opt_params['gamma'], epsilon=opt_params['epsilon'],
                                                kernel=self.svr_kernel_list[opt_params['kern']])
            opt_model.add_points(self.x_total, self.y_total[:, self.obj_cntr].flatten())
            opt_model.train()
            print(f"Opt SVR: {obj_cntr}, C: {opt_params['gamma']}, Kernel: {self.kernel_list[opt_params['kern']]}, "
                  f" Epsilon: {opt_params['epsilon']}")

        return opt_model

    def objective(self, space):
        # Build Model
        # model = RBF(n_dim=self.n_dim, c=space['c'], p_type=self.p_type, kernel_type=self.kernel_type)
        # model = RBF(n_dim=self.n_dim, c=space['c'], p_type=self.p_type, kernel_type=space['kernel'])
        model_type = space['model']
        if model_type in 'rbf':
            model = RBF(n_dim=self.n_dim, c=space['params']['c'],
                        p_type=self.p_type, kernel_type=space['params']['kernel'])
        elif model_type in 'kpls':
            model = KPLS(theta0=space['params']['theta0'] * np.ones(1),
                         corr="squar_exp",
                         poly="linear",
                         n_start=5,
                         print_global=False)
        elif model_type in 'svr':
            model = SVR(kernel=space['params']['kern'],
                        C=space['params']['gamma'],
                        epsilon=space['params']['epsilon'])

        try:
            if model_type in 'kpls':
                model.set_training_values(self.x_train, self.y_train[:, self.obj_cntr].flatten())
                model.train()
            else:
                model.fit(self.x_train, self.y_train[:, self.obj_cntr].flatten())

            # Calculate accuracy metric (objective)
            accuracy = self.evaluator(model, self.obj_cntr, model_type)

            # Assign good status
            status = STATUS_OK
        except np.linalg.LinAlgError:
            # print(f"RBF with c={space['c']}: Singular Matrix")
            status = STATUS_FAIL
            accuracy = np.inf

        # print(f"obj: {self.obj_cntr}, acc: {accuracy}, c: {space['c']} kernel: {space['kernel']}")
        # print(f"{self.obj_cntr},{accuracy[0]},{space['c']}")
        return {'loss': accuracy, 'status': status}

    def calc_mse(self, model, obj_cntr, model_type):
        # Extract test population size
        n_pop = len(self.y_test)

        # Iterate through individuals and calculate model prediction error
        accuracy = 0
        for idx in range(n_pop):
            if model_type in 'rbf':
                y_pred = model.predict(self.x_test[idx])
            elif model_type in 'svr':
                y_pred = model.predict(np.atleast_2d(self.x_test[idx]))
            elif model_type in 'kpls':
                y_pred = model.predict_values(np.atleast_2d(self.x_test[idx]))

            # print(idx, y_pred, self.y_test[idx, obj_cntr])
            accuracy += (self.y_test[idx, obj_cntr] - y_pred) ** 2    # MSE Calculation
        accuracy = (1 / n_pop) * accuracy                             # MSE Normalisation

        return accuracy

    def calc_sep(self, model, obj_cntr):
        # Extract test population size
        n_pop = len(self.y_test)

        # Iterate through individuals and calculate model sep error
        accuracy = 0

        # Predict surrogate function values
        pred_values = np.zeros(n_pop)
        for idx in range(n_pop):
            pred_values[idx] = model.predict(self.x_test[idx])

        # Evaluate the SEP
        for i in range(n_pop):
            for j in range(i + 1, n_pop):
                accuracy += self._pairwise_comp(self.y_test[i, obj_cntr], self.y_test[j, obj_cntr],
                                                pred_values[i], pred_values[j])

        # Normalise SEP
        accuracy *= 1 / (0.5 * n_pop * (n_pop - 1))

        return accuracy

    def _pairwise_comp(self, f_real_i, f_real_j, f_pred_i, f_pred_j):
        """
        Calculates the pairwise Selection Error Probability (SEP) metric (Ahrari2019)
        :param f_real_i: f(x_i)   : Real Function
        :param f_real_j: f(x_j)   : Real Function
        :param f_pred_i: f(x_i)   : Surrogate Function
        :param f_pred_j: f(x_j)   : Surrogate Function
        :return: q(x_i, x_j)      : pairwise match=0, dismatch=1
        """
        if (f_real_i - f_real_j)*(f_pred_i - f_pred_j) < 0:
            return 1
        else:
            return 0

    def _normalise(self, x):
        # Scale training data by variable bounds
        _x = (x - self.lb) / (self.ub - self.lb)
        return _x
