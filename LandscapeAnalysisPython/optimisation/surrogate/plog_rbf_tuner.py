import copy

import numpy as np
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, STATUS_FAIL

from optimisation.surrogate.models.rbf import RBF
from optimisation.operators.survival.rank_and_crowding_survival import RankAndCrowdingSurvival

from optimisation.util.misc import sp2log_transform

from sklearn.model_selection import StratifiedKFold


class RBFTuner(object):
    """
    Conducts hyperopt tuning of the width hyperparameter in a RBF surrogate
    """
    def __init__(self,
                 n_dim,
                 lb,
                 ub,
                 c=0.5,
                 p_type='linear',
                 kernel_type='gaussian',
                 method='mse',
                 width_range=(0.01, 10),
                 k1_range=(-2.0, 2.0),
                 k2_range=(0.1, 1e3),
                 max_evals=100,
                 verbose=False,
                 attempts=5,
                 train_test_split=0.15):

        # Store inputs internally
        self.n_dim = n_dim
        self.lb = lb
        self.ub = ub
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
        self.k1_min = k1_range[0]
        self.k1_max = k1_range[1]
        self.k2_min = k2_range[0]
        self.k2_max = k2_range[1]
        self.kernel_list = ['gaussian', 'multiquadratic', 'inv_multiquadratic', 'tps']
        self.survival = RankAndCrowdingSurvival()

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
        self.f_min = None
        self.f_max = None
        self.cv_training_indices = None
        self.n_pts = None
        self.cv_k = 5

    def do(self, problem, population, n_obj):
        population = copy.deepcopy(population)
        self.f_min, self.f_max = self._get_ranges(population.extract_obj())

        # Extract n_test and n_train individuals via RankandCrowding selection
        # n_survive = int(len(population) * self.train_test_split)
        # test_indices = self.survival._do(problem, population, n_survive)
        # train_indices = np.arange(len(population))[~np.isin(np.arange(len(population)), test_indices)]
        # train_pop = population[train_indices]
        # test_pop = population[test_indices]
        # print('n_train: ', len(train_pop), 'n_test: ', len(test_pop))

        # Extract total set from population
        self.x_total = np.atleast_2d(population.extract_var())
        self.x_total = self._normalise(self.x_total)
        self.y_total = np.atleast_2d(population.extract_obj())

        # # Extract test set from population
        # self.x_train = np.atleast_2d(train_pop.extract_var())
        # self.x_train = self._normalise(self.x_train)
        # self.y_train = np.atleast_2d(train_pop.extract_obj())
        #
        # # Extract test set from population
        # self.x_test = np.atleast_2d(test_pop.extract_var())
        # self.x_test = self._normalise(self.x_test)
        # self.y_test = np.atleast_2d(test_pop.extract_obj())

        # Cross-validation
        random_state = np.random.default_rng()
        self.n_pts = len(self.x_total)
        self.cv_training_indices = np.array_split(random_state.choice(self.n_pts, size=self.n_pts, replace=False), self.cv_k)

        # Determine evaluation metric
        if 'mse' in self.method:
            evaluator = self.calc_mse
        elif 'sep' in self.method:
            evaluator = self.calc_sep
        else:
            raise Exception(f"The provided metric {self.method} is not recognised!")
        self.evaluator = evaluator

        # Optimise width hyperparameter of each model on test population
        opt_models = []
        opt_params = []
        for obj_cntr in range(n_obj):
            # Conduct hyperopt routine
            model, hyperparams = self._do(obj_cntr)

            # Store optimised model and hyperparameters
            opt_models.append(model)
            opt_params.append(hyperparams)

        return opt_models, opt_params

    def _do(self, obj_cntr):
        self.obj_cntr = obj_cntr

        # Search space
        # space = {'c': hp.uniform('c', self.w_min, self.w_max)}
        space = {'c': hp.uniform('c', self.w_min, self.w_max),
                 'kernel': hp.choice('kernel', self.kernel_list),
                 'k1': hp.uniform('k1', self.k1_min, self.k1_max),
                 'k2': hp.uniform('k2', self.k2_min, self.k2_max)}

        # Hyperopt optimiser
        trials = Trials()
        opt_params = fmin(fn=self.objective,
                          space=space,
                          algo=tpe.suggest,
                          max_evals=self.max_evals,
                          trials=trials,
                          verbose=self.verbose)
        
        # Construct optimised model with full population
        for i in range(self.attempts):
            try:
                # opt_model = RBF(n_dim=self.n_dim, c=opt_params['c'], p_type=self.p_type, kernel_type=self.kernel_type)
                opt_model = RBF(n_dim=self.n_dim, c=opt_params['c'], p_type=self.p_type, kernel_type=self.kernel_list[opt_params['kernel']])
                opt_model.fit(self.x_total, self.y_total[:, self.obj_cntr].flatten())
                break
            except np.linalg.LinAlgError:
                opt_model = None
                opt_params['c'] = 1.05 * opt_params['c']
                continue

        if opt_model is None:
            opt_model = RBF(n_dim=self.n_dim, c=self.c, p_type=self.p_type, kernel_type=self.kernel_type)
            opt_model.fit(self.x_total, self.y_total[:, self.obj_cntr])
            opt_params['k1'] = 0.0
            opt_params['k2'] = 1.0
        
        # if self.verbose:
        print(f"Opt RBF: {obj_cntr}, C: {opt_params['c']}, Kernel: {self.kernel_list[opt_params['kernel']]}, "
              f"plog-k1: {opt_params['k1']}, plog-k2: {opt_params['k2']}")

        return opt_model, (opt_params['c'], opt_params['k1'], opt_params['k2'])

    def objective(self, space):
        # Build Model
        # model = RBF(n_dim=self.n_dim, c=space['c'], p_type=self.p_type, kernel_type=self.kernel_type)
        # model = RBF(n_dim=self.n_dim, c=space['c'], p_type=self.p_type, kernel_type=space['kernel'])
        cv_models = [RBF(n_dim=self.n_dim, c=space['c'], p_type=self.p_type, kernel_type=space['kernel'])
                     for _ in range(self.cv_k)]

        # Scale and transform y data with sp2log
        y_train = self._standard_scaler(self.y_total)
        _y_train = sp2log_transform(y_train, k1=space['k1'], k2=space['k2'])

        try:
            scores = []
            for i, model in enumerate(cv_models):
                model.fit(self.x_total[~np.in1d(np.arange(self.n_pts), self.cv_training_indices[i]), :],
                          _y_train[~np.in1d(np.arange(self.n_pts), self.cv_training_indices[i]), self.obj_cntr])
                # model.fit(self.x_train, _y_train[:, self.obj_cntr].flatten())

                # Calculate accuracy metric (objective)
                accuracy = self.evaluator(model, self.obj_cntr, space, i=i)
                scores.append(accuracy)

            # Assign good status
            status = STATUS_OK
        except np.linalg.LinAlgError:
            # print(f"RBF with c={space['c']}: Singular Matrix")
            status = STATUS_FAIL
            scores = [np.inf]

        # print(f"obj: {self.obj_cntr}, acc: {accuracy}, c: {space['c']} kernel: {space['kernel']}")
        # print(f"{self.obj_cntr},{accuracy[0]},{space['c']}")
        return {'loss': np.mean(scores), 'status': status}

    def calc_mse(self, model, obj_cntr, space, i):
        # Extract test population size
        x_test = self.x_total[np.in1d(np.arange(self.n_pts), self.cv_training_indices[i]), :]
        y_test = self.y_total[np.in1d(np.arange(self.n_pts), self.cv_training_indices[i]), :]
        n_pop = len(y_test)

        # Scale and transform y data with sp2log
        y_test = self._standard_scaler(y_test)
        _y_test = sp2log_transform(y_test, k1=space['k1'], k2=space['k2'])

        # Iterate through individuals and calculate model prediction error
        accuracy = 0
        for idx in range(n_pop):
            y_pred = model.predict(x_test[idx])
            # print(idx, y_pred, self.y_test[idx, obj_cntr])
            accuracy += (_y_test[idx, obj_cntr] - y_pred) ** 2    # MSE Calculation
        accuracy = (1 / n_pop) * accuracy                             # MSE Normalisation

        return accuracy

    def calc_sep(self, model, obj_cntr, space):
        # Extract test population size
        n_pop = len(self.y_test)

        # Iterate through individuals and calculate model sep error
        accuracy = 0

        # Predict surrogate function values
        pred_values = np.zeros(n_pop)
        for idx in range(n_pop):
            pred_values[idx] = model.predict(self.x_test[idx])

        # Scale and transform y data with sp2log
        y_test = self._standard_scaler(self.y_test)
        _y_test = sp2log_transform(y_test, k1=space['k1'], k2=space['k2'])

        # Evaluate the SEP
        for i in range(n_pop):
            for j in range(i + 1, n_pop):
                accuracy += self._pairwise_comp(_y_test[i, obj_cntr], _y_test[j, obj_cntr],
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

    def _get_ranges(self, obj):
        f_min = np.min(obj, axis=0)
        f_max = np.max(obj, axis=0)

        return f_min, f_max

    def _standard_scaler(self, obj):
        if self.f_min is None or self.f_max is None:
            self._get_ranges(obj)
        _obj = (2*obj - (self.f_max + self.f_min)) / (self.f_max - self.f_min)

        return _obj



