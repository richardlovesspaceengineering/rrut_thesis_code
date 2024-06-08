import copy

import numpy as np
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, STATUS_FAIL

from optimisation.surrogate.models.rbf import RBF

from optimisation.util.misc import bilog_transform, reverse_bilog_transform

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
        self.kernel_list = ['gaussian', 'multiquadratic', 'inv_multiquadratic', 'tps', 'cubic']
        self.transform_list = ['none', 'bilog']

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

    def do_obj(self, problem, population, train_population, test_population):
        # Extract total set from population
        self.x_total = np.atleast_2d(population.extract_var())
        self.x_total = self._normalise(self.x_total)
        self.y_total = np.atleast_2d(population.extract_obj())

        # Extract test set from population
        self.x_train = np.atleast_2d(train_population.extract_var())
        self.x_train = self._normalise(self.x_train)
        self.y_train = np.atleast_2d(train_population.extract_obj())

        # Extract test set from population
        self.x_test = np.atleast_2d(test_population.extract_var())
        self.x_test = self._normalise(self.x_test)
        self.y_test = np.atleast_2d(test_population.extract_obj())

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
        for obj_cntr in range(problem.n_obj):
            # Conduct hyperopt routine
            model, hyperparams = self._do(obj_cntr)

            # Store optimised model and hyperparameters
            opt_models.append(model)
            opt_params.append(hyperparams)

        return opt_models, opt_params

    def do_cons(self, problem, population, train_population, test_population):
        # Extract total set from population
        self.x_total = np.atleast_2d(population.extract_var())
        self.x_total = self._normalise(self.x_total)
        self.y_total = np.atleast_2d(population.extract_cons())

        # Extract test set from population
        self.x_train = np.atleast_2d(train_population.extract_var())
        self.x_train = self._normalise(self.x_train)
        self.y_train = np.atleast_2d(train_population.extract_cons())

        # Extract test set from population
        self.x_test = np.atleast_2d(test_population.extract_var())
        self.x_test = self._normalise(self.x_test)
        self.y_test = np.atleast_2d(test_population.extract_cons())

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
        for cons_cntr in range(problem.n_con):
            # Conduct hyperopt routine
            model, hyperparams = self._do(cons_cntr)

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
                 'transform': hp.choice('transform', self.transform_list)}

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
                # TODO: temp fix for cubic kernel with width hyperparameter
                if self.kernel_list[opt_params['kernel']] == 'cubic':
                    c_width = 0.5
                else:
                    c_width = opt_params['c']

                if self.transform_list[opt_params['transform']] == 'bilog':
                    y_total = bilog_transform(self.y_total)
                else:
                    y_total = self.y_total

                # build final model
                opt_model = RBF(n_dim=self.n_dim, c=c_width, p_type=self.p_type, kernel_type=self.kernel_list[opt_params['kernel']])
                opt_model.fit(self.x_total, y_total[:, self.obj_cntr].flatten())
                break
            except np.linalg.LinAlgError:
                opt_model = None
                opt_params['c'] = 1.05 * opt_params['c']
                continue

        if opt_model is None:
            opt_model = RBF(n_dim=self.n_dim, c=self.c, p_type=self.p_type, kernel_type=self.kernel_type)
            opt_model.fit(self.x_total, self.y_total[:, self.obj_cntr])
        
        # if self.verbose:
        print(f"Opt RBF: {obj_cntr}, C: {c_width}, Kernel: {self.kernel_list[opt_params['kernel']]}, Transform: {self.transform_list[opt_params['transform']]}")

        return opt_model, (opt_params['c'], opt_params['transform'])

    def objective(self, space):
        # TODO: temp fix for cubic kernel with width hyperparameter
        if space['c'] == 'cubic':
            c_width = 0.5
        else:
            c_width = space['c']

        # Apply bilog transformation
        if space['transform'] == 'bilog':
            y_train = bilog_transform(self.y_train)
            use_bilog = True
        else:
            y_train = self.y_train
            use_bilog = False

        # Build Model
        model = RBF(n_dim=self.n_dim, c=c_width, p_type=self.p_type, kernel_type=space['kernel'])

        try:
            model.fit(self.x_train, y_train[:, self.obj_cntr].flatten())

            # Calculate accuracy metric (objective)
            accuracy = self.evaluator(model, self.obj_cntr, use_bilog)

            # Assign good status
            status = STATUS_OK
        except np.linalg.LinAlgError:
            # print(f"RBF with c={space['c']}: Singular Matrix")
            status = STATUS_FAIL
            accuracy = np.inf

        # print(f"obj: {self.obj_cntr}, acc: {accuracy}, c: {space['c']} kernel: {space['kernel']}, transform: {space['transform']}")
        # print(f"{self.obj_cntr},{accuracy[0]},{space['c']}")
        return {'loss': accuracy, 'status': status}

    def calc_mse(self, model, obj_cntr, use_bilog):
        # Extract test population size
        n_test = len(self.y_test)

        # Iterate through individuals and calculate model prediction error
        accuracy = 0
        for idx in range(n_test):
            y_pred = model.predict(self.x_test[idx])
            if use_bilog:
                y_pred = reverse_bilog_transform(y_pred)
            # print(idx, y_pred, self.y_test[idx, obj_cntr])
            accuracy += (self.y_test[idx, obj_cntr] - y_pred) ** 2    # MSE Calculation
        accuracy = (1 / n_test) * accuracy                             # MSE Normalisation

        return accuracy

    def calc_sep(self, model, obj_cntr, use_bilog):
        # Extract test population size
        n_test = len(self.y_test)

        # Apply bilog transformation
        if use_bilog:
            y_test = bilog_transform(self.y_test)
        else:
            y_test = self.y_test

        # Iterate through individuals and calculate model sep error
        accuracy = 0

        # Predict surrogate function values
        pred_values = np.zeros(n_test)
        for idx in range(n_test):
            pred_values[idx] = model.predict(self.x_test[idx])

        # Evaluate the SEP
        for i in range(n_test):
            for j in range(i + 1, n_test):
                accuracy += self._pairwise_comp(y_test[i, obj_cntr], y_test[j, obj_cntr],
                                                pred_values[i], pred_values[j])

        # Normalise SEP
        accuracy *= 1 / (0.5 * n_test * (n_test - 1))

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
