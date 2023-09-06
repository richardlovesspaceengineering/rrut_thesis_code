import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel

from optimisation.model.surrogate import Surrogate


class GaussianProcess(Surrogate):

    def __init__(self, n_dim, l_b, u_b, cov_type='RBF', scale_kernel=True, p_type=None, apply_noise=True,
                 n_lml_opt_restarts=5, **kwargs):

        super().__init__(n_dim=n_dim, l_b=l_b, u_b=u_b, **kwargs)

        # Covariance function
        if cov_type == 'RBF':
            cov_func = RBF(length_scale=0.5*np.ones(self.n_dim,), length_scale_bounds=(0.05, 2.0))
        elif cov_type == 'matern_52':
            cov_func = Matern(length_scale=0.5*np.ones(self.n_dim,), length_scale_bounds=(0.05, 2.0),
                              nu=2.5)
        elif cov_type == 'matern_32':
            cov_func = Matern(length_scale=0.5*np.ones(self.n_dim,), length_scale_bounds=(0.05, 2.0),
                              nu=1.5)
        else:
            cov_func = None
            raise Exception('Undefined covariance function')

        # Scale function
        if scale_kernel:
            scale_func = ConstantKernel(constant_value=1.0, constant_value_bounds=(0.01, 100.0))
        else:
            scale_func = None

        # Global polynomial function
        if p_type is not None:
            if p_type == 'constant':
                global_func = ConstantKernel(constant_value=0.0, constant_value_bounds=(0.05, 1e6))
            else:
                raise Exception('Undefined global function')
        else:
            global_func = None

        # Noise function
        if apply_noise:
            noise_func = WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-6, 1e-2))
        else:
            noise_func = None

        # Form full kernel
        kernel = cov_func
        if scale_func is not None:
            kernel *= scale_func
        if global_func is not None:
            kernel += global_func
        if noise_func is not None:
            kernel += noise_func

        # Model instance (pySOT)
        self.kernel = kernel
        self.n_lml_opt_restarts = n_lml_opt_restarts
        self.model = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=self.n_lml_opt_restarts)
        # Uses the 'L-BGFS-B' algorithm from scipy.optimize.minimize to tune the kernel hyperparameters to fit the model
        # n_lml_opt_restarts is the number of times the optimiser can be restarted as the log-margin-likelihood function
        # may have multiple local optima

        self._mu = 0.0
        self._sigma = 0.0

    def _train(self):

        # Compute mean and std of training function values
        self._mu = np.mean(self.y)
        self._sigma = max([np.std(self.y), 1e-6])

        # Normalise function values
        _y = (self.y - self._mu)/self._sigma

        # Scale training data by variable bounds
        self._x = (self.x - self.l_b)/(self.u_b - self.l_b)

        # Train model
        self.model.fit(self._x, _y)

    def _predict(self, x):

        x = np.atleast_2d(x)

        # Scale input data by variable bounds
        _x = (x - self.l_b)/(self.u_b - self.l_b)

        # Predict function values & re-scale
        y = self._mu + self._sigma*self.model.predict(_x)

        return y

        # x = np.atleast_2d(x)
        #
        # if len(x.shape) > 2:
        #     x_new = x.reshape((x.shape[0] * x.shape[1], x.shape[2]))
        # else:
        #     x_new= x
        #
        # # Scale input data by variable bounds
        # _x = (x_new - self.l_b) / (self.u_b - self.l_b)
        #
        # # Predict function values & re-scale
        # # y = self._mu + self._sigma * self.model.predict([_x])
        # y = self._mu + self._sigma * self.model.predict(_x)
        #
        # if len(x.shape) > 2:
        #         y= y.reshape(x.shape[0], x.shape[1])
        #
        # if x.shape[0] == 1:
        #     return y[0]
        # else:
        #     return y

    def _predict_variance(self, x):

        # Scale input data by variable bounds
        _x = (x - self.l_b)/(self.u_b - self.l_b)

        # Predict standard deviation & re-scale
        _, std = self.model.predict(_x, return_std=True)
        std *= self._sigma

        return std**2.0

    def _cv_predict(self, model, model_y, x):

        x = np.atleast_2d(x)

        # Scale input data by variable bounds
        _x = (x - self.l_b)/(self.u_b - self.l_b)

        # Predict function values & re-scale
        _mu = np.mean(model_y)
        _sigma = max([np.std(model_y), 1e-6])
        y = _mu + _sigma*model.predict(_x)

        return y

    def update_cv_models(self):

        # k-fold LSO cross-validation indices
        random_state = np.random.default_rng()
        self.cv_training_indices = np.array_split(random_state.choice(self.n_pts, size=self.n_pts, replace=False), self.cv_k)
        self.cv_models = [GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=self.n_lml_opt_restarts)
                          for _ in range(self.cv_k)]

        # Training each of the cross-validation models
        for i, model in enumerate(self.cv_models):
            model.fit(self._x[~np.in1d(np.arange(self.n_pts), self.cv_training_indices[i]), :],
                      self.y[~np.in1d(np.arange(self.n_pts), self.cv_training_indices[i])])


class GP(object):
    def __init__(self, n_dim, cov_type='RBF', scale_kernel=True, p_type=None, apply_noise=True,
                 n_lml_opt_restarts=5):

        self.n_dim = n_dim

        self.x = np.zeros((0, self.n_dim))
        self.y = np.zeros(self.n_dim)

        # Covariance function
        if cov_type == 'RBF':
            cov_func = RBF(length_scale=0.5*np.ones(self.n_dim,), length_scale_bounds=(0.05, 2.0))
        elif cov_type == 'matern_52':
            cov_func = Matern(length_scale=0.5*np.ones(self.n_dim,), length_scale_bounds=(0.05, 2.0), nu=2.5)
        elif cov_type == 'matern_32':
            cov_func = Matern(length_scale=0.5*np.ones(self.n_dim,), length_scale_bounds=(0.05, 2.0), nu=1.5)
        else:
            cov_func = None
            raise Exception('Undefined covariance function')

        # Scale function
        if scale_kernel:
            scale_func = ConstantKernel(constant_value=1.0, constant_value_bounds=(0.01, 100.0))
        else:
            scale_func = None

        # Global polynomial function
        if p_type is not None:
            if p_type == 'constant':
                global_func = ConstantKernel(constant_value=0.0, constant_value_bounds=(0.05, 1e6))
            else:
                raise Exception('Undefined global function')
        else:
            global_func = None

        # Noise function
        if apply_noise:
            noise_func = WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-6, 1e-2))
        else:
            noise_func = None

        # Form full kernel
        kernel = cov_func
        if scale_func is not None:
            kernel *= scale_func
        if global_func is not None:
            kernel += global_func
        if noise_func is not None:
            kernel += noise_func

        # Model instance (pySOT)
        self.kernel = kernel
        self.n_lml_opt_restarts = n_lml_opt_restarts
        self.model = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=self.n_lml_opt_restarts)

        self._mu = 0.0
        self._sigma = 0.0

    def fit(self, x, y):

        self.x = np.array(x)
        self.y = np.array(y)

        # Train model
        self.model.fit(self.x, self.y)

    def predict(self, x):
        _x = np.array(x)

        # Predict model
        y = self.model.predict(np.atleast_2d(_x))

        return y
