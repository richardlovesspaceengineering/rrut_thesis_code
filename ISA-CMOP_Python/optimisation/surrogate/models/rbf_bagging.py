import numpy as np
import random

from optimisation.model.surrogate import Surrogate
from optimisation.surrogate.models.rbf import RBF
# from optimisation.surrogate.models.mars import MARSRegression
# from optimisation.surrogate.models.mars import MARS
# from optimisation.surrogate.models.svr import SupportVectorRegression


class RBFBaggingSurrogate(Surrogate):
    def __init__(self, n_dim, l_b, u_b, c=0.5, p_type='linear', kernel_types=None, n_base_learners=None,
                 use_same_kernel=True, sampling_frac=0.75, **kwargs):

        super().__init__(n_dim=n_dim, l_b=l_b, u_b=u_b, **kwargs)
        # List of implemented kernels
        self.kernel_types = ["cubic", "gaussian", "multiquadratic", "inverse_multiquadratic",
                                 "matern_32", "matern_52", "matern_12", "thin_plate_spline"]
                                 # "cauchy", "logistic", "hyperbolic_tangent_sigmoid"]
        # Kernel list
        if kernel_types is None:
            self.kernels = self.kernel_types
        else:
            self.kernels = kernel_types

        if n_base_learners is None:
            n_base_learners = len(self.kernels)
        else:
            assert n_base_learners <= len(self.kernels)

        # Parameters
        self.c = c
        self.p_type = p_type
        self.n_base_learners = n_base_learners
        self.use_same_kernel = use_same_kernel
        self.sampling_frac = sampling_frac

        # Create Ensemble of RBF kernels for bagging
        if self.use_same_kernel:
            ensemble = [RBF(n_dim=n_dim, c=self.c, p_type=self.p_type, kernel_type=self.kernels[0])
                        for _ in range(self.n_base_learners)]
        else:
            ensemble = [RBF(n_dim=n_dim, c=self.c, p_type=self.p_type, kernel_type=self.kernels[k])
                        for k in range(self.n_base_learners)]

        self.model = ensemble

        # Bagging indices
        self.sub_samples = []

    def _bagging_sampling(self):
        n_to_sample = int(len(self.x) * self.sampling_frac)
        index_range = list(range(len(self.x)))
        for i in range(self.n_base_learners):
            sub_indices = random.sample(index_range, n_to_sample)
            self.sub_samples.append(sub_indices)

    def _train(self):

        # Compute mean and std of training function values
        # self._mu = np.mean(self.y)
        self._mu = np.median(self.y)
        self._sigma = max([np.std(self.y), 1e-6])

        # Scale training data by variable bounds
        self._x = (self.x - self.l_b) / (self.u_b - self.l_b)

        # Bagging sampling of training points
        self._bagging_sampling()

        for i, model in enumerate(self.model):

            # Extract bagging sample points
            x_train = self._x[self.sub_samples[i]]
            y_train = self.y[self.sub_samples[i]]

            if isinstance(model, RBF):
                if model.p_type is None:
                    # Normalise function values
                    _y = y_train - self._mu

                    # Train model
                    model.fit(x_train, _y)
                else:
                    # Train model
                    model.fit(x_train, y_train)
            else:
                # Train model
                model.fit(x_train, y_train)

    def _predict(self, x):
        y = np.zeros(len(self.model))
        _x = (x - self.l_b) / (self.u_b - self.l_b)

        for i, model in enumerate(self.model):
            if isinstance(model, RBF):
                if model.p_type is None:
                    # Train model
                    y[i] = model.predict(_x) + self._mu
                else:
                    # Train model
                    y[i] = model.predict(_x)
            else:
                # Train model
                y[i] = model.predict(_x)

        y_out = np.mean(y)
        # y_out = np.median(y)
        return y_out

    def _cv_predict(self, model, model_y, x):

        # Scale input data by variable bounds
        _x = (x - self.l_b) / (self.u_b - self.l_b)

        # Predict function values
        if isinstance(model, RBF):
            if model.p_type is None:
                # _mu = np.mean(model_y)
                _mu = np.median(model_y)
                y = model.predict(_x) + _mu
            else:
                y = model.predict(_x)
        else:
            y = model._predict(_x)

        return y

    def _cv_predict_ensemble(self, x):
        y = np.zeros(len(self.model))
        for i, model in enumerate(self.model):
            y[i] = self._predict_model(x, model)

        # y_mean = np.mean(y)
        y_mean = np.median(y)
        y_std = np.std(y)
        return y_mean, y_std, y

    def _predict_variance(self, x):
        y = np.zeros(len(self.model))
        for i, model in enumerate(self.model):
            y[i] = self._predict_model(x, model)

        y_std = np.std(y)
        if model.p_type is None:
            y_std *= self._sigma
        return y_std ** 2

    def _predict_median(self, x):
        y = np.zeros(len(self.model))
        for i, model in enumerate(self.model):
            y[i] = self._predict_model(x, model)

        return y

    def update_cv_models(self):

        self.cv_models = self.model

        # Training each of the cross-validation models
        for i, model in enumerate(self.cv_models):
            model.fit(self._x, self.y)


