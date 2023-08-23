"""Bagging regressor from scikit learn
"""

import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVR

from optimisation.model.surrogate import Surrogate


class BaggingRegression(Surrogate):

    def __init__(self, n_dim, l_b, u_b, base_estimator=SVR(), n_estimators=10, random_state=42, **kwargs):

        super().__init__(n_dim=n_dim, l_b=l_b, u_b=u_b, **kwargs)

        # TODO add option for different base_estimator (see example in scikit learn documentation)
        # needs to be imported here to be used
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state


        self.model = BaggingRegressor(base_estimator=self.base_estimator, n_estimators=self.n_estimators,
                                      random_state=self.random_state)

    def _train(self):

        # Compute mean and std of training function values
        self._mu = np.mean(self.y)
        self._sigma = max([np.std(self.y), 1e-6])

        # Normalise function values
        _y = (self.y - self._mu) / self._sigma

        # Scale training data by variable bounds
        self._x = (self.x - self.l_b) / (self.u_b - self.l_b)

        # Train model
        self.model.fit(self._x, _y)

    def _predict(self, x):

        x = np.atleast_2d(x)

        # Scale input data by variable bounds
        _x = (x - self.l_b) / (self.u_b - self.l_b)

        # Predict function values & re-scale
        y = self._mu + self._sigma * self.model.predict(_x)

        return y

    def _cv_predict(self, model, model_y, x):

        x = np.atleast_2d(x)

        # Scale input data by variable bounds
        _x = (x - self.l_b) / (self.u_b - self.l_b)

        # Predict function values & re-scale
        _mu = np.mean(model_y)
        _sigma = max([np.std(model_y), 1e-6])
        y = _mu + _sigma * model.predict(_x)

        return y

    def _predict_variance(self, x):
        raise Exception('Variance prediction not implemented for Bagging')

    def update_cv_models(self):

        # k-fold LSO cross-validation indices
        random_state = np.random.default_rng()
        self.cv_training_indices = np.array_split(random_state.choice(self.n_pts, size=self.n_pts, replace=False),
                                                  self.cv_k)
        self.cv_models = [BaggingRegressor() for _ in range(self.cv_k)]

        # Training each of the cross-validation models
        for i, model in enumerate(self.cv_models):
            model.fit(self._x[~np.in1d(np.arange(self.n_pts), self.cv_training_indices[i]), :],
                      self.y[~np.in1d(np.arange(self.n_pts), self.cv_training_indices[i])])

