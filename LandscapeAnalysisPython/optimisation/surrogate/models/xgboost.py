"""sudo pip install xgboost"""

import xgboost
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import numpy as np

from optimisation.model.surrogate import Surrogate


class XGBoostRegression(Surrogate):

    def __init__(self, n_dim, l_b, u_b, n_estimators=1000, max_depth=20, eta=0.01, subsample=1.0,
                                       colsample_bytree=1.0, light=True, **kwargs):

        super().__init__(n_dim=n_dim, l_b=l_b, u_b=u_b, **kwargs)

        """ parameters
        n_estimators: The number of trees in the ensemble, often increased until no further improvements are seen.
        max_depth: The maximum depth of each tree, often values are between 1 and 10.
        eta: The learning rate used to weight each model, often set to small values such as 0.3, 0.1, 0.01, or smaller.
        subsample: The number of samples (rows) used in each tree, set to a value between 0 and 1, 
                    often 1.0 to use all samples.
        colsample_bytree: Number of features (columns) used in each tree, set to a value between 0 and 1, 
                    often 1.0 to use all features."""
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.eta = eta
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.light = light

        if self.light:
            self.model = LGBMRegressor()
        else:
            self.model = XGBRegressor(n_estimators=self.n_estimators,
                                      max_depth=self.max_depth,
                                      eta=self.eta,
                                      subsample=self.subsample,
                                      colsample_bytree=self.colsample_bytree)

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
        self.cv_models = [XGBRegressor() for _ in range(self.cv_k)]

        # Training each of the cross-validation models
        for i, model in enumerate(self.cv_models):
            model.fit(self._x[~np.in1d(np.arange(self.n_pts), self.cv_training_indices[i]), :],
                      self.y[~np.in1d(np.arange(self.n_pts), self.cv_training_indices[i])])

