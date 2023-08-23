"""sudo pip install xgboost"""

import xgboost
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import numpy as np

from optimisation.model.surrogate import Surrogate


class XGBoostClassification(Surrogate):

    def __init__(self, n_dim, l_b, u_b, max_depth=2, gamma=2, eta=0.01, reg_alpha=0.5,
                 reg_lambda=0.5, light=True, **kwargs):

        super().__init__(n_dim=n_dim, l_b=l_b, u_b=u_b, **kwargs)

        """ parameters
        n_estimators: The number of trees in the ensemble, often increased until no further improvements are seen.
        max_depth: The maximum depth of each tree, often values are between 1 and 10.
        eta: The learning rate used to weight each model, often set to small values such as 0.3, 0.1, 0.01, or smaller.
        guides from https://www.kaggle.com/rafjaa/dealing-with-very-small-datasets for small datasets
        max_depth low / gamma and eta high to make model more conservative / regularisation high 
        """
        self.max_depth = max_depth
        self.eta = eta
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.light = light

        if self.light:
            self.model = LGBMClassifier()
        else:
            self.model = XGBClassifier(max_depth=self.max_depth,
                                        gamma=self.gamma,
                                        eta=self.eta,
                                        reg_alpha=self.reg_alpha,
                                        reg_lambda=self.reg_lambda
                                        )


    def _train(self):


        # Scale training data by variable bounds
        self._x = (self.x - self.l_b) / (self.u_b - self.l_b)

        # Train model
        self.model.fit(self._x, self.y)

    def _predict(self, x):

        x = np.atleast_2d(x)

        # Scale input data by variable bounds
        _x = (x - self.l_b) / (self.u_b - self.l_b)

        # Predict function values & re-scale
        y = self.model.predict_proba(_x)

        return y

    def _cv_predict(self, model, model_y, x):

        x = np.atleast_2d(x)

        # Scale input data by variable bounds
        _x = (x - self.l_b) / (self.u_b - self.l_b)

        # Predict function values & re-scale
        y = model.predict_proba(_x)

        return y

    def _predict_variance(self, x):
        raise Exception('Variance prediction not implemented for Bagging')

    def update_cv_models(self):

        # k-fold LSO cross-validation indices
        random_state = np.random.default_rng()
        self.cv_training_indices = np.array_split(random_state.choice(self.n_pts, size=self.n_pts, replace=False),
                                                  self.cv_k)
        self.cv_models = [XGBClassifier() for _ in range(self.cv_k)]

        # Training each of the cross-validation models
        for i, model in enumerate(self.cv_models):
            model.fit(self._x[~np.in1d(np.arange(self.n_pts), self.cv_training_indices[i]), :],
                      self.y[~np.in1d(np.arange(self.n_pts), self.cv_training_indices[i])])

