"""sudo pip install xgboost"""

import xgboost as xgb
from xgboost import XGBClassifier
import numpy as np

from optimisation.model.surrogate import Surrogate

xgb.set_config(verbosity=0)


class XGBoostClassification(Surrogate):

    def __init__(self, n_dim, l_b, u_b, **kwargs):

        super().__init__(n_dim=n_dim, l_b=l_b, u_b=u_b, **kwargs)

        """ parameters
        n_estimators: The number of trees in the ensemble, often increased until no further improvements are seen.
        max_depth: The maximum depth of each tree, often values are between 1 and 10.
        eta: The learning rate used to weight each model, often set to small values such as 0.3, 0.1, 0.01, or smaller.
        guides from https://www.kaggle.com/rafjaa/dealing-with-very-small-datasets for small datasets
        max_depth low / gamma and eta high to make model more conservative / regularisation high 
        """

        self._mid = 0.5
        self._range = 0.5

        self.verbose = True

        # multi:softprob/ softmax for multi-class. binary:logistic default for binary-class (xgboost)
        # multiclass/ softmax (lightgbm)
        if 'objective' in kwargs:
            self.objective = kwargs['objective']
        else:
            raise Exception('No objective provided!')

        if 'class_weights' in kwargs:
            self.class_weights = kwargs['class_weights']
        else:
            self.class_weights = None

        if 'num_classes' in kwargs:
            self.num_classes = kwargs['num_classes']
        else:
            self.num_classes = 2  # Defaulting to binary classification

        if 'min_child_weight' in kwargs:
            self.min_child_weight = kwargs['min_child_weight']
        else:
            self.min_child_weight = 1

        if 'max_depth' in kwargs:
            self.max_depth = kwargs['max_depth']
        else:
            self.max_depth = 8

        if 'eta' in kwargs:
            self.eta = kwargs['eta']
        else:
            self.eta = 0.15

        if 'gamma' in kwargs:
            self.gamma = kwargs['gamma']
        else:
            self.gamma = 1.5

        if 'reg_alpha' in kwargs:
            self.reg_alpha = kwargs['reg_alpha']
        else:
            self.reg_alpha = 0.5

        if 'reg_lambda' in kwargs:
            self.reg_lambda = kwargs['reg_lambda']
        else:
            self.reg_lambda = 1.0

        self.model = XGBClassifier(max_depth=self.max_depth,
                                   objective=self.objective,
                                   num_classes=self.num_classes,
                                   gamma=self.gamma,
                                   learning_rate=self.eta,
                                   reg_alpha=self.reg_alpha,
                                   reg_lambda=self.reg_lambda,
                                   min_child_weight=self.min_child_weight
                                   )

    def _train(self):

        # Scale training data by variable bounds
        self._x = (self.x - self.l_b) / (self.u_b - self.l_b)

        # Normalise to [-1, 1]
        self._x = (self._x - self._mid) / self._range

        # Train model
        self.model.fit(self._x, self.y, verbose=self.verbose, sample_weight=self.class_weights)

    def _predict(self, x):

        x = np.atleast_2d(x)

        # Scale input data by variable bounds
        _x = (x - self.l_b) / (self.u_b - self.l_b)

        # Normalise to [-1, 1]
        _x = (_x - self._mid) / self._range

        # Predict labels
        y = self.model.predict(_x)

        return y

    def _predict_proba(self, x):

        x = np.atleast_2d(x)

        # Scale input data by variable bounds
        _x = (x - self.l_b) / (self.u_b - self.l_b)

        # Normalise to [-1, 1]
        _x = (_x - self._mid) / self._range

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

