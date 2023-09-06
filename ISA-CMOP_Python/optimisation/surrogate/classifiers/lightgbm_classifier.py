"""sudo pip install lightgbm"""

from lightgbm import LGBMClassifier
import numpy as np

from optimisation.model.surrogate import Surrogate


class LightGBMClassification(Surrogate):

    def __init__(self, n_dim, l_b, u_b, **kwargs):

        super().__init__(n_dim=n_dim, l_b=l_b, u_b=u_b, **kwargs)

        self._mid = 0.5
        self._range = 0.5

        self.verbose = False

        # 'binary', 'multiclass'
        if 'objective' in kwargs:
            self.objective = kwargs['objective']
        else:
            raise Exception('No objective provided!')

        if 'class_weights' in kwargs:
            self.class_weights = kwargs['class_weights']
        else:
            self.class_weights = None

        if 'num_class' in kwargs:
            self.num_class = kwargs['num_class']
        else:
            self.num_class = 2  # Defaulting to binary classification

        if self.num_class > 2:
            self.is_unbalance = True
        else:
            self.is_unbalance = False

        if 'num_leaves' in kwargs:
            self.num_leaves = kwargs['num_leaves']
        else:
            self.num_leaves = 2

        if 'min_child_samples' in kwargs:
            self.min_child_samples = kwargs['min_child_samples']
        else:
            self.min_child_samples = 1

        if 'reg_alpha' in kwargs:
            self.reg_alpha = kwargs['reg_alpha']
        else:
            self.reg_alpha = 2.5

        if 'reg_lambda' in kwargs:
            self.reg_lambda = kwargs['reg_lambda']
        else:
            self.reg_lambda = 2.5

        if 'eta' in kwargs:
            self.eta = kwargs['eta']
        else:
            self.eta = 0.15

        if 'max_depth' in kwargs:
            self.max_depth = kwargs['max_depth']
        else:
            self.max_depth = 8

        if 'num_iterations' in kwargs:
            self.num_iterations = kwargs['num_iterations']
        else:
            self.num_iterations = 100

        if 'device_type' in kwargs:
            self.device_type = kwargs['device_type']
        else:
            self.device_type = 'cpu'

        self.model = LGBMClassifier(objective=self.objective,
                                    num_iterations=self.num_iterations,
                                    num_class=self.num_class,
                                    is_unbalance=self.is_unbalance,
                                    learning_rate=self.eta,
                                    max_depth=self.max_depth,
                                    num_leaves=self.num_leaves,
                                    min_child_samples=self.min_child_samples,
                                    reg_alpha=self.reg_alpha,
                                    reg_lambda=self.reg_lambda,
                                    device_type=self.device_type
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
        raise Exception('Variance prediction not implemented for Gradient Boosting!')

    def update_cv_models(self):

        # k-fold LSO cross-validation indices
        random_state = np.random.default_rng()
        self.cv_training_indices = np.array_split(random_state.choice(self.n_pts, size=self.n_pts, replace=False),
                                                  self.cv_k)
        self.cv_models = [LGBMClassifier() for _ in range(self.cv_k)]

        # Training each of the cross-validation models
        for i, model in enumerate(self.cv_models):
            model.fit(self._x[~np.in1d(np.arange(self.n_pts), self.cv_training_indices[i]), :],
                      self.y[~np.in1d(np.arange(self.n_pts), self.cv_training_indices[i])])

