import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import ElasticNetCV

from optimisation.model.surrogate import Surrogate

# TODO - do not use. Does not work very well
class ElasticNetRegression(Surrogate):

    def __init__(self, n_dim, l_b, u_b, cv=5, l1_ratio=0.5, **kwargs):

        super().__init__(n_dim=n_dim, l_b=l_b, u_b=u_b, **kwargs)

        self.cv = cv
        self.l1_ratio = l1_ratio

        cv = RepeatedKFold(n_splits=self.cv, n_repeats=3, random_state=1)

        ratios = np.arange(0, 1, 0.01)
        alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]
        self.model = ElasticNetCV(l1_ratio=ratios, alphas=alphas, cv=cv, n_jobs=-1)

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
        raise Exception('Variance prediction not implemented for RBF')

    def update_cv_models(self):

        # k-fold LSO cross-validation indices
        random_state = np.random.default_rng()
        self.cv_training_indices = np.array_split(random_state.choice(self.n_pts, size=self.n_pts, replace=False),
                                                  self.cv_k)
        cv = RepeatedKFold(n_splits=self.cv, n_repeats=3, random_state=1)

        ratios = np.arange(0, 1, 0.01)
        alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]

        self.cv_models = [ElasticNetCV(l1_ratio=ratios, alphas=alphas, cv=cv, n_jobs=-1) for _ in range(self.cv_k)]

        # Training each of the cross-validation models
        for i, model in enumerate(self.cv_models):
            model.fit(self._x[~np.in1d(np.arange(self.n_pts), self.cv_training_indices[i]), :],
                      self.y[~np.in1d(np.arange(self.n_pts), self.cv_training_indices[i])])


