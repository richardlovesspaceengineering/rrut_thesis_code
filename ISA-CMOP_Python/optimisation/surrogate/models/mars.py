"""MARS algorithm - Multi-variate adaptive regression splines
from py-earth sci-kit learn contribution
install via conda on ubuntu
conda install -c conda-forge sklearn-contrib-py-earth
"""

import numpy as np
from pyearth import Earth


from optimisation.model.surrogate import Surrogate


class MARSRegression(Surrogate):

    def __init__(self, n_dim, l_b, u_b, max_terms=500, max_degree=4, **kwargs):

        super().__init__(n_dim=n_dim, l_b=l_b, u_b=u_b, **kwargs)

        self.max_terms = max_terms
        self.max_degree = max_degree

        self.model = Earth(max_terms=self.max_terms, max_degree=self.max_degree)

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

        if len(x.shape) > 2:
            x_new = x.reshape((x.shape[0] * x.shape[1], x.shape[2]))
        else:
            x_new = x

        # Scale input data by variable bounds
        _x = (x_new - self.l_b) / (self.u_b - self.l_b)

        # Predict function values & re-scale
        # y = self._mu + self._sigma * self.model.predict([_x])
        y = self._mu + self._sigma * self.model.predict(_x)

        if len(x.shape) > 2:
            y = y.reshape(x.shape[0], x.shape[1])

        if x.shape[0] == 1:
            return y[0]
        else:
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
        raise Exception('Variance prediction not implemented for MARS')

    def update_cv_models(self):

        # k-fold LSO cross-validation indices
        random_state = np.random.default_rng()
        self.cv_training_indices = np.array_split(random_state.choice(self.n_pts, size=self.n_pts, replace=False),
                                                  self.cv_k)
        self.cv_models = [Earth() for _ in range(self.cv_k)]

        # Training each of the cross-validation models
        for i, model in enumerate(self.cv_models):
            model.fit(self._x[~np.in1d(np.arange(self.n_pts), self.cv_training_indices[i]), :],
                      self.y[~np.in1d(np.arange(self.n_pts), self.cv_training_indices[i])])


class MARS(object):
    def __init__(self, n_dim, max_terms=500, max_degree=4):

        self.n_dim = n_dim

        self.x = np.zeros((0, self.n_dim))
        self.y = np.zeros(self.n_dim)

        self.max_terms = max_terms
        self.max_degree = max_degree

        self.model = Earth(max_terms=self.max_terms, max_degree=self.max_degree)

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
