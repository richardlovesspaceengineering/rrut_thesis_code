
from sklearn.svm import SVC
import numpy as np

from optimisation.model.surrogate import Surrogate


class SupportVectorClassification(Surrogate):

    def __init__(self, n_dim, l_b, u_b, C=1.0, kernel='rbf', **kwargs):

        super().__init__(n_dim=n_dim, l_b=l_b, u_b=u_b, **kwargs)

        self.C = C
        self.kernel = kernel
        self.model = SVC(C=self.C, kernel=self.kernel, probability=True)

    def _train(self):

        # Compute mean and std of training function values
        self._mu = np.mean(self.y)
        self._sigma = max([np.std(self.y), 1e-6])

        # Train model
        self.model.fit(self._x, self.y)

    def _predict(self, x):

        x = np.atleast_2d(x)

        # Scale input data by variable bounds
        _x = (x - self.l_b) / (self.u_b - self.l_b)

        # Predict function values & re-scale
        y = self.model.predict(_x)

        return y

    def _predict_proba(self, x):

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

        y = model.predict_proba(_x)

        return y

    def _predict_variance(self, x):
        raise Exception('Variance prediction not implemented for Bagging')

    def update_cv_models(self):

        # k-fold LSO cross-validation indices
        random_state = np.random.default_rng()
        self.cv_training_indices = np.array_split(random_state.choice(self.n_pts, size=self.n_pts, replace=False),
                                                  self.cv_k)
        self.cv_models = [SVC(C=self.C, kernel=self.kernel, gamma=self.gamma, probability=True)
                          for _ in range(self.cv_k)]

        # Training each of the cross-validation models
        for i, model in enumerate(self.cv_models):
            model.fit(self._x[~np.in1d(np.arange(self.n_pts), self.cv_training_indices[i]), :],
                      self.y[~np.in1d(np.arange(self.n_pts), self.cv_training_indices[i])])

