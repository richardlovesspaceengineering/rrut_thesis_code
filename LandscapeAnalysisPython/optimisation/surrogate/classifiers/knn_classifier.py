import copy
import warnings

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsClassifier

from optimisation.model.surrogate import Surrogate


class KNNClassification(Surrogate):

    def __init__(self, n_dim, l_b, u_b, k=5, weights='distance', p=2, **kwargs):

        super().__init__(n_dim=n_dim, l_b=l_b, u_b=u_b, **kwargs)

        # SVM Parameters
        self.k = k
        self.weights = weights
        self.p = p
        self.model = KNeighborsClassifier(n_neighbors=self.k, weights=self.weights, p=self.p)

    def _train(self):
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

        # Calculate sigmoid probability function
        y = self.model.predict_proba(x)

        return y

    def _cv_predict(self, model, model_y, x):
        raise Exception('Cross-validation probability prediction not implemented for P-SVM!')

    def _predict_variance(self, x):
        raise Exception('Variance prediction not implemented for P-SVM!')

    def update_cv_models(self):
        pass
