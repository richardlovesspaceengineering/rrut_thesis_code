"""sudo pip install sklearn"""

from sklearn.multiclass import OutputCodeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

from optimisation.model.surrogate import Surrogate


class ECOCClassification(Surrogate):

    def __init__(self, n_dim, l_b, u_b, **kwargs):

        super().__init__(n_dim=n_dim, l_b=l_b, u_b=u_b, **kwargs)

        """ parameters
        estimator: [string] of desired base learners ('svm' is default)
        code_size: [int] of encoding size (total_classifiers = code_size * n_classes) (2 is default)
        n_jobs: [int] number of processors to utilise during training (-1 is default)
        """

        self._mid = 0.5
        self._range = 0.5

        self.random_state = 0

        if 'n_jobs' in kwargs:
            self.n_jobs = kwargs['n_jobs']
        else:
            self.n_jobs = -1  # Utilise all available cores

        # Available base learners: svm, logistic, decision_tree
        if 'estimator' in kwargs:
            self.estimator = kwargs['estimator']
        else:
            self.estimator = 'svm'

        if 'svc' in self.estimator:
            self.base_estimator = SVC(kernel='rbf')
        elif 'logistic' in self.estimator:
            self.base_estimator = LogisticRegression()
        elif 'decision_tree' in self.estimator:
            self.base_estimator = DecisionTreeClassifier()
        elif 'knn' in self.estimator:
            self.base_estimator = KNeighborsClassifier(n_neighbors=3)

        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 3

        self.model = OutputCodeClassifier(estimator=self.base_estimator,
                                          code_size=self.code_size,
                                          random_state=self.random_state,
                                          n_jobs=self.n_jobs)

    def _train(self):

        # Scale training data by variable bounds
        self._x = (self.x - self.l_b) / (self.u_b - self.l_b)

        # Normalise to [-1, 1]
        # self._x = (self._x - self._mid) / self._range

        # Train model
        self.model.fit(self._x, self.y)

    def _predict(self, x):

        x = np.atleast_2d(x)

        # Scale input data by variable bounds
        _x = (x - self.l_b) / (self.u_b - self.l_b)

        # Normalise to [-1, 1]
        # _x = (_x - self._mid) / self._range

        # Predict labels
        y = self.model.predict(_x)

        return y

    def _predict_proba(self, x):

        raise Exception('Prediction probability not implemented for ECOC classifier!')

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
        self.cv_models = [OutputCodeClassifier(estimator=self.base_estimator) for _ in range(self.cv_k)]

        # Training each of the cross-validation models
        for i, model in enumerate(self.cv_models):
            model.fit(self._x[~np.in1d(np.arange(self.n_pts), self.cv_training_indices[i]), :],
                      self.y[~np.in1d(np.arange(self.n_pts), self.cv_training_indices[i])])