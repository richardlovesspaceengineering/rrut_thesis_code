import numpy as np

from sklearn.svm import SVR

from optimisation.model.surrogate import Surrogate
from optimisation.surrogate.models.deep_belief_net.dbn import SupervisedDBNRegression

class DBNRegression(Surrogate):

    def __init__(self, n_dim, l_b, u_b, x,
                 learning_rate_rbm=0.005,
                 learning_rate=0.005,
                 n_epochs_rbm=500,
                 n_iter_backprop=30000,
                 batch_size=16,
                 activation_function='relu',
                 verbose=True,
                 **kwargs):

        super().__init__(n_dim=n_dim, l_b=l_b, u_b=u_b, **kwargs)


        self.hidden_layers_structure = [int(x.shape[0]), int(2*x.shape[0]), int(2*x.shape[0])]
        self.learning_rate_rbm = learning_rate_rbm
        self.learning_rate = learning_rate
        self.n_epochs_rbm = n_epochs_rbm
        self.n_iter_backprop = n_iter_backprop
        self.batch_size = batch_size
        self.activation_function = activation_function


        self.model = SupervisedDBNRegression(hidden_layers_structure=self.hidden_layers_structure,
                                             learning_rate_rbm=self.learning_rate_rbm,
                                             learning_rate=self.learning_rate,
                                             n_epochs_rbm=self.n_epochs_rbm,
                                             n_iter_backprop=self.n_iter_backprop,
                                             batch_size=self.batch_size,
                                             activation_function=self.activation_function,
                                             verbose=verbose)

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
        self.cv_models = [SupervisedDBNRegression(hidden_layers_structure=self.hidden_layers_structure,
                                    learning_rate_rbm=self.learning_rate_rbm,
                                    learning_rate=self.learning_rate,
                                    n_epochs_rbm=self.n_epochs_rbm,
                                    n_iter_backprop=self.n_iter_backprop,
                                    batch_size=self.batch_size,
                                    activation_function=self.activation_function) for _ in range(self.cv_k)]

        # Training each of the cross-validation models
        for i, model in enumerate(self.cv_models):
            model.fit(self._x[~np.in1d(np.arange(self.n_pts), self.cv_training_indices[i]), :],
                      self.y[~np.in1d(np.arange(self.n_pts), self.cv_training_indices[i])])


