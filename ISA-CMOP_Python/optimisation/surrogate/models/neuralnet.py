"""piggybacks of keras and tensorflow
install as follows
pip3 install --upgrade setuptools
pip3 install tensorflow
pip3 install keras
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping

from optimisation.model.surrogate import Surrogate


class NeuralNetRegression(Surrogate):

    def __init__(self, n_dim, l_b, u_b, output_dim,
                 n_neurons=30, dropout_rate=0.2,
                 batch_size=50, epochs=2000, verbose=False, patience=50, **kwargs):

        super().__init__(n_dim=n_dim, l_b=l_b, u_b=u_b, **kwargs)

        self.output_dim = output_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.patience = patience
        self.n_neurons = n_neurons
        self.dropout_rate = dropout_rate

        model = NeuralNetRegressor(
            RegressorModule(n_dim=self.n_dim, num_units=self.n_neurons),
            max_epochs=self.epochs,
            lr=0.01,
            callbacks=[EarlyStopping(patience=350)],
            batch_size=16,
            optimizer=torch.optim.SGD,
            optimizer__momentum=0.90,
            # verbose=0
        )


        self.model = model

    def _train(self):

        # Compute mean and std of training function values
        self._mu = np.mean(self.y)
        self._sigma = max([np.std(self.y), 1e-6])

        # Normalise function values
        _y = (self.y - self._mu) / self._sigma

        # Scale training data by variable bounds
        self._x = (self.x - self.l_b) / (self.u_b - self.l_b)

        _x = self._x.astype(np.float32)
        _y = _y.astype(np.float32)
        _y = _y.reshape(-1, 1)
        # Train model
        self.model.fit(_x, _y)

    def _predict(self, x):

        x = np.atleast_2d(x)

        # Scale input data by variable bounds
        _x = (x - self.l_b) / (self.u_b - self.l_b)

        # Predict function values & re-scale
        _x = _x.astype(np.float32)
        y = self._mu + self._sigma * self.model.predict(_x)

        return y


    def _cv_predict(self, model, model_y, x):

        x = np.atleast_2d(x)

        # Scale input data by variable bounds
        _x = (x - self.l_b) / (self.u_b - self.l_b)
        _x = _x.astype(np.float32)

        # Predict function values & re-scale
        _mu = np.mean(model_y)
        _sigma = max([np.std(model_y), 1e-6])
        y = _mu + _sigma * model.predict(_x)

        return y

    def _predict_variance(self, x):
        raise Exception('Variance prediction not implemented for RBF')

    def update_cv_models(self):
        # k-fold LSO cross-validation indices
        # random_state = np.random.default_rng()
        # self.cv_training_indices = np.array_split(random_state.choice(self.n_pts, size=self.n_pts, replace=False),
        #                                           self.cv_k)
        # self.cv_models = [self.model for _ in range(self.cv_k)]
        #
        # # Training each of the cross-validation models
        # self._x = self._x.astype(np.float32)
        # self.y = self.y.astype(np.float32)
        # for i, model in enumerate(self.cv_models):
        #     x_test = self._x[~np.in1d(np.arange(self.n_pts), self.cv_training_indices[i]), :]
        #     y_test = self.y[~np.in1d(np.arange(self.n_pts), self.cv_training_indices[i])]
        #     y_test = np.atleast_2d(y_test.T)
        #     model.fit(x_test, y_test)
        pass


class RegressorModule(nn.Module):
    def __init__(
            self,
            num_units=10,
            nonlin=F.selu,
            n_dim=2,
    ):
        super(RegressorModule, self).__init__()
        self.num_units = num_units
        self.nonlin = nonlin

        self.dense0 = nn.Linear(n_dim, num_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(0.2)
        self.dense1 = nn.Linear(num_units, 10)
        self.output = nn.Linear(10, 1)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = F.selu(self.dense1(X))
        X = self.output(X)
        return X