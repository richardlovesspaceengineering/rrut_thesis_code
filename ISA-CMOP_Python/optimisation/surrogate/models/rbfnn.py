"""piggybacks of keras and tensorflow
install as follows
pip3 install --upgrade setuptools
pip3 install tensorflow
pip3 install keras
"""

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop, Adagrad, Adam
from tensorflow.keras.losses import MSE

import matplotlib.pyplot as plt

from optimisation.surrogate.models.rbf_neural_net.rbflayer import RBFLayer
from optimisation.surrogate.models.rbf_neural_net.kmeans_initializer import InitCentersKMeans
from optimisation.surrogate.models.rbf_neural_net.rbflayer import InitCentersRandom

from optimisation.model.surrogate import Surrogate


class RBFNeuralNetRegression(Surrogate):

    def __init__(self, n_dim, l_b, u_b, output_dim=1, n_neurons=50, betas=2.0,
                 batch_size=50, epochs=20000, verbose=False, patience=50, kernel='gaussian', **kwargs):

        super().__init__(n_dim=n_dim, l_b=l_b, u_b=u_b, **kwargs)

        self.output_dim = output_dim
        # if initialiser.lower() == 'kmeans':
        #     self.initialiser = InitCentersKMeans((0.5*np.ones - self.l_b) / (self.u_b - self.l_b))
        self.initialiser = None
        self.n_neurons = n_neurons
        self.betas = betas
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.patience = patience
        self.kernel = kernel

        model = Sequential()
        rbflayer = RBFLayer(self.n_neurons,
                            initializer=self.initialiser,
                            betas=self.betas,
                            input_shape=(n_dim,),
                            kernel='gaussian')
        outputlayer = Dense(1, use_bias=False)

        model.add(rbflayer)
        model.add(outputlayer)

        model.compile(loss='mean_squared_error',
                      optimizer=RMSprop())


        self.model = model

    def _train(self):

        # Compute mean and std of training function values
        self._mu = np.mean(self.y)
        self._sigma = max([np.std(self.y), 1e-6])

        # Normalise function values
        _y = (self.y - self._mu) / self._sigma

        # Scale training data by variable bounds
        self._x = (self.x - self.l_b) / (self.u_b - self.l_b)

        # Train model
        early_stopping_cb = EarlyStopping(patience=self.patience, restore_best_weights=True, monitor='loss')
        self.model.fit(self._x, _y, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose,
                       callbacks=[early_stopping_cb], use_multiprocessing=False)

    def _predict(self, x):

        x = np.atleast_2d(x)

        if len(x.shape) > 2:
            x_new = x.reshape((x.shape[0] * x.shape[1], x.shape[2]))
        else:
            x_new= x

        # Scale input data by variable bounds
        _x = (x_new - self.l_b) / (self.u_b - self.l_b)

        # Predict function values & re-scale
        y = self._mu + self._sigma * self.model.predict([_x])

        if len(x.shape) > 2:
            if self.output_dim > 1:
                y = y.reshape(x.shape[0], x.shape[1], self.output_dim)
            else:
                y= y.reshape(x.shape[0], x.shape[1])

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
        y = _mu + _sigma * model.predict([_x])

        if x.shape[0] == 1:
            return y[0]
        else:
            return y

    def _predict_variance(self, x):
        raise Exception('Variance prediction not implemented for RBF')

    def update_cv_models(self):
        # k-fold LSO cross-validation indices
        random_state = np.random.default_rng()
        self.cv_training_indices = np.array_split(random_state.choice(self.n_pts, size=self.n_pts, replace=False),
                                                  self.cv_k)
        self.cv_models = [self.model for _ in range(self.cv_k)]

        # Training each of the cross-validation models
        early_stopping_cb = EarlyStopping(patience=self.patience,restore_best_weights=True, monitor='loss')
        for i, model in enumerate(self.cv_models):
            model.fit(self._x[~np.in1d(np.arange(self.n_pts), self.cv_training_indices[i]), :],
                      self.y[~np.in1d(np.arange(self.n_pts), self.cv_training_indices[i])],
                      batch_size = self.batch_size, epochs = self.epochs, verbose = self.verbose,
                      callbacks = [early_stopping_cb], use_multiprocessing=False)

