import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.nn import Module
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from optimisation.model.surrogate import Surrogate


# TODO: REMOVE KERAS AND IMPLEMENT PYTORCH
class DeepNeuralNetClassification(Surrogate):
    """
    Feed-Forward Deep Neural Network implemented with Pytorch
    Requires: torch, Sklearn
    Nvidia GPU support: CUDA, cuDNN, TensorRT, GPU driver/s
    Reference: https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/
    """
    def __init__(self, n_dim, l_b, u_b, activation_kernels=None, n_neurons=None, n_layers=3, n_outputs=1, **kwargs):

        super().__init__(n_dim=n_dim, l_b=l_b, u_b=u_b, **kwargs)

        # # Override for one-hot encoding training labels
        # self.y = np.zeros((0, n_outputs))

        # self._mu = 0.0
        # self._sigma = 0.0
        self._mid = 0.5
        self._range = 0.5

        # Input Arguments
        self.n_inputs = n_dim
        self.n_layers = n_layers
        self.n_outputs = n_outputs

        # To be initialised
        self.data_model = None

        if activation_kernels is None:
            self.activation_kernels = ['relu' for _ in range(n_layers-1)]
            self.activation_kernels.append('sigmoid')
        else:
            if len(activation_kernels) != n_layers:
                raise Exception('Number of layers and provided activation functions do not match!')
            else:
                self.activation_kernels = activation_kernels

        if n_neurons is None:
            self.n_neurons = [self.n_inputs]
            self.n_neurons.extend([200 for _ in range(n_layers-2)])
            self.n_neurons.append(self.n_outputs)
        else:
            if len(n_neurons) != self.n_layers:
                raise Exception('Number of layers and provided list of neurons does not match!')
            else:
                self.n_neurons = n_neurons

        if 'device' in kwargs:
            self.device = kwargs['device']
        else:
            # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.device = torch.device('cpu')

        # Setup model
        self.model = NeuralNet(self.n_inputs, self.n_outputs, self.n_neurons, self.n_layers, self.activation_kernels)
        self.model.to(self.device)

        if 'lr' in kwargs:
            self.lr = kwargs['lr']
        else:
            self.lr = 0.001

        if 'weight_decay' in kwargs:
            self.weight_decay = kwargs['weight_decay']

        if 'optimizer' in kwargs:
            self.optimizer = kwargs['optimizer']
        else:
            self.optimizer = 'adam'  # Options: 'adam', rmsprop

        if 'class_weights' in kwargs:
            self.class_weights = kwargs['class_weights']
            if self.class_weights is not None:
                self.class_weights = torch.tensor(list(self.class_weights.values()), device=self.device).float()
        else:
            self.class_weights = None

        # Proper setup of optimiser
        if 'adam' in self.optimizer:
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if 'epoch' in kwargs:
            self.epochs = kwargs['epoch']
        else:
            self.epochs = 100

        if 'batch_size' in kwargs:
            self.batch_size = kwargs['batch_size']
        else:
            self.batch_size = 10

    # # Override base class method for one-hot encoding training labels
    # def add_points(self, x, y):
    #
    #     x = np.array(x)
    #     y = np.array(y)
    #
    #     self.x = np.vstack((self.x, x))
    #     self.y = np.vstack((self.y, y))
    #     self._x = (self.x - self.l_b)/(self.u_b - self.l_b)
    #     self.n_pts = np.shape(self.y)[0]
    #
    #     self.updated = False
    #     self.cv_updated = False
    #
    # # Override base class method for one-hot encoding training labels
    # def reset(self):
    #
    #     # Reset training data
    #     self.n_pts = 0
    #     self.x = np.zeros((0, self.n_dim))
    #     self._x = np.zeros((0, self.n_dim))
    #     self.y = np.zeros((0, self.n_outputs), dtype=int)
    #     self.updated = False

    def _train(self):
        # Normalise to [-1, 1]
        _x = (self.x - self._mid) / self._range

        # Prepare data
        x_train, _ = self.__prepare_data([_x, self.y])

        # Train model
        self.__train_model(x_train)

    def _predict(self, x):

        x = np.atleast_2d(x)

        # Scale input data by variable bounds
        _x = (x - self.l_b) / (self.u_b - self.l_b)

        # Normalise to [-1, 1]
        _x = (_x - self._mid) / self._range

        # Avoid calculating gradients in inference calculation
        self.model.eval()  # (Only really needed for Dropout & BatchNorm)
        with torch.no_grad():
            y = self.__predict(_x)

            return y

    def _predict_proba(self, x):

        x = np.atleast_2d(x)

        # Scale input data by variable bounds
        _x = (x - self.l_b) / (self.u_b - self.l_b)

        # Normalise to [-1, 1]
        _x = (_x - self._mid) / self._range

        # Avoid calculating gradients in inference calculation
        self.model.eval()  # (Only really needed for Dropout & BatchNorm)
        with torch.no_grad():
             y = self.__predict_proba(_x)

        return y

    def _cv_predict(self, model, model_y, x):

        x = np.atleast_2d(x)

        # Scale input data by variable bounds
        _x = (x - self.l_b) / (self.u_b - self.l_b)

        y = model.predict(_x)

        return y

    def _predict_variance(self, x):
        raise Exception('Variance prediction not implemented for FNN!')

    def update_cv_models(self):
        raise Exception('Variance prediction not implemented for FNN!')

    # prepare the dataset
    def __prepare_data(self, data):
        # load the dataset
        self.data_model = NeuralNetData(data)

        # calculate split
        train, test = self.data_model.get_splits(n_test=0.0)

        # prepare data loaders
        train_dl = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        test_dl = DataLoader(test, batch_size=self.batch_size, shuffle=False)
        return train_dl, test_dl

    # train the model
    def __train_model(self, batches):
        self.model.train()

        # enumerate epochs
        for epoch in range(self.epochs):

            # enumerate mini batches
            for i, batch in enumerate(batches):
                # Extract inputs, outputs
                inputs, targets = batch

                # clear the gradients
                self.optimizer.zero_grad()

                # compute the model output
                yhat = self.model(inputs)

                # calculate loss
                loss = self.criterion(yhat, targets)

                # credit assignment
                loss.backward()

                # update model weights
                self.optimizer.step()

    # make a class prediction for one row of data
    def __predict(self, x):
        # convert row to data
        data = Tensor(x)

        # make prediction
        yhat = self.model(data)

        # Softmax output
        y = functional.softmax(yhat, dim=1)

        # retrieve numpy array
        y = y.detach().numpy()

        # Apply label
        label = np.argmax(y, axis=1)

        return label

    def __predict_proba(self, x):
        # convert row to data
        data = Tensor(x)

        # make prediction
        yhat = self.model(data)

        # Softmax output
        y = functional.softmax(yhat, dim=1)

        # retrieve numpy array
        y = y.detach().numpy()

        return y

    # evaluate the model
    def __evaluate_model(self, test_dl, model):
        predictions, actuals = list(), list()
        for i, (inputs, targets) in enumerate(test_dl):
            # evaluate the model on the test set
            yhat = model(inputs)
            # retrieve numpy array
            yhat = yhat.detach().numpy()
            actual = targets.numpy()
            # convert to class labels
            yhat = np.argmax(yhat, axis=1)
            # reshape for stacking
            actual = actual.reshape((len(actual), 1))
            yhat = yhat.reshape((len(yhat), 1))
            # store
            predictions.append(yhat)
            actuals.append(actual)
        predictions, actuals = np.vstack(predictions), np.vstack(actuals)
        # calculate accuracy
        acc = accuracy_score(actuals, predictions)
        return acc


class NeuralNetData(Dataset):
    # load the dataset
    def __init__(self, data):
        # store the inputs and outputs
        self.X = data[0]
        self.y = data[1]
        # ensure input data is floats
        self.X = self.X.astype('float32')
        # label encode target and ensure the values are floats
        self.y = LabelEncoder().fit_transform(self.y)

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.0):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size

        # calculate the split
        return random_split(self, [train_size, test_size])


# model definition
class NeuralNet(Module):
    # define model elements
    def __init__(self, n_inputs, n_outputs, n_neurons, n_layers, activation):
        super(NeuralNet, self).__init__()

        # Build model layers
        self.layers = []
        first_layer = nn.Sequential(nn.Linear(n_inputs, n_neurons[0]), get_activation(activation[0]))
        self.layers.append(first_layer)
        for idx in range(1, n_layers-1):
            layer = nn.Sequential(nn.Linear(n_neurons[idx-1], n_neurons[idx]), get_activation(activation[idx]))
            self.layers.append(layer)

        self.layers = nn.ModuleList(self.layers)
        self.output_layer = nn.Linear(n_neurons[-2], n_outputs)

    # forward propagate input
    def forward(self, x):
        # Hidden layers
        for layer in self.layers:
            x = layer(x)

        # Output Layer
        x = self.output_layer(x)
        return x


def get_activation(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    else:
        raise RuntimeError("activation should be relu/tanh/sigmoid, not %s." % activation)

