import os

import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""       # TO DISABLE GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # TO ENABLE GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'      # Silent

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
# from tensorflow.python.keras.optimizer_v1 import Adam
from tensorflow.python.keras.optimizer_v2.adam import Adam
from keras.regularizers import L2

# from sklearn.model_selection import StratifiedKFold, cross_val_score

from optimisation.model.surrogate import Surrogate


class DeepNeuralNetClassification(Surrogate):
    """
    Feed-Forward Deep Neural Network implemented with Keras and wrapped around SKlearn methods
    Requires: Sklearn, Keras, Tensorflow
    Nvidia GPU support: CUDA, cuDNN, TensorRT, GPU driver/s
    """
    def __init__(self, n_dim, l_b, u_b, **kwargs):

        super().__init__(n_dim=n_dim, l_b=l_b, u_b=u_b, **kwargs)

        # # Override for one-hot encoding training labels
        # self.y = np.zeros((0, n_outputs))

        # self._mu = 0.0
        # self._sigma = 0.0
        self._mid = 0.5
        self._range = 0.5

        # Input Arguments
        self.n_inputs = int(n_dim)

        if 'n_outputs' in kwargs:
            self.n_outputs = kwargs['n_outputs']
        else:
            raise Exception('Number of outputs not provided!')

        if 'n_layers' in kwargs:
            self.n_layers = kwargs['n_layers']
        else:
            self.n_layers = 3  # Input, hidden, output

        if 'n_neurons' in kwargs:
            self.n_neurons = kwargs['n_neurons']
            assert len(self.n_neurons) == self.n_layers
        else:
            self.n_neurons = [self.n_inputs]
            self.n_neurons.extend([2*n_dim for _ in range(self.n_layers - 2)])
            self.n_neurons.append(self.n_outputs)

        if 'activation_kernels' in kwargs:
            self.activation_kernels = kwargs['activation_kernels']
            assert len(self.activation_kernels) == self.n_layers
        else:
            self.activation_kernels = ['relu' for _ in range(self.n_layers-1)]
            self.activation_kernels.append('sigmoid')

        if 'loss' in kwargs:
            self.loss = kwargs['loss']
        else:
            if self.n_outputs == 1:
                self.loss = 'binary_crossentropy'
            else:
                self.loss = 'sparse_categorical_crossentropy'  # 'categorical_crossentropy'

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
        else:
            self.class_weights = None

        # Proper setup of optimiser
        if 'adam' in self.optimizer:
            self.optimizer = Adam(learning_rate=self.lr,
                                  beta_1=0.9,
                                  beta_2=0.999,
                                  epsilon=1e-07,
                                  amsgrad=False,
                                  name="Adam",
                                  )

        if 'epoch' in kwargs:
            self.epochs = kwargs['epoch']
        else:
            self.epochs = 100

        if 'batch_size' in kwargs:
            self.batch_size = kwargs['batch_size']
        else:
            self.batch_size = 10

        if 'seed' in kwargs:
            self.seed = kwargs['seeds']
        else:
            self.seed = np.random.randint(low=0, high=10000, size=(1,))

        # To omit printout to stdout during training: 0 else 1
        self.verbose = 0

        # Setup a sequential model
        self._model = Sequential()

        # Initialisation (currently hardcoded to Kaiming He)
        self.initialiser = tf.keras.initializers.HeUniform()

        # Overfitting regulariser (currently hardcoded to L2 penalty as per the weight_decay in Pytorch.Adam)
        self.regulariser = L2(self.weight_decay)

        # Add first layer, itermediate layers and final layer
        self._model.add(Dense(self.n_neurons[0], input_shape=(self.n_inputs,), activation=self.activation_kernels[0]))
        for idx in range(1, self.n_layers):
            self._model.add(Dense(self.n_neurons[idx], activation=self.activation_kernels[idx],
                                  kernel_initializer=self.initialiser, kernel_regularizer=self.regulariser))

        # compile the keras model
        self._model.compile(loss=self.loss, optimizer=self.optimizer)  # TODO: metrics=['accuracy']) needed?

        # Create Sklearn-compatible model
        self.model = KerasClassifier(build_fn=self._baseline_model, verbose=self.verbose)

        # KFold model
        # self.kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.seed)

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

    def _baseline_model(self):
        return self._model

    def _train(self):

        # # Compute mean and std of training function values
        # self._mu = np.mean(self.x, )
        # self._sigma = max([np.std(self.y), 1e-6])

        # Normalise training variables to [-1, 1]
        # _max = np.max(np.max(self.x, axis=1))
        # _min = np.min(np.min(self.x, axis=1))
        # self._mid = (_max + _min) / 2
        # self._range = (_max - _min) / 2

        _x = (self.x - self._mid) / self._range

        # Train model
        if self.class_weights is not None:
            self.model.fit(_x, self.y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose,
                           class_weight=self.class_weights)
        else:
            self.model.fit(_x, self.y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)

    def _predict(self, x):

        x = np.atleast_2d(x)

        # Scale input data by variable bounds
        _x = (x - self.l_b) / (self.u_b - self.l_b)

        # Normalise to [-1, 1]
        _x = (_x - self._mid) / self._range

        # Predict function values & re-scale
        # y = self.model.predict(_x)
        y = np.argmax(self._model.predict(_x), axis=-1)

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

        y = model.predict(_x)

        return y

    def _predict_variance(self, x):
        raise Exception('Variance prediction not implemented for FNN!')

    def update_cv_models(self):
        raise Exception('Variance prediction not implemented for FNN!')





