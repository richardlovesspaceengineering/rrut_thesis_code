import numpy as np


from optimisation.model.surrogate import Surrogate
from optimisation.surrogate.models.rbf import RBF

class EnsembleSurrogate(Surrogate):
    def __init__(self, n_dim, l_b, u_b, c=0.5, p_type='linear', kernel_types=None, **kwargs):

        super().__init__(n_dim=n_dim, l_b=l_b, u_b=u_b, **kwargs)

        # Todo: Implement surrogates for each model in models (models is a list of models)
        # self.models = SurrogateHandler(models)
        # TODO: expand beyond RBF's

        if kernel_types is None:
            self.kernel_types = ["gaussian", "cubic", "multiquadratic", "inverse_multiquadratic",
                       "matern_32", "matern_52",  # "exp",#"periodic",
                       "matern_12"]
        else:
            self.kernel_types = kernel_types
        self.c = c
        self.p_type = p_type

        # set up ensemble
        k = len(self.kernel_types)
        ensemble = [RBF(n_dim=n_dim, c=self.c, p_type=self.p_type, kernel_type=self.kernel_types[j]) for j in range(k)]

        self.model = ensemble


    def _train(self):

        # Compute mean and std of training function values
        self._mu = np.mean(self.y)
        self._sigma = max([np.std(self.y), 1e-6])

        # Scale training data by variable bounds
        self._x = (self.x - self.l_b)/(self.u_b - self.l_b)

        for i, model in enumerate(self.model):
            if model.p_type is None:
                # Normalise function values
                _y = self.y - self._mu

                # Train model
                model.fit(self._x, _y)
            else:
                # Train model
                model.fit(self._x, self.y)

    def _predict(self, x):
        y = np.zeros(len(self.model))
        _x = (x - self.l_b) / (self.u_b - self.l_b)

        for i, model in enumerate(self.model):
            if model.p_type is None:
                # Train model
                y[i] = model.predict(_x) + self._mu
            else:
                # Train model
                y[i] = model.predict(_x)

        y_mean = np.mean(y)
        return y_mean

    def _predict_model(self, x, model):
        x = np.atleast_2d(x)

        if len(x.shape) > 2:
            x_new = x.reshape((x.shape[0] * x.shape[1], x.shape[2]))
        else:
            x_new = x

        # Scale input data by variable bounds
        _x = (x_new - self.l_b) / (self.u_b - self.l_b)

        # Predict function values & re-scale
        # Predict function values
        y = np.zeros((_x.shape[0],1))
        for cntr in range(_x.shape[0]):
            _xtemp = _x[cntr,:]
            if model.p_type is None:
                y[cntr] = model.predict(_xtemp) + self._mu
            else:
                y[cntr] = model.predict(_xtemp)

        if len(x.shape) > 2:
            y = y.reshape(x.shape[0], x.shape[1])

        if x.shape[0] == 1:
            return y[0]
        else:
            return y

    def _cv_predict(self, model, model_y, x):

        # Scale input data by variable bounds
        _x = (x - self.l_b) / (self.u_b - self.l_b)

        # Predict function values
        if model.p_type is None:
            _mu = np.mean(model_y)
            y = model.predict(_x) + _mu
        else:
            y = model.predict(_x)

        return y

    def _cv_predict_ensemble(self, x):
        y = np.zeros(len(self.model))
        for i, model in enumerate(self.model):
            y[i] = self._predict_model(x, model)

        y_mean = np.mean(y)
        y_std = np.std(y)
        return y_mean, y_std, y

    def _predict_variance(self, x):
        y = np.zeros(len(self.model))
        for i, model in enumerate(self.model):
            y[i] = self._predict_model(x, model)

        y_std = np.std(y)
        if model.p_type is None:
            y_std *= self._sigma
        return y_std**2

    def update_cv_models(self):

        self.cv_models = self.model

        # Training each of the cross-validation models
        for i, model in enumerate(self.cv_models):
            model.fit(self._x, self.y)


