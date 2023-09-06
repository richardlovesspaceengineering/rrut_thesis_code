import numpy as np

from smt.surrogate_models import KPLS

from optimisation.model.surrogate import Surrogate


class KPLSRegression(Surrogate):
    """
    Partial Least Squares Kriging: https://smt.readthedocs.io/en/latest/_src_docs/surrogate_models/kpls.html?highlight=kpls
    """
    def __init__(self, n_dim, l_b, u_b, theta0=1e-2, cov_type='squar_exp', p_type='constant', n_lml_opt_restarts=5, **kwargs):

        super().__init__(n_dim=n_dim, l_b=l_b, u_b=u_b, **kwargs)

        # Covariance function
        if cov_type not in ['abs_exp', 'squar_exp']:
            raise Exception('Undefined covariance function')

        # Regression function type
        if p_type not in ['constant', 'linear', 'quadratic']:
            raise Exception('Undefined regression function')

        # Model instance (SMT)
        self.n_lml_opt_restarts = n_lml_opt_restarts
        self.model = KPLS(theta0=theta0*np.ones(1),  # TODO: how to pass n_obj here for theta0
                          corr=cov_type,
                          poly=p_type,
                          n_start=self.n_lml_opt_restarts,
                          print_global=False)
        # Uses COBYLA optimiser as default, can select from: ['Cobyla', 'TNC']

        self._mu = 0.0
        self._sigma = 0.0

    def _train(self):

        # Compute mean and std of training function values
        self._mu = np.mean(self.y)
        self._sigma = max([np.std(self.y), 1e-6])

        # Normalise function values
        _y = (self.y - self._mu)/self._sigma

        # Scale training data by variable bounds
        self._x = (self.x - self.l_b)/(self.u_b - self.l_b)

        # Train model
        self.model.set_training_values(self._x, _y)
        self.model.train()

    def _predict(self, x):

        x = np.atleast_2d(x)

        # Scale input data by variable bounds
        _x = (x - self.l_b)/(self.u_b - self.l_b)

        # Predict function values & re-scale
        y = self._mu + self._sigma*self.model.predict_values(_x)

        return y

    def _predict_variance(self, x):

        # Scale input data by variable bounds
        _x = (x - self.l_b)/(self.u_b - self.l_b)

        # Predict standard deviation & re-scale
        std = self.model.predict_variances(_x)
        std *= self._sigma

        return std**2.0



