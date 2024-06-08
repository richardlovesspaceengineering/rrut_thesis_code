import numpy as np
from scipy import spatial

from optimisation.model.surrogate import Surrogate


class RadialBasisFunctions(Surrogate):

    def __init__(self, n_dim, l_b, u_b, c=0.5, p_type=None, kernel_type='gaussian', **kwargs):

        super().__init__(n_dim=n_dim, l_b=l_b, u_b=u_b, **kwargs)

        self.c = c
        self.model = RBF(n_dim=self.n_dim, c=self.c, p_type=p_type, kernel_type=kernel_type)

        self._mu = 0.0
        self._sigma = 0.0

    def _train(self):

        # Compute mean and std of training function values
        self._mu = np.mean(self.y)
        self._sigma = max([np.std(self.y), 1e-6])

        # Scale training data by variable bounds
        self._x = (self.x - self.l_b) / (self.u_b - self.l_b)

        if self.model.p_type is None:
            # Normalise function values
            _y = self.y - self._mu

            # Train model
            self.model.fit(self._x, _y)
        else:
            # Train model
            self.model.fit(self._x, self.y)

    def _predict(self, x):

        x = np.atleast_2d(x)

        if len(x.shape) > 2:
            x_new = x.reshape((x.shape[0] * x.shape[1], x.shape[2]))
        else:
            x_new = x

        # Scale input data by variable bounds
        _x = (x_new - self.l_b) / (self.u_b - self.l_b)

        # Predict function values & re-scale
        # Predict function values
        y = np.zeros((_x.shape[0], 1))
        for cntr in range(_x.shape[0]):
            _xtemp = _x[cntr, :]
            if self.model.p_type is None:
                y[cntr] = self.model.predict(_xtemp) + self._mu
            else:
                y[cntr] = self.model.predict(_xtemp)

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

    def _predict_variance(self, x):
        raise Exception('Variance prediction not implemented for RBF')

    def update_cv_models(self):

        # k-fold LSO cross-validation indices
        random_state = np.random.default_rng()
        self.cv_training_indices = np.array_split(random_state.choice(self.n_pts, size=self.n_pts, replace=False),
                                                  self.cv_k)
        self.cv_models = [RBF(n_dim=self.n_dim, c=self.c, p_type='linear') for _ in range(self.cv_k)]

        # Training each of the cross-validation models
        for i, model in enumerate(self.cv_models):
            model.fit(self._x[~np.in1d(np.arange(self.n_pts), self.cv_training_indices[i]), :],
                      self.y[~np.in1d(np.arange(self.n_pts), self.cv_training_indices[i])])


class RBF(object):
    def __init__(self, n_dim, c=0.5, p_type=None, kernel_type='gaussian'):

        self.n_dim = n_dim

        self.psi = np.zeros((0, 0))
        self.w = np.zeros(0)

        self.c = c
        self.kernel_type = kernel_type

        self.p_type = p_type
        if self.p_type is not None:
            if self.p_type == 'constant':
                self.n_p = 1
            elif self.p_type == 'linear':
                self.n_p = self.n_dim + 1
            else:
                raise Exception('Undefined p_type')
            self.w_p = np.zeros((self.n_p, 1))
            self.g = np.zeros((0, self.n_p))
        else:
            self.w_p = None
            self.g = None

        self.x = np.zeros((0, self.n_dim))
        self.y = np.zeros(self.n_dim)

    def fit(self, x, y):

        self.x = np.array(x)
        self.y = np.array(y)

        self.psi = np.zeros((len(x), len(x)))
        self.w = np.zeros(len(y))

        # Computing matrix of RBF values
        dist = spatial.distance.cdist(self.x, self.x)
        if self.kernel_type.lower() == 'gaussian':
            self.psi = self.gaussian_basis(dist)
        elif self.kernel_type.lower() == 'cubic':
            self.psi = self.cubic_basis(dist)
        elif self.kernel_type.lower() == 'mod_cubic':
            self.psi = self.modified_cubic_basis(dist)
        elif self.kernel_type.lower() == 'linear':
            self.psi = self.linear_basis(dist)
        elif self.kernel_type.lower() == 'tps':
            self.psi = self.thin_plate_spline_basis(dist)
        elif self.kernel_type.lower() == 'multiquadratic':
            self.psi = self.multiquadratic_basis(dist)
        elif self.kernel_type.lower() == 'inv_multiquadratic':
            self.psi = self.inverse_multiquadratic_basis(dist)
        elif self.kernel_type.lower() == 'matern_52':
            self.psi = self.matern5_2_basis(dist)
        elif self.kernel_type.lower() == 'matern_32':
            self.psi = self.matern3_2_basis(dist)
        elif self.kernel_type.lower() == 'matern_12':
            self.psi = self.matern1_2_basis(dist)
        elif self.kernel_type.lower() == 'exp':
            self.psi = self.exponential_basis(dist)
        elif self.kernel_type.lower() == 'periodic':
            self.psi = self.periodic_basis(dist)
        elif self.kernel_type.lower() == 'logistic':
            self.psi = self.logistic_basis(dist)
        elif self.kernel_type.lower() == 'hyperbolic_tangent_sigmoid':
            self.psi = self.hyperbolic_tangent_sigmoid_basis(dist)
        elif self.kernel_type.lower() == 'cauchy':
            self.psi = self.cauchy_basis(dist)
        else:
            self.psi = self.gaussian_basis(dist)

        if self.p_type is None:
            # Solve the linear system to obtain weights
            self.w = np.linalg.solve(self.psi, self.y)
        else:
            if self.p_type == 'constant':
                self.g = np.ones((len(self.x), self.n_p))
            elif self.p_type == 'linear':
                self.g = np.hstack((np.ones((len(self.x), 1)), self.x))
            else:
                raise Exception('Undefined p_type')

            # Form full LHS matrix
            a_upper = np.hstack((self.psi, self.g))
            a_lower = np.hstack((self.g.transpose(), np.zeros((self.n_p, self.n_p))))
            a = np.vstack((a_upper, a_lower))

            # Form RHS vector
            b = np.concatenate((self.y, np.zeros(self.n_p)))

            # Solve linear system to obtain weights
            x = np.linalg.solve(a, b)

            # Extract RBF & linear weights/coefficients
            self.w = x[:len(self.x)]
            self.w_p = x[len(self.x):]

    def predict(self, x):
        x = np.array(x)

        # Compute influence matrix of RBF from training points to sample point
        dist = spatial.distance.cdist(np.atleast_2d(x), self.x)
        if self.kernel_type.lower() == 'gaussian':
            psi = self.gaussian_basis(dist)
        elif self.kernel_type.lower() == 'cubic':
            psi = self.cubic_basis(dist)
        elif self.kernel_type.lower() == 'mod_cubic':
            psi = self.modified_cubic_basis(dist)
        elif self.kernel_type.lower() == 'linear':
            psi = self.linear_basis(dist)
        elif self.kernel_type.lower() == 'tps':
            psi = self.thin_plate_spline_basis(dist)
        elif self.kernel_type.lower() == 'multiquadratic':
            psi = self.multiquadratic_basis(dist)
        elif self.kernel_type.lower() == 'inv_multiquadratic':
            psi = self.inverse_multiquadratic_basis(dist)
        elif self.kernel_type.lower() == 'matern_52':
            psi = self.matern5_2_basis(dist)
        elif self.kernel_type.lower() == 'matern_32':
            psi = self.matern3_2_basis(dist)
        elif self.kernel_type.lower() == 'matern_12':
            psi = self.matern1_2_basis(dist)
        elif self.kernel_type.lower() == 'exp':
            psi = self.exponential_basis(dist)
        elif self.kernel_type.lower() == 'periodic':
            psi = self.periodic_basis(dist)
        elif self.kernel_type.lower() == 'logistic':
            psi = self.logistic_basis(dist)
        elif self.kernel_type.lower() == 'hyperbolic_tangent_sigmoid':
            psi = self.hyperbolic_tangent_sigmoid_basis(dist)
        elif self.kernel_type.lower() == 'cauchy':
            psi = self.cauchy_basis(dist)
        else:
            psi = self.gaussian_basis(dist)

        # Compute function output at sample point
        if self.p_type is None:
            y = np.dot(psi, self.w)
        else:
            # Compute global polynomial value
            if self.p_type == 'constant':
                g = np.ones((1, self.n_p))
            elif self.p_type == 'linear':
                g = np.hstack((np.ones((1, 1)), np.atleast_2d(x)))
            else:
                raise Exception('Undefined p_type')

            y = np.dot(psi, self.w) + np.dot(g, self.w_p)

        return y

    def gaussian_basis(self, dist):
        # Gaussian kernel
        return np.exp(-self.c * dist ** 2.0)

    def linear_basis(self, dist):
        # Linear kernel
        return dist

    def cubic_basis(self, dist):
        # Cubic kernel
        return dist ** 3.0

    def modified_cubic_basis(self, dist):
        return (self.c * dist) ** 3.0

    def thin_plate_spline_basis(self, dist):
        # Thin plate spline kernel
        phi = dist ** 2.0 * np.log(self.c * dist, where=dist != 0)
        return np.nan_to_num(phi)

    def multiquadratic_basis(self, dist):
        # Multiquadratic kernel
        return np.sqrt(dist ** 2.0 + self.c ** 2.0)

    def inverse_multiquadratic_basis(self, dist):
        # Inverse multiquadratic kernel
        return 1.0 / np.sqrt(dist ** 2.0 + self.c ** 2.0)

    def matern3_2_basis(self, dist):
        # Matern 3/2 kernel
        exp_component = np.exp(-np.sqrt(1.5 * 2) * dist)
        constant_component = (np.sqrt(3) * dist)
        constant_component += 1

        return constant_component * exp_component

    def matern5_2_basis(self, dist):
        exp_component = np.exp(-np.sqrt(2.5 * 2) * dist)
        constant_component = (np.sqrt(5) * dist)
        constant_component += 1
        constant_component += (5.0 / 3.0 * dist ** 2)

        return constant_component * exp_component

    def matern1_2_basis(self, dist):
        exp_component = np.exp(-np.sqrt(0.5 * 2) * dist)
        constant_component = 1
        return constant_component * exp_component

    def exponential_basis(self, dist):
        return np.exp(-self.c * dist)

    def periodic_basis(self, dist):
        # http://proceedings.mlr.press/v28/wilson13.pdf eq 15

        phi = np.exp(-2 * (np.sin(2 * np.pi * self.c)) ** 2 / dist ** 2, where=dist != 0)
        return np.nan_to_num(phi)

    def logistic_basis(self, dist):
        # ANN Transfer Function : "An efficient multi-objective optimization method based on the adaptive approximation model of the radial basis function"

        return 1 / (1 + np.exp(-self.c * dist))

    def hyperbolic_tangent_sigmoid_basis(self, dist):
        # ANN Transfer Function : "A Critical Review of Surrogate Assisted Robust Design Optimization"

        phi = (np.exp(dist) - np.exp(-dist)) / (np.exp(dist) + np.exp(-dist))
        return phi

    def cauchy_basis(self, dist):
        # ANN Transfer Function : "A Critical Review of Surrogate Assisted Robust Design Optimization"

        return 1.0 / (dist ** 2.0 + self.c ** 2.0)