import numpy as np
from scipy import spatial
from scipy.stats import norm

class Surrogate(object):

    def __init__(self, n_dim, l_b, u_b, **kwargs):

        # Surrogate dimensions & bounds
        self.n_dim = n_dim
        self.l_b = np.array(l_b)
        self.u_b = np.array(u_b)

        # Training data
        self.n_pts = 0
        self.x = np.zeros((0, self.n_dim))
        self._x = np.zeros((0, self.n_dim))
        self.y = np.zeros(0)

        # Surrogate model instance
        self.model = None
        self.updated = False

        # Cross-validation
        self.cv_models = []
        self.cv_training_indices = []
        self.cv_k = 10
        self.cv_updated = False
        self.cv_training_y = np.zeros(self.n_pts)
        self.cv_error = np.zeros(self.n_pts)
        self.cv_training_mean = np.zeros(self.n_pts)
        self.cv_training_std = np.zeros(self.n_pts)
        self.cv_rmse = 1.0
        self.cv_mae = 1.0

    def reset(self):

        # Reset training data
        self.n_pts = 0
        self.x = np.zeros((0, self.n_dim))
        self._x = np.zeros((0, self.n_dim))
        self.y = np.zeros(0)
        self.updated = False

    def add_points(self, x, y):

        x = np.array(x)
        y = np.array(y)

        self.x = np.vstack((self.x, x))
        self.y = np.hstack((self.y, y))
        self._x = (self.x - self.l_b)/(self.u_b - self.l_b)
        self.n_pts = np.shape(self.y)[0]

        self.updated = False
        self.cv_updated = False

    def train(self):
        self._train()
        self.updated = True

    def predict(self, x):
        y = self._predict(x)
        return y

    def predict_variance(self, x):
        var = self._predict_variance(x)
        return var

    def cv_predict(self, model, model_y, x):
        y = self._cv_predict(model, model_y, x)
        return y

    def update_cv(self):
        self.update_cv_models()
        self.update_cv_error()
        self.cv_updated = True

    def update_cv_models(self):
        pass

    def update_cv_error(self, use_only_excluded_pts=False):

        if use_only_excluded_pts:
            # Compute the cross-validation surrogate predictions at each of the training points excluded for each of
            # the models
            self.cv_training_y = np.zeros(self.n_pts)
            self.cv_error = np.zeros(self.n_pts)
            k = 0
            for i, model in enumerate(self.cv_models):

                x_excluded = self.x[np.in1d(np.arange(self.n_pts), self.cv_training_indices[i]), :]
                y_excluded = self.y[np.in1d(np.arange(self.n_pts), self.cv_training_indices[i])]
                model_y = self.y[~np.in1d(np.arange(self.n_pts), self.cv_training_indices[i])]
                for j in range(len(self.cv_training_indices[i])):
                    self.cv_training_y[k+j] = self.cv_predict(model, model_y, x_excluded[j, :])
                    self.cv_error[k+j] = self.cv_training_y[k+j] - y_excluded[j]
                k += len(self.cv_training_indices[i])
        else:
            # Compute the cross-validation surrogate predictions at each of the training points
            self.cv_training_y = np.zeros((len(self.cv_models), self.n_pts))
            self.cv_error = np.zeros((len(self.cv_models), self.n_pts))
            for i, model in enumerate(self.cv_models):
                model_y = np.copy(self.y)
                for j in range(self.n_pts):
                    self.cv_training_y[i, j] = self.cv_predict(model, model_y, self._x[j, :])
                    self.cv_error[i, j] = self.cv_training_y[i, j] - self.y[j]

        # Compute the cross-validation root mean square error
        self.cv_rmse = np.sqrt((1.0/self.cv_training_y.size)*np.sum(self.cv_error**2.0))

        # Compute the cross-validation mean absolute error
        self.cv_mae = (1.0/self.cv_training_y.size)*np.sum(np.abs(self.cv_error))

        # Compute the mean of the cross-validation surrogate predictions at each of the training points
        self.cv_training_mean = np.mean(self.cv_training_y, axis=0)

        # Compute the mean of the cross-validation surrogate predictions at each of the training points
        self.cv_training_std = np.std(self.cv_training_y, axis=0)

    def predict_iqr(self, x):
        if not self.cv_updated:
            self.update_cv()

        iqr = self._predict_iqr(x)
        return iqr

    def predict_lcb(self, x, alpha, use_iqr=False):
        if not self.cv_updated:
            self.update_cv()

        lcb = self._predict_lcb(x, alpha=alpha, use_iqr=use_iqr)
        return lcb

    def predict_normalised_lcb(self, x, alpha):
        if not self.cv_updated:
            self.update_cv()

        lcb = self._predict_normalised_lcb(x, alpha=alpha)
        return lcb

    def predict_idw(self, x, delta):
        idw = self._predict_idw(x, delta=delta)
        return idw

    def predict_ei(self, x, ksi):
        if not self.cv_updated:
            self.update_cv()
        ei = self._predict_ei(x, ksi=ksi)
        return ei

    def _train(self):
        pass

    def _predict(self, x):
        pass

    def _predict_variance(self, x):
        pass

    def _cv_predict(self, model, model_y, x):
        pass

    def _predict_iqr(self, x):

        x = np.atleast_2d(x)

        # Compute surrogate predictions at x
        cv_z = np.zeros(len(self.cv_models))
        for i, model in enumerate(self.cv_models):
            cv_z[i] = model.predict(x)

        # Compute quantiles & IQR of eCDF
        cv_quantiles = np.quantile(a=cv_z, q=np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
        iqr = cv_quantiles[3] - cv_quantiles[1]

        return iqr

    def _predict_lcb(self, x, alpha=0.5, use_iqr=False):

        x = np.atleast_2d(x)

        # Compute surrogate predictions at x
        cv_z = np.zeros(len(self.cv_models))
        for i, model in enumerate(self.cv_models):
            cv_z[i] = model.predict(x)

        # Compute mean of surrogate predictions at x
        cv_mean = np.mean(cv_z)

        if use_iqr:
            # Compute quantiles & IQR of eCDF
            cv_quantiles = np.quantile(a=cv_z, q=np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
            cv_iqr = cv_quantiles[3] - cv_quantiles[1]

            # Compute experimental LCB using IQR
            cv_lcb = cv_mean - (alpha**0.5)*cv_iqr
        else:
            # Compute standard deviation of surrogate predictions at x
            cv_sigma = np.std(cv_z)

            # Compute experimental LCB
            # cv_lcb = cv_mean - (alpha**0.5)*cv_sigma
            cv_lcb = (1.0 - alpha)*cv_mean + alpha*cv_sigma

        return cv_lcb

    def _predict_normalised_lcb(self, x, alpha=0.5):

        """
        This method implements the adaptive acquisition function from Wang2020, modified slightly in that the mean
        and standard deviation at x are normalised by the maximum mean and standard deviation values at the each of the
        training points, predicted by the cross-validation models
        :param x:
        :param alpha:
        :return:
        """

        # Compute surrogate predictions at x
        cv_z = np.zeros(len(self.cv_models))
        for i, model in enumerate(self.cv_models):
            cv_z[i] = model.predict(x)

        # Compute mean of surrogate predictions at x
        cv_mean = np.mean(cv_z)

        # Compute standard deviation of surrogate predictions at x
        cv_sigma = np.std(cv_z)

        # Compute normalised experimental LCB acquisition function value
        cv_normalised_lcb = (1.0 - alpha)*(cv_mean/np.amax(self.cv_training_mean)) + alpha*(cv_sigma/np.amax(self.cv_training_std))

        return cv_normalised_lcb

    def _predict_idw(self, x, delta=2):

        x = np.atleast_2d(x)

        y = self.predict(x)

        y_max = np.max(self.y)
        y_min = np.min(self.y)

        scaled_objective = y / (y_max - y_min)

        idw = np.zeros(scaled_objective.shape)
        x = np.atleast_2d(x)
        dist = spatial.distance.cdist(x, self.x)
        if np.min(dist) == 0:
            z_function = 0
        else:
            weight_i = 1 / dist ** 2
            weight_sum = np.sum(weight_i)
            z_function = np.arctan(1 / weight_sum)
        idw = scaled_objective - delta * z_function

        return idw

    def _predict_ei(self, x, ksi=0.01, tol=1e-5, toldist=1e-3):
        # Uncertainty
        x = np.atleast_2d(x)
        if self.x.shape[1] < 2:
            dist, _ = compute_distance(x, self.x)
        else:
            dist = spatial.distance.cdist(x, self.x)

        y_min = np.min(self.y)
        mu = self.predict(x)
        try:
            sigma = np.sqrt(self._predict_variance(x))
        except:
            #use cross-validation insteady
            cv_z = np.zeros(len(self.cv_models))
            for i, model in enumerate(self.cv_models):
                cv_z[i] = model.predict(x)

            # Compute mean of surrogate predictions at x
            sigma = np.std(cv_z)

        # calculate the value of expected improvement
        # Exploitation-exploration trade-off parameter. - high ksi means more exploration
        # ksi = 0.01
        # http://krasserm.github.io/2018/03/21/bayesian-optimization/
        improvement = y_min - mu - ksi * np.abs(y_min)
        # assume a normal (gaussian) distribution
        Z = improvement / sigma
        EI = -mu + improvement * norm.cdf(Z, loc=mu, scale=sigma) + sigma * norm.pdf(Z, loc=mu, scale=sigma)
        # EI = improvement * norm.cdf(Z, loc=mu, scale=sigma) + sigma * norm.pdf(Z, loc=mu, scale=sigma)

        if sigma < np.sqrt(tol):
            EI = 0.0
        #
        if np.any(dist < toldist):
            EI *= np.min(dist)

        return EI

def compute_distance(x, x_sample, power=1, tol=1e-3, method="min"):
    """
    :param x: selected point
    :param x_sample: surrogate sampled points
    :param power: specifies distance function (i.e. power=2: euclidean)
    :param tol: placeholder not used
    :param method: "max", "min" (default)
    :return: smallest distance between x and points in x_sample
    """
    dist = np.zeros(len(x_sample))
    for i in range(len(x_sample)):
        dist[i] = abs(x - x_sample[i])**(1/power)

    if method == "min":
        # Find minimum distance and position between x and x_sample
        ind = np.argmin(dist)
        x = x_sample[ind]
        d = dist[ind]
    elif method == "max":
        ind = np.argmax(dist)
        x = x_sample[ind]
        d = dist[ind]
    else:
        d = 0

    return d, x


if __name__ == '__main__':

    n_dim = 4
    l_b = np.zeros(n_dim)
    u_b = np.ones(n_dim)

    test_surrogate = Surrogate(4, l_b, u_b)
    null = 0


