import numpy as np
from scipy import spatial
from scipy.stats import norm

from optimisation.util.misc import bilog_transform


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

        # self.use_bilog = False

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

        # if self.use_ks:
        #     y = self._predict_ks(x, y)

        # if self.use_bilog:
        #     y = bilog_transform(y)

        self.x = np.vstack((self.x, x))
        self.y = np.hstack((self.y, y))
        self._x = (self.x - self.l_b) / (self.u_b - self.l_b)
        self.n_pts = np.shape(self.y)[0]

        self.updated = False
        self.cv_updated = False

    def train(self):
        self._train()
        self.updated = True

    def predict(self, x):
        y = self._predict(x)
        return y

    def predict_proba(self, x):
        y = self._predict_proba(x)
        return y

    def predict_model(self, x, idx):
        y = self._predict_model(x, idx)
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

    def predict_lcb(self, x, alpha, flag="cv_models"): # "iqr"
        if not self.cv_updated:
            self.update_cv()

        lcb = self._predict_lcb(x, alpha=alpha, flag=flag)
        return lcb

    def predict_lcb_idw(self, x, alpha): # "iqr"
        if not self.cv_updated:
            self.update_cv()

        lcb = self._predict_lcb_idw(x, alpha=alpha)
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

    def predict_ei_idw(self, x, ksi):
        if not self.cv_updated:
            self.update_cv()
        ei = self._predict_ei_idw(x, ksi=ksi)
        return ei

    def predict_wee(self, x, weight):
        if not self.cv_updated:
            self.update_cv()
        wee = self._predict_wee(x, weight=weight)
        return wee

    def predict_wb2s(self, x, scale):
        if not self.cv_updated:
            self.update_cv()
        wb2s = self._predict_wb2s(x, scale)
        return wb2s

    def predict_mu(self, x):
        if not self.cv_updated:
            self.update_cv()
        mu = self._predict_mu(x)
        return mu

    def predict_idw_dist(self, x):
        idw_distance = self._predict_idw_dist(x)
        return idw_distance

    def predict_wei(self, x, weight):
        if not self.cv_updated:
            self.update_cv()
        wei = self._predict_wei(x, weight=weight)
        return wei

    def predict_pof(self, x, obj_val, k):
        if not self.cv_updated:
            self.update_cv()
        pof = self._predict_pof(x, k=k)
        return pof*obj_val

    def predict_pof_isc(self, x, obj_val, k):
        if not self.cv_updated:
            self.update_cv()
        pof = self._predict_pof(x, k=k)
        isc = self._predict_ISC14(x)

        return isc*pof*obj_val

    def predict_pof_if(self, x, obj_val, opt, k):
        if not self.cv_updated:
            self.update_cv()
        pof = self._predict_pof(x, k=k)
        if_d = self._predict_influence_function(x, opt)

        return if_d*pof*obj_val

    def predict_if(self, x, theta, scale=0.1):
        # if not self.cv_updated:
        #     self.update_cv()
        temp_ind = np.argmin(spatial.distance.cdist(np.atleast_2d(x), self.x))
        x2 = self.x[temp_ind, :]
        IF = self.corr_function(x, x2, theta=theta)

        y_max = np.max(self.y)
        y_min = np.min(self.y)
        dF = y_max - y_min

        return scale*IF*dF

    def predict_ev(self, x, Pev, weight):
        if not self.cv_updated:
            self.update_cv()
        ev = self._predict_ev(x, Pev=Pev, weight=weight)
        return ev

    def predict_conslcb(self, x, alpha):
        if not self.cv_updated:
            self.update_cv()
        conslcb = self._predict_conslcb(x, alpha=alpha)
        return conslcb

    def predict_conswee(self, x, weight):
        if not self.cv_updated:
            self.update_cv()
        wee = self._predict_conswee(x, weight=weight)
        return wee

    def _train(self):
        pass

    def _predict(self, x):
        pass

    def _predict_proba(self, x):
        pass

    def _predict_model(self, x, idx):
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

    def _predict_lcb(self, x, alpha=0.5, flag=False):
        # Mean prediction
        x = np.atleast_2d(x)
        mu = self.predict(x)

        # Obtain uncertainty
        try:
            sigma = np.sqrt(self._predict_variance(x))
        except:
            if flag == 'cv_models':
                #use cross-validation insteady
                cv_z = np.zeros(len(self.cv_models))
                for i, model in enumerate(self.cv_models):
                    cv_z[i] = model.predict(x)

                # Compute mean of surrogate predictions at x
                sigma = np.std(cv_z)

            elif flag == 'iqr':
                # Compute quantiles & IQR of eCDF
                cv_quantiles = np.quantile(a=cv_z, q=np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
                sigma = cv_quantiles[3] - cv_quantiles[1]

        # LCB
        lcb = mu - alpha*sigma

        # Compute surrogate predictions at x
        #cv_z = np.zeros(len(self.cv_models))
        #for i, model in enumerate(self.cv_models):
        #    cv_z[i] = model.predict(x)
            # print(cv_z[i])

        # Compute mean of surrogate predictions at x
        #cv_mean = np.mean(cv_z)

        #if use_iqr:
            # Compute quantiles & IQR of eCDF
        #    cv_quantiles = np.quantile(a=cv_z, q=np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
        #    cv_iqr = cv_quantiles[3] - cv_quantiles[1]

            # Compute experimental LCB using IQR
        #    cv_lcb = cv_mean - (alpha**0.5)*cv_iqr
        #else:
            # Compute standard deviation of surrogate predictions at x
        #    cv_sigma = np.std(cv_z)

            # Compute experimental LCB
            # cv_lcb = cv_mean - (alpha**0.5)*cv_sigma
        #    cv_lcb = (1.0 - alpha)*cv_mean + alpha*cv_sigma

        return lcb

    def _predict_lcb_idw(self, x, alpha=0.5):
        # Mean prediction
        x = np.atleast_2d(x)
        mu = self.predict(x)

        # Obtain uncertainty
        # use idw function for uncertainty
        sigma = self._predict_idw_dist(x)

        # LCB
        lcb = mu - alpha*sigma

        return lcb

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
        z_function = self._predict_idw_dist(x)

        idw = np.zeros(scaled_objective.shape)
        idw = scaled_objective - delta * z_function

        return idw

    def _predict_ei_idw(self, x, ksi=0.01, tol=1e-5, toldist=1e-3):
        # Uncertainty
        x = np.atleast_2d(x)
        if self.x.shape[1] < 2:
            dist, _ = compute_distance(x, self.x)
        else:
            dist = spatial.distance.cdist(x, self.x)

        y_min = np.min(self.y)
        mu = self.predict(x)

        # use idw function for uncertainty
        sigma = self._predict_idw_dist(x)

        # calculate the value of expected improvement
        # Exploitation-exploration trade-off parameter. - high ksi means more exploration
        # ksi = 0.01
        # http://krasserm.github.io/2018/03/21/bayesian-optimization/
        improvement = y_min - mu - ksi * np.abs(y_min)
        # assume a normal (gaussian) distribution
        Z = improvement / sigma
        # EI = -mu + improvement * norm.cdf(Z, loc=mu, scale=sigma) + sigma * norm.pdf(Z, loc=mu, scale=sigma)
        #EI = -mu + improvement * norm.cdf(Z, loc=mu) + sigma * norm.pdf(Z, loc=mu)
        EI = improvement * norm.cdf(Z, loc=mu, scale=sigma) + sigma * norm.pdf(Z, loc=mu, scale=sigma)

        if sigma < np.sqrt(tol):
            EI = 0.0
        #
        if np.any(dist < toldist):
            EI *= np.min(dist)

        return -EI

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
        # EI = -mu + improvement * norm.cdf(Z, loc=mu, scale=sigma) + sigma * norm.pdf(Z, loc=mu, scale=sigma)
        #EI = -mu + improvement * norm.cdf(Z, loc=mu) + sigma * norm.pdf(Z, loc=mu)
        EI = improvement * norm.cdf(Z, loc=mu, scale=sigma) + sigma * norm.pdf(Z, loc=mu, scale=sigma)

        if sigma < np.sqrt(tol):
            EI = 0.0
        #
        if np.any(dist < toldist):
            EI *= np.min(dist)

        return -EI

    def _predict_wee(self, x, weight):

        # Return ensemble predictions
        x = np.atleast_2d(x)
        y = self._predict_median(x)

        # Scaled median and std
        f_max = np.max(self.y)
        f_min = np.min(self.y)
        f_scale = y / (f_max - f_min)
        f_median = np.median(f_scale)
        # sigma = np.std(f_scale)

        ## Try using IDW distance measure
        sigma = self._predict_idw_dist(x)

        # Modified Weighted Exploration Exploitation (MWEE)
        v = (f_max - f_min) * (weight*f_median - (1 - weight)*sigma)

        return v

    def _predict_wb2s(self, x, scale, beta=100):

        # Scaling term by beta if non-zero
        if scale != 0:
            s = beta*scale
        else:
            s = 1

        # Mean prediction & EI
        mu = self.predict(x)
        EI = _predict_ei(x)

        # WB2S = sEI - mu
        wb2s = s*EI - mu

        return wb2s

    def _predict_mu(self, x):

        # Purely exploitation based off mean prediction
        mu = self.predict(x)

        return mu

    def _predict_idw_dist(self, x):

        # Compute IDW distance
        x = np.atleast_2d(x)
        dist = spatial.distance.cdist(x, self.x)
        if np.min(dist) == 0:
            z_function = 0
        else:
            weight_i = 1 / dist ** 2
            weight_sum = np.sum(weight_i)
            z_function = np.arctan(1 / weight_sum)

        return z_function

    def _predict_wei(self, x, weight, tol=1e-5, toldist=1e-3):
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
            # use cross-validation insteady
            cv_z = np.zeros(len(self.cv_models))
            for i, model in enumerate(self.cv_models):
                cv_z[i] = model.predict(x)

            # Compute mean of surrogate predictions at x
            sigma = np.std(cv_z)

        # calculate the value of expected improvement
        improvement = y_min - mu

        # assume a normal (gaussian) distribution
        Z = improvement / sigma
        WEI = weight * improvement * norm.cdf(Z, loc=mu, scale=sigma) + (1 - weight) * sigma * norm.pdf(Z, loc=mu, scale=sigma) # More exploratory
        # WEI = weight * improvement * norm.cdf(Z, loc=mu) + (1 - weight) * sigma * norm.pdf(Z, loc=mu) # More aggressive

        if sigma < np.sqrt(tol):
            WEI = 0.0
        if np.any(dist < toldist):
            WEI *= np.min(dist)

        return -WEI

    def _predict_pof(self, x, k=1.5):

        # Predict constraint mean value and uncertainty
        g_mu = self.predict(x)

        # Scale
        g_mu / (np.max(self.y) - np.min(self.y))

        try:
            g_sigma = np.sqrt(self._predict_variance(x))
        except:
            # use cross-validation insteady
            cv_z = np.zeros(len(self.cv_models))
            for i, model in enumerate(self.cv_models):
                cv_z[i] = model.predict(x)

            # Compute mean of surrogate predictions at x
            g_sigma = np.std(cv_z)

        # Violation term
        violation = -g_mu / g_sigma

        # Calculate PF
        p = 2 * k - 2
        if violation >= 0:
            pof = p * (1 - norm.cdf(violation, loc=g_mu, scale=g_sigma)) + 1
        else:
            pof = 2 * k * norm.cdf(2 * k * violation, loc=g_mu, scale=g_sigma)

        return pof

    def _predict_ks(self, x, y, rho=50):

        # Take the bilog of constraints
        y = bilog_transform(y)

        if np.ndim(x) == 1:
            term = 0
            for n_c in range(len(y)):
                term += np.exp(rho * y[n_c])
            ks_sample = np.array((1 / rho) * np.log(term))
        else:
            ks_sample = np.zeros(len(x))
            for i in range(len(x)):
                term = 0
                for n_c in range(len(y[0, :])):
                    term += np.exp(rho * y[i, n_c])
                ks_sample[i] = (1 / rho) * np.log(term)

        return ks_sample

    def _predict_influence_function(self, x, opt):

        # Pseudo-optimum
        var_opt = opt.var

        # Correlation function about pseudo-opt location
        R_corr = self.corr_function(x, var_opt)

        # Influence Function
        IF = 1 - R_corr
        return IF

    def corr_function(self, x, f_opt_var, theta=10):
        r = 1
        for i in range(len(x)):
            r *= np.exp(-theta * np.abs(x[i] - f_opt_var[i]) ** 2)
        return r

    def _predict_ISC14(self, x):
        # Feasible points
        feas_mask = self.y <= 0
        x_feas = self.x[feas_mask]
        if ~feas_mask.all():
            min_viol = np.argmin(self.y)
            x_feas = self.x[min_viol]

        # Compute distances
        x = np.atleast_2d(x)
        x_feas = np.atleast_2d(x_feas)
        dist = spatial.distance.cdist(x, x_feas)

        # Minimum distance & normalise
        isc_d = np.min(dist) / np.mean(self.u_b - self.l_b)

        return isc_d

    def _predict_ev(self, x, Pev=0.001, weight=0.5):
        # Uncertainty
        x = np.atleast_2d(x)
        if self.x.shape[1] < 2:
            dist, _ = compute_distance(x, self.x)
        else:
            dist = spatial.distance.cdist(x, self.x)

        mu_g = self.predict(x)
        try:
            sigma = np.sqrt(self._predict_variance(x))
        except:
            #use cross-validation insteady
            cv_z = np.zeros(len(self.cv_models))
            for i, model in enumerate(self.cv_models):
                cv_z[i] = model.predict(x)

            # Compute mean of surrogate predictions at x
            sigma = np.std(cv_z)

        # calculate the value of expected violation (EV)
        Z = mu_g / sigma
        EV = -weight * mu_g * norm.cdf(Z, loc=mu_g, scale=sigma) + (1 - weight) * sigma * norm.pdf(Z, loc=mu_g, scale=sigma)

        return -EV

    def _predict_conslcb(self, x, alpha=1.96):
        # Uncertainty
        x = np.atleast_2d(x)
        if self.x.shape[1] < 2:
            dist, _ = compute_distance(x, self.x)
        else:
            dist = spatial.distance.cdist(x, self.x)

        mu_g = self.predict(x)
        try:
            sigma = np.sqrt(self._predict_variance(x))
        except:
            # use cross-validation insteady
            cv_z = np.zeros(len(self.cv_models))
            for i, model in enumerate(self.cv_models):
                cv_z[i] = model.predict(x)

            # Compute mean of surrogate predictions at x
            sigma = np.std(cv_z)

        # calculate the cons LCB
        cons_lcb = mu_g - alpha*sigma
        return cons_lcb

    def _predict_conswee(self, x, weight, k=1.1):

        # Return ensemble predictions
        x = np.atleast_2d(x)
        y = self._predict_median(x)

        # Scaled median and std
        f_max = np.max(self.y)
        f_min = np.min(self.y)
        f_scale = y / (f_max - f_min)
        f_median = np.median(f_scale)
        # sigma = np.std(f_scale)

        ## Try using IDW distance measure
        sigma = self._predict_idw_dist(x)

        # PoF bias factor around boundary
        bias = k*np.exp(-f_median**2) + 1

        # Modified Weighted Exploration Exploitation (MWEE)
        v = (f_max - f_min) * (weight*bias*f_median - (1 - weight)*sigma)

        return v


def bilog_transform(obj, beta=1):
    bilog_obj = np.zeros(np.shape(obj))

    for i in range(len(obj)):
        bilog_obj[i] = np.sign(obj[i])*np.log(beta + np.abs(obj[i]))

    return bilog_obj



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


