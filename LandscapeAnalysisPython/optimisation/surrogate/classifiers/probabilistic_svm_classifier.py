import copy
import warnings

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.svm import SVC
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, STATUS_FAIL

from optimisation.model.surrogate import Surrogate


class PSVMClassification(Surrogate):

    def __init__(self, n_dim, l_b, u_b, c=1.0, gamma='scale', kernel='rbf', **kwargs):

        super().__init__(n_dim=n_dim, l_b=l_b, u_b=u_b, **kwargs)

        # SVM Parameters
        self.c = c
        self.gamma = gamma
        self.kernel = kernel
        self.model = SVC(C=self.c, kernel=self.kernel, gamma=self.gamma, probability=False, class_weight='balanced')

        # Modified Probability Sigmoid model parameters
        self.A = -1.0
        self.B = -1.0
        self.d_minus = 0.0
        self.d_plus = 0.0
        self.tau = 1e-10

        # Hyoperopt tuning
        self.max_evals = 300
        self.verbose = False
        self.deci = None
        self.label = None
        self.prior1 = None
        self.prior0 = None
        self.t = None

    def _train(self):
        # Train model
        self.model.fit(self._x, self.y)

        # Tune A and B parameters
        s = self.model.decision_function(self._x)
        n_plus = np.count_nonzero(self.y == 1)  # NOTE: 1 = feasible, 0 = infeasible
        n_neg = len(self.y) - n_plus

        # Maximum likelihood - line-search
        # self.maximum_likelihood_optimisation(s, self.y, n_plus, n_neg)

        # Maximum likelihood - hyperopt
        self.hyperopt_tuning_params(s, self.y, n_plus, n_neg)

    def _predict(self, x):
        x = np.atleast_2d(x)

        # Scale input data by variable bounds
        _x = (x - self.l_b) / (self.u_b - self.l_b)

        # Predict function values & re-scale
        y = self.model.predict(_x)

        return y

    def _predict_proba(self, x):
        x = np.atleast_2d(x)

        # Calculate sigmoid probability function
        y = self.probability_sigmoid_function(x)

        return y

    def _cv_predict(self, model, model_y, x):
        raise Exception('Cross-validation probability prediction not implemented for P-SVM!')

    def _predict_variance(self, x):
        raise Exception('Variance prediction not implemented for P-SVM!')

    def update_cv_models(self):
        pass

    def probability_sigmoid_function(self, x):
        # Scale input data by variable bounds
        _x = (x - self.l_b) / (self.u_b - self.l_b)

        # Predict raw svm output
        s = self.model.decision_function(_x)
        s_mask = s >= 0.0

        # Determine minimum distances from closest +1 (1) and -1 (0) samples
        self.d_plus = np.min(cdist(x, self.x[self.y == 1]))
        self.d_minus = np.min(cdist(x, self.x[self.y == 0]))

        # Calculate sigmoid function
        fraction = self.d_minus / (self.d_plus + self.tau) - self.d_plus / (self.d_minus + self.tau)
        exponent = self.A * s + self.B * fraction
        exp_mask = exponent >= 0.0
        # prob = 1 / (1 + np.exp(exponent))

        # Invert probability as necessary
        y = np.zeros(len(s))
        y[exp_mask] = np.exp(-exponent[exp_mask]) / (1.0 + np.exp(-exponent[exp_mask]))
        y[~exp_mask] = 1.0 / (1.0 + np.exp(exponent[~exp_mask]))
        # y[s_mask] = 1 - prob[s_mask]
        # y[~s_mask] = prob[~s_mask]
        # y = prob

        return y

    def maximum_likelihood_optimisation(self, deci, label, prior1, prior0):
        # Parameter Setting
        maxiter = 100     # Maximum number of iterations
        minstep = 1e-10   # Minimum step taken in line search
        sigma = 1e-12     # For numerically strict PD of Hessian
        eps = 1e-12        # Stopping criteria

        # Construct Target Support
        hiTarget = (prior1 + 1.0) / (prior1 + 2.0)
        loTarget = 1.0 / (prior0 + 2.0)
        length = prior1 + prior0

        label_mask = label > 0.0
        t = np.zeros(len(label))
        t[label_mask] = hiTarget
        t[~label_mask] = loTarget

        # Initial Point and Initial Function Value
        # A, B = 0.0, -np.log((prior0 + 1.0) / (prior1 + 1.0))
        A, B = -3.0 / min(np.max(deci), -np.min(deci)), -np.log((prior0 + 1.0) / (prior1 + 1.0))
        fApB = A * deci + B
        fapb_mask = fApB >= 0.0
        fval = np.sum(t[fapb_mask] * fApB[fapb_mask] + np.log(1 + np.exp(-fApB[fapb_mask]))) + \
               np.sum((t[~fapb_mask] - 1) * fApB[~fapb_mask] + np.log(1 + np.exp(fApB[~fapb_mask])))

        # Conduct optimisation iterations until convergence or maxiter
        iter = 0
        for _ in range(maxiter):
            # print(f"iter: {iter} A,B: {A, B}, fval:{fval}")

            # Update Gradient and Hessian (use H' = H + sigma I)
            fApB = deci * A + B
            fapb_mask = fApB >= 0.0

            p = np.zeros(length)
            q = np.zeros(length)
            p[fapb_mask] = np.exp(-fApB[fapb_mask]) / (1.0 + np.exp(-fApB[fapb_mask]))
            q[fapb_mask] = 1.0 / (1.0 + np.exp(-fApB[fapb_mask]))
            p[~fapb_mask] = 1.0 / (1.0 + np.exp(fApB[~fapb_mask]))
            q[~fapb_mask] = np.exp(fApB[~fapb_mask]) / (1.0 + np.exp(fApB[~fapb_mask]))

            d1 = t - p
            d2 = p * q
            h11 = sigma + np.sum(deci * deci * d2)
            h22 = sigma + np.sum(d2)
            h21 = np.sum(deci * d2)
            g1 = np.sum(deci * d1)
            g2 = np.sum(d1)

            # Stopping Criteria
            if abs(g1) < eps and abs(g2) < eps:
                break

            # Finding Newton direction: -inv(H') * g
            det = h11 * h22 - h21 * h21
            dA = -(h22 * g1 - h21 * g2) / det
            dB = -(-h21 * g1 + h11 * g2) / det
            gd = g1 * dA + g2 * dB

            # Line Search
            stepsize = 1
            while stepsize >= minstep:
                newA = A + stepsize * dA
                newB = B + stepsize * dB

                # New function value
                fApB = deci * newA + newB
                fapb_mask = fApB >= 0.0
                newf = np.sum(t[fapb_mask] * fApB[fapb_mask] + np.log(1 + np.exp(-fApB[fapb_mask]))) + \
                       np.sum((t[~fapb_mask] - 1) * fApB[~fapb_mask] + np.log(1 + np.exp(fApB[~fapb_mask])))

                # Check sufficient decrease
                if newf < fval + 0.0001 * stepsize * gd:
                    A, B, fval = newA, newB, newf
                    break
                else:
                    stepsize = 0.5 * stepsize

            # Termination condition
            if stepsize < minstep:
                warnings.warn('Line search failed in maximum likelihood optimisation!', UserWarning)
                break

            # Update counter
            iter += 1

        # Assign newly optimised values
        self.A, self.B = A, B

    def hyperopt_tuning_params(self, deci, label, prior1, prior0):
        # Construct Target Support
        hiTarget = (prior1 + 1.0) / (prior1 + 2.0)
        loTarget = 1.0 / (prior0 + 2.0)
        label_mask = label > 0.0
        t = np.zeros(len(label))
        t[label_mask] = hiTarget
        t[~label_mask] = loTarget
        self.deci = deci
        self.label = label
        self.prior1 = prior1
        self.prior0 = prior0
        self.t = t

        # Search Space for A and B
        space = {'A': hp.uniform('A', -10.0, 0.0),
                 'B': hp.uniform('B', -10.0, 0.0)}

        # Hyperopt optimiser
        trials = Trials()
        opt_params = fmin(fn=self._objective_func,
                          space=space,
                          algo=tpe.suggest,
                          max_evals=self.max_evals,
                          trials=trials,
                          verbose=self.verbose)

        self.A = opt_params['A']
        self.B = opt_params['B']
        print('Final A, B:', self.A, self.B)

    def _objective_func(self, space):
        A = space['A']
        B = space['B']
        fApB = A * self.deci + B
        fapb_mask = fApB >= 0.0
        fval = np.sum(self.t[fapb_mask] * fApB[fapb_mask] + np.log(1 + np.exp(-fApB[fapb_mask]))) + \
               np.sum((self.t[~fapb_mask] - 1) * fApB[~fapb_mask] + np.log(1 + np.exp(fApB[~fapb_mask])))

        # print(f"A,B: {A:4.2f}, {B:4.2f}, fval: {fval:.4f}")
        return {'loss': fval, 'status': STATUS_OK}
