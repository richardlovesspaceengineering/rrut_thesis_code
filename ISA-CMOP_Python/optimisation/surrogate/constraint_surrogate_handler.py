import copy
import warnings

import numpy as np
from optimisation.surrogate.classifiers.probabilistic_svm_classifier import PSVMClassification
from optimisation.model.surrogate import Surrogate


class MonoPSVMClassifier(Surrogate):

    def __init__(self, n_dim, l_b, u_b, n_cons, feas_fraction=0.4, svm_C=1.0, **kwargs):

        super().__init__(n_dim=n_dim, l_b=l_b, u_b=u_b, **kwargs)

        # Regularisation parameter C for SVM
        self.svm_C = svm_C

        # Desired Feasibility Fraction for training data
        self.feas_fraction = feas_fraction

        # Base classifier class
        self.model = PSVMClassification(n_dim, l_b, u_b)

        # Mono-surrogate variables
        self.n_cons = n_cons
        self.y = np.zeros((0, self.n_cons))
        self.labels = None
        self.ind_feas_frac = None
        self.eps = None

    def add_points(self, x, y):
        x = np.array(x)
        y = np.array(y)

        self.x = np.vstack((self.x, x))
        self.y = np.vstack((self.y, y))
        self._x = (self.x - self.l_b) / (self.u_b - self.l_b)
        self.n_pts = np.shape(self.y)[0]

        self.updated = False
        self.cv_updated = False

    def _train(self):

        # Aggregate training data to binary feasible-infeasible classes
        self._aggregate_constraints()

        # Train model
        self.model.add_points(self._x, self.labels)
        self.model.train()

    def _predict(self, x):
        x = np.atleast_2d(x)

        # Scale input data by variable bounds
        _x = (x - self.l_b) / (self.u_b - self.l_b)

        # Predict function values & re-scale
        y = self.model.predict_proba(_x)

        return y

    def _predict_proba(self, x):
        raise Exception('Predict method already returns probability for Mono P-SVM classifier surrogate!')

    def _cv_predict(self, model, model_y, x):
        raise Exception('Cross-validation probability prediction not implemented for P-SVM!')

    def _predict_variance(self, x):
        raise Exception('Variance prediction not implemented for P-SVM!')

    def update_cv_models(self):
        pass

    def _aggregate_constraints(self):
        n, n_cons = self.y.shape

        # Determine individual feasibility fractions
        self.ind_feas_frac = np.count_nonzero(self.y <= 0.0, axis=0) / n

        # Sort constraints by decreasing individual feasible fraction
        sorted_mask = np.argsort(-self.ind_feas_frac)
        sorted_frac = self.ind_feas_frac[sorted_mask]
        sorted_g = copy.deepcopy(self.y[:, sorted_mask])
        eps = np.zeros(n_cons)

        # Apply Epsilon-relaxation to individual constraints to obtain the required feasible_fraction
        if np.prod(self.ind_feas_frac) < self.feas_fraction:

            # Loop through sorted constraints to determine required eps value
            for i in range(1, n_cons+1):
                # Combine individual constraints
                new_g = copy.deepcopy(sorted_g[:, :i])
                new_g[new_g <= 0.0] = 0.0
                new_cv = np.sum(new_g, axis=1)
                feas_frac = np.count_nonzero(new_cv <= 0.0) / n

                # If constraint aggregation is feasible above required fraction, continue aggregating
                if feas_frac < self.feas_fraction:
                    # Initialise eps as lowest cv in individual constraint
                    greater_than_zero = sorted_g[:, i-1] > 0.0
                    eps[i-1] = np.min(sorted_g[greater_than_zero, i-1]) / 1.01
                    feas_frac = 0.0
                    while feas_frac < self.feas_fraction:
                        # Increase eps gradually to satisfy feasible fraction
                        eps[i-1] *= 1.01

                        # Combine individual constraints
                        new_g = copy.deepcopy(sorted_g[:, :i]) - eps[:i]
                        new_g[new_g <= 0.0] = 0.0
                        new_cv = np.sum(new_g, axis=1)
                        feas_frac = np.count_nonzero(new_cv <= 0.0) / n

        # Calculate epsilon-modified CV sum and training labels
        g_vals = sorted_g - eps
        g_vals[g_vals <= 0.0] = 0.0
        y_cv = np.sum(g_vals, axis=1)
        pos_mask = y_cv == 0.0
        g_labels = np.zeros(n)
        g_labels[pos_mask] = 1.0
        g_labels[~pos_mask] = 0.0

        # New feasible fraction
        mod_feas_frac = np.count_nonzero(g_labels == 1) / n
        print(f"Mono P-SVM Eps: {eps}, feasible frac: {mod_feas_frac:.2f}")

        # Assign binary class training data internally
        self.labels = g_labels
        self.eps = eps

