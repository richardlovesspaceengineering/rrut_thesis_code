import numpy as np
import random

from optimisation.model.surrogate import Surrogate
from optimisation.surrogate.models.rbf import RBF
from optimisation.surrogate.models.mars import MARSRegression
from optimisation.surrogate.models.mars import MARS
from optimisation.surrogate.models.svr import SupportVectorRegression


class RBFKernelEnsembleSurrogate(Surrogate):
    def __init__(self, n_dim, l_b, u_b, c=0.5, p_type='linear', kernel_types=None, n_kernels=3, kernel_update_iter=5,
                 **kwargs):

        super().__init__(n_dim=n_dim, l_b=l_b, u_b=u_b, **kwargs)

        if kernel_types is None:
            self.kernel_types = ["gaussian", "cubic", "multiquadratic", "inverse_multiquadratic",
                                 "matern_32", "matern_52", "matern_12", "thin_plate_spline",
                                 "cauchy", "logistic", "hyperbolic_tangent_sigmoid"]
        else:
            self.kernel_types = kernel_types
        self.c = c
        self.p_type = p_type

        # Create Ensemble of RBF kernels
        k = len(self.kernel_types)
        ensemble = [RBF(n_dim=n_dim, c=self.c, p_type=self.p_type, kernel_type=self.kernel_types[j]) for j in range(k)]

        # Add MARS surrogate too
        extra_model = MARS(n_dim=n_dim, max_terms=5*n_dim, max_degree=3)
        ensemble.append(extra_model)

        self.model = ensemble
        self.n_kernels = n_kernels

        # Add on for MARS
        self.kernel_types.append("mars")
        self.kernel_types = np.array(self.kernel_types)

        # Matrix to store ranks
        self.rank_matrix = None
        self.kernel_update_iter = kernel_update_iter
        self.kernel_idx_to_keep = random.sample(range(len(self.kernel_types)), self.n_kernels)
        print('Using kernels: ', self.kernel_types[self.kernel_idx_to_keep])

    def _train(self):

        # Compute mean and std of training function values
        # self._mu = np.mean(self.y)
        self._mu = np.median(self.y)
        self._sigma = max([np.std(self.y), 1e-6])

        # Scale training data by variable bounds
        self._x = (self.x - self.l_b) / (self.u_b - self.l_b)

        for i, model in enumerate(self.model):
            if isinstance(model, RBF):
                if model.p_type is None:
                    # Normalise function values
                    _y = self.y - self._mu

                    # Train model
                    model.fit(self._x, _y)
                else:
                    # Train model
                    model.fit(self._x, self.y)
            # elif isinstance(model, MARSRegression):
            #     model.model.fit(self._x, self.y)
            #     model.train()
            else:
                # Train model
                model.fit(self._x, self.y)

    def _predict(self, x):
        y = np.zeros(len(self.model))
        _x = (x - self.l_b) / (self.u_b - self.l_b)

        for i, model in enumerate(self.model):
            if isinstance(model, RBF):
                if model.p_type is None:
                    # Train model
                    y[i] = model.predict(_x) + self._mu
                else:
                    # Train model
                    y[i] = model.predict(_x)
            else:
                # Train model
                y[i] = model.predict(_x)

        # y_mean = np.mean(y)
        y_median = np.median(y)
        return y_median

    def _predict_model(self, x, model_idx):
        _x = (x - self.l_b) / (self.u_b - self.l_b)

        # Select model from index of selected kernels
        model = self.model[self.kernel_idx_to_keep[model_idx]]
        if isinstance(model, RBF):
            if model.p_type is None:
                # Train model
                y = model.predict(_x) + self._mu
            else:
                # Train model
                y = model.predict(_x)
        else:
            # Train model
            y = model.predict(_x)

        return y

    def _cv_predict(self, model, model_y, x):

        # Scale input data by variable bounds
        _x = (x - self.l_b) / (self.u_b - self.l_b)

        # Predict function values
        if isinstance(model, RBF):
            if model.p_type is None:
                # _mu = np.mean(model_y)
                _mu = np.median(model_y)
                y = model.predict(_x) + _mu
            else:
                y = model.predict(_x)
        else:
            y = model._predict(_x)

        return y

    def _cv_predict_ensemble(self, x):
        y = np.zeros(len(self.model))
        for i, model in enumerate(self.model):
            y[i] = self._predict_model(x, model)

        # y_mean = np.mean(y)
        y_mean = np.median(y)
        y_std = np.std(y)
        return y_mean, y_std, y

    def _predict_variance(self, x):
        y = np.zeros(len(self.model))
        for i, model in enumerate(self.model):
            y[i] = self._predict_model(x, model)

        y_std = np.std(y)
        if model.p_type is None:
            y_std *= self._sigma
        return y_std ** 2

    def _predict_median(self, x):
        y = np.zeros(len(self.model))
        for i, model in enumerate(self.model):
            y[i] = self._predict_model(x, model)

        return y

    def update_cv_models(self):

        self.cv_models = self.model

        # Training each of the cross-validation models
        for i, model in enumerate(self.cv_models):
            model.fit(self._x, self.y)

    def predict_mse_accuracy(self, infill_population):
        """
        Prediction accuracy of kernels based off MSE (Do not call after surrogates are trained on infill_population)
        """
        x_var = infill_population.extract_var()
        obj_values = infill_population.extract_obj()
        prediction_accuracy = np.zeros(len(self.model))

        # Calculate prediction of models with given infill points
        for cntr, model in enumerate(self.model):
            accuracy = 0
            for idx in range(len(infill_population)):
                opt_pred = model.predict(x_var[idx])
                accuracy += (obj_values[idx] - opt_pred)**2                      # MSE Calculation
            prediction_accuracy[cntr] = (1 / len(infill_population)) * accuracy  # MSE Calculation

        # Determine order of prediction accuracy
        prediction_indices = np.argsort(prediction_accuracy)

        # Update ranks and change kernel selections if number of update iterations reached
        kernel_ranks = prediction_indices.argsort()
        if self.rank_matrix is None:
            self.rank_matrix = np.atleast_2d(kernel_ranks)
        else:
            self.rank_matrix = np.vstack((self.rank_matrix, kernel_ranks))
        if len(self.rank_matrix[:, 0]) >= self.kernel_update_iter:
            self.update_kernel_selection()

        return self.kernel_types[self.kernel_idx_to_keep]

    def predict_rank_accuracy(self, infill_population):
        """
        Surrogate accuracy based off rank proximity to the real objective function given infill_population
        (The larger the infill_population the better)
        """

        # Extract infill population
        x_var = infill_population.extract_var()
        obj_values = infill_population.extract_obj().flatten()

        # Rank objective values of real function
        real_ranks = np.argsort(obj_values).argsort()

        # Rank objective of surrogate predictions
        predicted_ranks = np.zeros((len(self.model), len(infill_population)))
        for cntr, model in enumerate(self.model):
            surrogate_obj = np.zeros(len(infill_population))
            for idx in range(len(infill_population)):
                surrogate_obj[idx] = model.predict(x_var[idx])

            # Rank by surrogate objective values
            predicted_ranks[cntr, :] = np.argsort(surrogate_obj).argsort()

        # Calculate absolute error of real and predicted ranks
        delta_ranks = np.sum(np.abs(predicted_ranks - real_ranks), axis=1)
        print('delta ranks: ', delta_ranks)

        # If all kernels rank equally, call MSE ranking and skip current ranking
        if len(np.unique(delta_ranks)) == 1:
            kernels_to_keep = self.predict_mse_accuracy(infill_population)
            print('Used MSE ranking instead')
            return kernels_to_keep

        # Perform ranking on the changes in ranking
        kernel_ranks = np.argsort(delta_ranks).argsort()
        if self.rank_matrix is None:
            self.rank_matrix = np.atleast_2d(kernel_ranks)
        else:
            self.rank_matrix = np.vstack((self.rank_matrix, kernel_ranks))
        if len(self.rank_matrix[:, 0]) >= self.kernel_update_iter:
            self.update_kernel_selection()

        return self.kernel_types[self.kernel_idx_to_keep]

    def predict_sep_accuracy(self, infill_population):
        """
        Ranks kernels according to SEP (Do not call after surrogates are trained on infill_population)
        """
        # Extract infill population
        x_var = infill_population.extract_var()
        obj_values = infill_population.extract_obj().flatten()

        # Loop through all the surrogate models
        N = len(infill_population)
        E_sep = np.zeros(len(self.model))
        for cntr, model in enumerate(self.model):

            # Predict surrogate function values
            pred_values = np.zeros(N)
            for idx in range(N):
                pred_values[idx] = model.predict(x_var[idx])

            # Evaluate the SEP
            for i in range(N):
                for j in range(i+1, N):
                    E_sep[cntr] += self._pairwise_comp(obj_values[i], obj_values[j], pred_values[i], pred_values[j])

            # Normalise SEP
            E_sep[cntr] *= 1 / (0.5*N*(N-1))

        # If all kernels rank equally, call MSE ranking and skip current ranking
        if len(np.unique(E_sep)) == 1:
            kernels_to_keep = self.predict_mse_accuracy(infill_population)
            print('Used MSE ranking instead')
            return kernels_to_keep

        # Rank kernels according to SEP
        kernel_ranks = np.argsort(E_sep).argsort()
        if self.rank_matrix is None:
            self.rank_matrix = np.atleast_2d(kernel_ranks)
        else:
            self.rank_matrix = np.vstack((self.rank_matrix, kernel_ranks))
        if len(self.rank_matrix[:, 0]) >= self.kernel_update_iter:
            self.update_kernel_selection()

        return self.kernel_types[self.kernel_idx_to_keep]

    def _pairwise_comp(self, f_real_i, f_real_j, f_pred_i, f_pred_j):
        """
        Calculates the pairwise Selection Error Probability (SEP) metric (Ahrari2019)
        :param f_real_i: f(x_i)   : Real Function
        :param f_real_j: f(x_j)   : Real Function
        :param f_pred_i: f(x_i)   : Surrogate Function
        :param f_pred_j: f(x_j)   : Surrogate Function
        :return: q(x_i, x_j)      : pairwise match=0, dismatch=1
        """
        if (f_real_i - f_real_j)*(f_pred_i - f_pred_j) < 0:
            return 1
        else:
            return 0

    def predict_random_accuracy(self):
        """
        Ranks kernels randomly (Do not call after surrogates are trained on infill_population)
        """

        # Perform random ranking
        kernel_ranks = np.random.randint(0, len(self.kernel_types), len(self.kernel_types))
        if self.rank_matrix is None:
            self.rank_matrix = np.atleast_2d(kernel_ranks)
        else:
            self.rank_matrix = np.vstack((self.rank_matrix, kernel_ranks))
        if len(self.rank_matrix[:, 0]) >= self.kernel_update_iter:
            # self.kernel_idx_to_keep = random.sample(range(len(self.kernel_types)), self.n_kernels)
            self.update_kernel_selection()

        return self.kernel_types[self.kernel_idx_to_keep]

    def update_kernel_selection(self):

        # Find combined ranks after n iterations
        combined_ranks = np.sum(self.rank_matrix, axis=0)
        ranked_indices = np.argsort(combined_ranks).argsort()

        # Select k most accurate kernels
        self.kernel_idx_to_keep = np.argpartition(ranked_indices, self.n_kernels)[:self.n_kernels]

        # Sort indices from best to worst kernel
        ordered_indices = np.argsort(ranked_indices[self.kernel_idx_to_keep])
        self.kernel_idx_to_keep = self.kernel_idx_to_keep[ordered_indices]

        # Clear rank matrix
        self.rank_matrix = None

    def clear_ranks(self):

        # Clear rank matrix
        self.rank_matrix = None


