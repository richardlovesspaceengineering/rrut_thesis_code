import copy
import numpy as np


def calculate_finite_difference_grad(problem, x_var, model, eps=1e-4, k=0):
    delta = eps * (problem.x_upper - problem.x_lower)

    cons_grad = np.zeros(problem.n_var)
    for j in range(problem.n_var):
        x_diff = np.zeros(problem.n_var)
        x_diff[j] = delta[j]
        cons_grad[j] = (model.predict(x_var + x_diff) - model.predict(x_var - x_diff)) / (2 * delta[j])

        # # Perturb x
        # x_diff = np.zeros(problem.n_var)
        # x_diff[j] = delta[j]
        # x_e1 = x_var + x_diff
        # x_e2 = x_var - x_diff

        # Maintain within bounds
        # x_e1 = np.maximum(np.minimum(x_e1, problem.x_upper), problem.x_lower)
        # x_e2 = np.maximum(np.minimum(x_e2, problem.x_upper), problem.x_lower)
        # dx = np.linalg.norm(x_e1 - x_e2)

        # Calculate central diference
        # cons_grad[j] = (model.predict(x_e1) - model.predict(x_e2)) / dx
        # cons_grad[j] = (model.predict(x_e1) - model.predict(x_var)) / delta[j]

        # # TODO: Evalute cheaply with real constraint functions
        # val = (problem.cons_func_specific(x_var + x_diff) - problem.cons_func_specific(x_var - x_diff)) / (2 * delta[j])
        # cons_grad[j] = val[k]

    return cons_grad

