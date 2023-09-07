from features.cv_distr import cv_distr

from features.cv_mdl import cv_mdl
from features.dist_corr import dist_corr
from features.f_corr import f_corr
from features.f_decdist import f_decdist
from features.f_skew import f_skew
from cases.MW_setup import MW3
import numpy as np


if __name__ == "__main__":
    # Scalable with dimensionality (give it number of input variables i.e. n_dim = 10)
    setup = MW3()  # use default dimensionality.
    n_points = 5
    n_variables = setup.dim
    n_obj = setup.n_objectives
    n_cons = setup.n_constraints

    # Lower bounds/ upper bounds
    x_lower = setup.lb
    x_upper = setup.ub
    print(x_lower, x_upper)

    # Decision variables
    x = np.random.uniform(x_lower, x_upper, size=(n_points, n_variables))

    # Exact/ approximated pareto front
    pareto_front = setup.f_opt

    # Initialise constraints and objectives.
    obj_array = np.zeros(n_points, n_obj)
    cons_array = np.zeros(n_points, n_cons)

    # evaluate objectives and constraints for each sample.
    for i in range(n_points):
        obj_array[:, i] = setup.obj_func_specific(x[i, :])
        cons_array[:, i] = setup.cons_func_specific(x[i, :])
