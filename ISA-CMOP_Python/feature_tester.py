from features.cv_distr import cv_distr

# from features.cv_mdl import cv_mdl
# from features.dist_corr import dist_corr
# from features.f_corr import f_corr
# from features.f_decdist import f_decdist
# from features.f_skew import f_skew
# from cases.MW_setup import *
# import numpy as np
# import random


# if __name__ == "__main__":
#     # Scalable with dimensionality (give it number of input variables i.e. n_dim = 10)
#     setup = MW3()
#     # use default dimensionality.

#     # Lower bounds/ upper bounds
#     x_lower = setup.lb
#     x_upper = setup.ub
#     print(x_lower, x_upper)

#     # Decision variables
#     x = random.uniform(x_lower, x_upper)

#     # # To evaluate objectives
#     # obj = setup.obj_func_specific(x)
#     # cons = setup.cons_func_specific(x)

#     # # Exact/ approximated pareto front
#     # pareto_front = setup.f_opt

#     # # Eval a bunch of variables
#     # x_var == array ((n_points, n_dim))

#     # obj_arr = np.zeros((n_points, n_obj)
#     # for I in range(n_points)):
#     # obj_arr[I]  = setup.obj_specific_func(x_var[i])
