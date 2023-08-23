import numpy as np


def split_by_feasibility(cons_val, sort_infeasible_by_cv=True):

    # Create mask for feasible solutions (<= zero constraint sum)
    feasible_mask = (cons_val <= 0.0)
    # feasible_mask = (cons_val == 0.0)

    # Filtering population indices by feasible and infeasible
    feasible = np.where(feasible_mask)[0]
    infeasible = np.where(~feasible_mask)[0]

    if sort_infeasible_by_cv:
        infeasible = infeasible[np.argsort(cons_val[infeasible])]

    return feasible, infeasible
