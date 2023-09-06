import numpy as np
# import numba


# @numba.njit
def calculate_domination_matrix(f, cv=None, domination_type='pareto'):

    # Obtain domination relation method
    dom_func = select_dom(domination_type)

    # Number of individuals
    n = f.shape[0]

    # Check if constraint values are passed
    if cv is None:
        cv = [None for _ in range(n)]

    # Output matrix
    m = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            m[i, j] = dom_func(f[i, :], f[j, :], cv[i], cv[j])
            m[j, i] = -m[i, j]

    return m


def select_dom(flag):
    """
    Returns pointer to dominance relation method
    """
    # Switch to different domination relations
    if 'pareto' in flag:
        dom_func = get_relation
    elif 'alpha' in flag:
        dom_func = get_alpha_relation
    elif 'c_alpha' in flag:
        dom_func = get_c_alpha_relation
    elif 'nlad' in flag:
        dom_func = get_nlad_relation
    elif 'epsilon' in flag:
        dom_func = get_epsilon_relation
    else:
        dom_func = get_relation
    return dom_func


# @numba.njit
def get_relation(a, b, cv_a=None, cv_b=None):

    # If constraint values are passed, compare them
    if cv_a is not None and cv_b is not None:
        if cv_a < cv_b:
            return 1
        elif cv_b < cv_a:
            return -1

    # If constraint values are equal for both individuals, or are not passed
    val = 0
    # Compare a and b
    for i in range(len(a)):
        if a[i] < b[i]:
            if val == -1:
                # We are now indifferent because elements in both individuals dominate the other
                return 0
            val = 1
        elif b[i] < a[i]:
            if val == 1:
                # We are now indifferent because elements in both individuals dominate the other
                return 0
            val = -1
    return val


# @numba.njit
def get_ref_vec_relation(c_i, c_j, d_i, d_j):
    # If not belonging to the same cluster: non-dominated
    if c_i != c_j:
        return 0
    else:
        # If same cluster and d_i < d_j: i dominates j
        if d_i < d_j:
            return 1
        else:
            # Otherwise if d_i > d_j: j dominates i
            return -1


# @numba.njit
def get_epsilon_relation(a, b, cv_a=None, cv_b=None, epsilon=0.1):
    # If constraint values are passed, compare them
    if cv_a is not None and cv_b is not None:
        if cv_a < cv_b:
            return 1
        elif cv_b < cv_a:
            return -1

    # If constraint values are equal for both individuals, or are not passed
    val = 0
    # Compare a and b
    for i in range(len(a)):
        # Alpha dominance terms

        if (a[i] - epsilon) < b[i]:
            if val == -1:
                # We are now indifferent because elements in both individuals dominate the other
                return 0
            val = 1
        elif b[i] < (a[i] - epsilon):
            if val == 1:
                # We are now indifferent because elements in both individuals dominate the other
                return 0
            val = -1
    return val


# @numba.njit
def get_alpha_relation(a, b, cv_a=None, cv_b=None, alpha=0.1):
    # If constraint values are passed, compare them
    if cv_a is not None and cv_b is not None:
        if cv_a < cv_b:
            return 1
        elif cv_b < cv_a:
            return -1

    # If constraint values are equal for both individuals, or are not passed
    val = 0
    # Compare a and b
    for i in range(len(a)):
        # Alpha dominance terms
        term_a = alpha * np.sum(a[np.arange(len(a)) != i])
        term_b = alpha * np.sum(b[np.arange(len(a)) != i])

        if (a[i] + term_a) < (b[i] + term_b):
            if val == -1:
                # We are now indifferent because elements in both individuals dominate the other
                return 0
            val = 1
        elif (b[i] + term_b) < (a[i] + term_a):
            if val == 1:
                # We are now indifferent because elements in both individuals dominate the other
                return 0
            val = -1
    return val


# @numba.njit
def get_c_alpha_relation(a, b, cv_a=None, cv_b=None, alpha=0.1):
    # If constraint values are passed, compare them
    if cv_a is not None and cv_b is not None:
        if cv_a < cv_b:
            return 1
        elif cv_b < cv_a:
            return -1

    # If constraint values are equal for both individuals, or are not passed
    val = 0
    # Compare a and b
    for i in range(len(a)):
        # CAlpha dominance terms
        a_dash = a[i] + alpha * np.sum(a[np.arange(len(a)) != i])
        b_dash = b[i] + alpha * np.sum(b[np.arange(len(b)) != i])

        a_hat = np.maximum(a_dash, np.sqrt(np.sum(a_dash[np.arange(len(a)) != i] ** 2)))
        b_hat = np.maximum(b_dash, np.sqrt(np.sum(b_dash[np.arange(len(a)) != i] ** 2)))

        if a_hat < b_hat:
            if val == -1:
                # We are now indifferent because elements in both individuals dominate the other
                return 0
            val = 1
        elif b_hat < a_hat:
            if val == 1:
                # We are now indifferent because elements in both individuals dominate the other
                return 0
            val = -1
    return val


# @numba.njit
def get_nlad_relation(a, b, cv_a=None, cv_b=None, alpha=0.1):
    # If constraint values are passed, compare them
    if cv_a is not None and cv_b is not None:
        if cv_a < cv_b:
            return 1
        elif cv_b < cv_a:
            return -1

    # If constraint values are equal for both individuals, or are not passed
    val = 0
    # Compare a and b
    for i in range(len(a)):
        # Alpha dominance terms
        term_a = alpha * a[i] ** 3 + np.sum(a[np.arange(len(a)) != i])
        term_b = alpha * b[i] ** 3 + np.sum(b[np.arange(len(a)) != i])
        if term_a < term_b:
            if val == -1:
                # We are now indifferent because elements in both individuals dominate the other
                return 0
            val = 1
        elif term_b < term_a:
            if val == 1:
                # We are now indifferent because elements in both individuals dominate the other
                return 0
            val = -1
    return val