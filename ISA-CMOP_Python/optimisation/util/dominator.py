import numpy as np


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


def calculate_domination_matrix(f, cv=None):

    # Number of individuals
    n = f.shape[0]

    # Check if constraint values are passed
    if cv is None:
        cv = [None for idx in range(n)]

    # Output matrix
    m = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            m[i, j] = get_relation(f[i, :], f[j, :], cv[i], cv[j])
            m[j, i] = -m[i, j]

    return m
