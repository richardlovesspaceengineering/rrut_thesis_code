import numpy as np
import scipy.spatial
from scipy.special import erf


def random_permutations(n, x):

    # Compute a list of n randomly permuted sequences
    perms = []
    for i in range(n):
        # Randomly permuting a sequence from 0 to x
        perms.append(np.random.permutation(x))

    # Concatenate list of sequences into one numpy array
    p = np.concatenate(perms)
    return p


def find_duplicates(x, epsilon=1e-16):

    # calculate the distance matrix from each point to another
    dist = scipy.spatial.distance.cdist(x, x)

    # set the diagonal to infinity
    dist[np.triu_indices(len(x))] = np.inf

    # set as duplicate if a point is really close to this one
    is_duplicate = np.any(dist < epsilon, axis=1)

    return is_duplicate


def calc_perpendicular_distance(n, ref_dirs):

    u = np.tile(ref_dirs, (len(n), 1))
    v = np.repeat(n, len(ref_dirs), axis=0)

    norm_u = np.linalg.norm(u, axis=1)

    scalar_proj = np.sum(v * u, axis=1) / norm_u
    proj = scalar_proj[:, None] * u / norm_u[:, None]
    val = np.linalg.norm(proj - v, axis=1)
    matrix = np.reshape(val, (len(n), len(ref_dirs)))

    return matrix


def intersect(a, b):

    h = set()
    for entry in b:
        h.add(entry)

    ret = []
    for entry in a:
        if entry in h:
            ret.append(entry)

    return ret


def at_least_2d_array(x, extend_as='row'):

    if not isinstance(x, np.ndarray):
        x = np.array([x])

    if x.ndim == 1:
        if extend_as == 'row':
            x = x[None, :]
        elif extend_as == 'column':
            x = x[:, None]

    return x


def to_1d_array_if_possible(x):

    if not isinstance(x, np.ndarray):
        x = np.array([x])

    if x.ndim == 2:
        if x.shape[0] == 1 or x.shape[1] == 1:
            x = x.flatten()

    return x


def bilog_transform(obj, beta=1):
    bilog_obj = np.zeros(np.shape(obj))

    for i in range(len(obj)):
        bilog_obj[i] = np.sign(obj[i])*(np.log(beta + np.abs(obj[i])) - np.log(beta))

    return bilog_obj


def reverse_bilog_transform(bilog_obj, beta=1):
    obj = np.zeros(np.shape(bilog_obj))

    for i in range(len(bilog_obj)):
        obj[i] = np.sign(bilog_obj[i])*(np.exp(bilog_obj[i])**np.sign(bilog_obj[i]) - beta)

    return obj



def sp2log_transform(obj_arr, k1=0.0, k2=1.0):
    plog_obj = np.sign(obj_arr + k1) * np.log(1.0 + k2 * np.abs(obj_arr))

    return plog_obj


def vectorized_cdist(A, B, fill_diag_with_inf=False):
    assert A.ndim <= 2 and B.ndim <= 2

    A = at_least_2d_array(A, extend_as="row")
    B = at_least_2d_array(B, extend_as="row")

    u = np.repeat(A, B.shape[0], axis=0)
    v = np.tile(B, (A.shape[0], 1))

    D = np.sqrt(((u - v) ** 2).sum(axis=1))
    M = np.reshape(D, (A.shape[0], B.shape[0]))

    if fill_diag_with_inf:
        np.fill_diagonal(M, np.inf)

    return M


def calc_gamma(V):
    gamma = np.arccos((- np.sort(-1 * V @ V.T))[:, 1])
    gamma = np.maximum(gamma, 1e-64)
    return gamma


def calc_V(ref_dirs):
    return ref_dirs / np.linalg.norm(ref_dirs, axis=1)[:, None]


def min_prob_of_improvement(population, representatives):
    pop_obj = np.atleast_2d(population.extract_obj())
    reps_obj = np.atleast_2d(representatives.extract_obj())
    n = len(population)
    m = len(representatives)

    mpoi_arr = np.zeros(n)
    for i in range(n):
        prob = np.zeros(m)
        for j in range(m):
            prob[j] = predict_mpoi(f1=pop_obj[i], f2=reps_obj[j])
        mpoi_arr[i] = np.min(prob)

    return mpoi_arr


def predict_mpoi(f1, f2, s1=0.1, s2=0.0):
    term = (f1 - f2) / 0.1  # TODO: np.sqrt(s1 - s2)

    return 1.0 - 0.5 * np.prod(1.0 + erf(term / np.sqrt(2)))
