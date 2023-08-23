import numpy as np
import scipy.spatial


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

def calc_gamma(V):
    gamma = np.arccos((- np.sort(-1 * V @ V.T))[:, 1])
    gamma = np.maximum(gamma, 1e-64)
    return gamma


def calc_V(ref_dirs):
    return ref_dirs / np.linalg.norm(ref_dirs, axis=1)[:, None]
