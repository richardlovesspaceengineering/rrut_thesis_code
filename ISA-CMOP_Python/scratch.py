import numpy as np

# cv_distr scratch working.
A = np.array([[2 + 2j, -2j], [3, 6]])
# print(A.shape)
rmag = np.nonzero(np.sum(np.imag(A) != 0, axis=1))
# print(rmag)
A = np.delete(A, rmag, axis=0)
# print(A)
# print(A.shape)

from scipy.stats import pearsonr

matrix = np.array([[6, -3, -2, 4], [6, 6, 6, 6], [6, 5, 6, 9]])
# result = np.corrcoef(matrix[0, :], matrix[1, :])
# print(matrix)
# print(f_corr(matrix))


from scipy.spatial.distance import cdist

A = np.array([[0.8147, 0.9134], [0.9058, 0.6324], [0.127, 0.0975]])

B = np.array([[0.2785, 0.9659], [0.5469, 0.1576], [0.9575, 0.9706]])

print(A)
print(np.min(A[:, 0], axis=0))

dist_mat = cdist(A, B)
# print(dist_mat)
# print(np.min(dist_mat, axis=0))
# print(np.take_along_axis(dist_mat, np.argmin(dist_mat, axis=0), axis=0))

# ranksort = np.array([[3, 1, 5]])
# n1 = 3
# print(ranksort == n1)
# print(A[ranksort == n1, :])
# # print(A[ranksort == n1, :])

import numpy as np

A = np.array([[0.8147, 0.9134], [0.9058, 0.6324], [0.127, 0.0975]])
ranksort = np.array([3, 1, 5])
n1 = 3

# print(ranksort == n1)
# print(A[ranksort == n1, :])
