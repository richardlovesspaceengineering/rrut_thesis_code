import numpy as np
from features.f_corr import f_corr

# cv_distr scratch working.
A = np.array([[2 + 2j, -2j], [3, 6]])
print(A.shape)
rmag = np.nonzero(np.sum(np.imag(A) != 0, axis=1))
print(rmag)
A = np.delete(A, rmag, axis=0)
print(A)
print(A.shape)

from scipy.stats import pearsonr

matrix = np.array([[6, -3, -2, 4], [6, 6, 6, 6], [6, 5, 6, 9]])
# result = np.corrcoef(matrix[0, :], matrix[1, :])
print(matrix)
print(f_corr(matrix))
