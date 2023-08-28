import numpy as np

# cv_distr scratch working.
A = np.array([[2 + 2j, -2j], [3, 6]])
rmag = np.nonzero(np.sum(np.imag(A) != 0, axis=1))
print(rmag)
A = np.delete(A, rmag, axis=0)
print(A)
