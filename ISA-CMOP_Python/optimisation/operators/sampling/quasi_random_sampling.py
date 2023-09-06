import numpy as np
import sobol
from optimisation_framework.optimisation.operators.sampling.latin_hypercube_sampling import LatinHypercubeSampling
# from optimisation_framework.optimisation.model.sampling import Sampling
from optimisation_framework.cvt_sampling import SamplingMethods

class QuasiRandomSampling(object):
    def __init__(self, dim_sample, points_sample):
        self.dim = dim_sample
        self.n_sample = points_sample

    def HaltonSequence(self):
        big_number = 10
        while 'Not enought primes':
            base = self.primes_from_2_to(big_number)[:self.dim]
            if len(base) == self.dim:
                break
            big_number += 1000

        # Generate a sample using a Van der Corput sequence per dimension.
        sample = [self.van_der_corput(self.n_sample + 1, self.dim) for self.dim in base]
        sample = np.stack(sample, axis=-1)[1:]

        return sample

    def HammersleySequence(self):
        """Yields n Hammersley points on the unit square in the xy plane.
        This function uses a base of 2.
        """
        vec = np.ndarray((self.n_sample, 2))
        for k in range(self.n_sample):
            u = 0
            p = 0.5
            kk = k
            while kk > 0:
                if kk & 1:
                    u += p
                p *= 0.5
                kk >>= 1
            v = (k + 0.5) / self.n_sample
            vec[k, :] = (u, v)
        return vec

    def SobolSequence(self, skip=0):
        sobol = self.generator()
        for i in range(skip):
            next(sobol)
        points = np.empty((self.n_sample, self.dim))
        for i in range(self.n_sample):
            points[i] = next(sobol)
        return points

    def rightmost_zero(self, n):
        """Position of the lowest 0-bit in the binary representation of integer `n`."""
        s = np.binary_repr(n)
        i = s[::-1].find("0")
        if i == -1:
            i = len(s)
        return i

    def generator(self):
        """Generator for the Sobol sequence"""
        DIMS = 1111  # maximum number of dimensions
        BITS = 30  # maximum number of bits

        if not (1 <= self.dim <= DIMS):
            raise ValueError("Sobol: self.dimension must be between 1 and %i." % DIMS)

        # initialize direction numbers
        V = np.zeros((DIMS, BITS), dtype=int)
        data = np.genfromtxt("sobol1111.tsv", dtype=int)
        poly = data[:, 0]
        V[:, :13] = data[:, 1:14]
        V[0, :] = 1
        for i in range(1, self.dim):
            m = len(np.binary_repr(poly[i])) - 1
            include = np.array([int(b) for b in np.binary_repr(poly[i])[1:]])
            for j in range(m, BITS):
                V[i, j] = V[i, j - m]
                for k in range(m):
                    if include[k]:
                        V[i, j] = np.bitwise_xor(V[i, j], 2 ** (k + 1) * V[i, j - k - 1])
        V = V[:self.dim] * 2 ** np.arange(BITS)[::-1]

        point = np.zeros(self.dim, dtype=int)
        for i in range(2 ** BITS):
            point = np.bitwise_xor(point, V[:, self.rightmost_zero(i)])
            yield point / 2 ** BITS

    def primes_from_2_to(self, n):
        """Prime number from 2 to n.

        From `StackOverflow <https://stackoverflow.com/questions/2068372>`_.

        :param int n: sup bound with ``n >= 6``.
        :return: primes in 2 <= p < n.
        :rtype: list
        """
        sieve = np.ones(n // 3 + (n % 6 == 2), dtype=np.bool_)
        for i in range(1, int(n ** 0.5) // 3 + 1):
            if sieve[i]:
                k = 3 * i + 1 | 1
                sieve[k * k // 3::2 * k] = False
                sieve[k * (k - 2 * (i & 1) + 4) // 3::2 * k] = False
        return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]

    def van_der_corput(self, n_sample, base=2):
        """Van der Corput sequence.

        :param int n_sample: number of element of the sequence.
        :param int base: base of the sequence.
        :return: sequence of Van der Corput.
        :rtype: list (n_samples,)
        """
        n_sample, base = int(n_sample), int(base)
        sequence = []
        for i in range(n_sample):
            n_th_number, denom = 0., 1.
            while i > 0:
                i, remainder = divmod(i, base)
                denom *= base
                n_th_number += remainder / denom
            sequence.append(n_th_number)
        return sequence


class HammersleySampling(SamplingMethods):
    """
    A class that performs Hammersley Sampling.
    Hammersley samples are generated in a similar way to Halton samples - based on the reversing/flipping the base conversion of numbers using primes.
    To generate :math:`n` samples in a :math:`p`-dimensional space, the first :math:`\\left(p-1\\right)` prime numbers are used to generate the samples. The first dimension is obtained by uniformly dividing the region into **no_samples points**.
    Note:
        Use of this method is limited to use in low-dimensionality problems (less than 10 variables). At higher dimensionalities, the performance of the sampling method has been shown to degrade.
    To use: call class with inputs, and then ``sample_points`` function.
    **Example:**

    .. code-block:: python

        # For the first 10 Hammersley samples in a 2-D space:
        >>> b = rbf.HammersleySampling(data, 10, sampling_type="selection")
        >>> samples = b.sample_points()
    """

    def __init__(self, data_input, number_of_samples=None, sampling_type=None):
        """
        Initialization of **HammersleySampling** class. Two inputs are required.
        Args:
            data_input (NumPy Array): The input data set or range to be sampled.
                - When the aim is to select a set of samples from an existing dataset, the dataset must be a NumPy Array or a Pandas Dataframe and **sampling_type** option must be set to "selection". The output variable (Y) is assumed to be supplied in the last column.
                - When the aim is to generate a set of samples from a data range, the dataset must be a list containing two lists of equal lengths which contain the variable bounds and **sampling_type** option must be set to "creation". It is assumed that no range contains no output variable information  in this case.
            number_of_samples(int): The number of samples to be generated. Should be a positive integer less than or equal to the number of entries (rows) in **data_input**.
            sampling_type(str) : Option to generate sample from a supplied range ("creation"). Default is "creation".
            Returns:
                **self** function containing the input information.
            Raises:
                Exception: When the **number_of_samples** is invalid (not an integer, too large, zero, negative)
        """
        self.sampling_type = sampling_type

        if self.sampling_type == 'creation':
            data_headers = []
            self.data = data_input
            self.data_headers = data_headers

            # Catch potential errors in number_of_samples
            if number_of_samples is None:
                print("\nNo entry for number of samples to be generated. The default value of 5 will be used.")
                number_of_samples = 5
            elif not isinstance(number_of_samples, int):
                raise Exception('number_of_samples must be an integer.')
            elif number_of_samples <= 0:
                raise Exception('number_of_samples must a positive, non-zero integer.')
            self.number_of_samples = number_of_samples
            self.x_data = self.data  # Only x data will be present in this case

        if self.x_data.shape[1] > 10:
            raise Exception(
                'Dimensionality problem: This method is not available for problems with dimensionality > 10: the performance of the method degrades substantially at higher dimensions')

    def sample_points(self):
        """
        The **sampling_type** method generates the Hammersley sample points. The steps followed here are:
            1. Determine the number of features :math:`n_{f}` in the input data.
            2. Generate the list of :math:`\\left(n_{f}-1\\right)` primes to be considered by calling prime_number_generator.
            3. Divide the space [0,**number_of_samples**-1] into **number_of_samples** places to obtain the first dimension for the Hammersley sequence.
            4. For the other :math:`\\left(n_{f}-1\\right)` dimensions, create first **number_of_samples** elements of the Hammersley sequence for each of the :math:`\\left(n_{f}-1\\right)` primes.
            5. Create the Hammersley samples by combining the corresponding elements of the Hammersley sequences created in steps 3 and 4
            6. When in "selection" mode, determine the closest corresponding point in the input dataset using Euclidean distance minimization. This is done by calling the ``nearest_neighbours`` method in the sampling superclass.
        Returns:
            NumPy Array or Pandas Dataframe:     A numpy array or Pandas dataframe containing **number_of_samples** Hammersley sample points.
        """
        no_features = self.x_data.shape[1]
        if no_features == 1:
            prime_list = []
        else:
            prime_list = self.prime_number_generator(no_features - 1)
        sample_points = np.zeros((self.number_of_samples, no_features))
        sample_points[:, 0] = (np.arange(0, self.number_of_samples)) / self.number_of_samples
        for i in range(0, len(prime_list)):
            sample_points[:, i + 1] = self.data_sequencing(self.number_of_samples, prime_list[i])

        unique_sample_points = self.sample_point_selection(self.data, sample_points, self.sampling_type)
        if len(self.data_headers) > 0:
            unique_sample_points = pd.DataFrame(unique_sample_points, columns=self.data_headers)
        return unique_sample_points



