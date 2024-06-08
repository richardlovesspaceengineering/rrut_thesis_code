from __future__ import division, print_function
import numpy as np
import pandas as pd
import warnings


from optimisation.model.sampling import Sampling


class CentroidalVoronoiTessellationSampling(Sampling):

    def __init__(self, tolerance=1e-7):
        super().__init__()

        self.tolerance = tolerance
        self.sampling_type = 'creation'
        self.cvt = None

    def _do(self, dim, n_samples, seed=None):
        bounds = np.vstack((np.zeros(dim), np.ones(dim)))
        self.cvt = CVTSampling(data_input=bounds, number_of_samples=n_samples,
                               sampling_type=self.sampling_type, tolerance=self.tolerance)
        self.x = self.cvt.sample_points()


class FeatureScaling:
    """
    A class for scaling and unscaling input and output data. The class contains three main functions
    """

    def __init__(self):
        pass

    @staticmethod
    def data_scaling_minmax(data):
        """
        This function performs column-wise minimax scaling on the input dataset.
            Args:
                data (NumPy Array or Pandas Dataframe): The input data set to be scaled. Must be a numpy array or dataframe.
            Returns:
                scaled_data(NumPy Array): A 2-D numpy array containing the scaled data. All array values will be between [0, 1].
                data_minimum(NumPy Array): A 2-D row vector containing the column-wise minimums of the input data
                data_maximum(NumPy Array): A 2-D row vector containing the column-wise maximums of the input data
            Raises:
                TypeError: Raised when the input data is not a numpy array or dataframe
        """
        # Confirm that data type is an array or DataFrame
        if isinstance(data, np.ndarray):
            input_data = data
        elif isinstance(data, pd.DataFrame):
            input_data = data.values
        else:
            raise TypeError('original_data_input: Pandas dataframe or numpy array required.')

        if input_data.ndim == 1:
            input_data = input_data.reshape(len(input_data), 1)
        data_minimum = np.min(input_data, axis=0)
        data_maximum = np.max(input_data, axis=0)
        scale = data_maximum - data_minimum
        scale[scale == 0.0] = 1.0
        scaled_data = (input_data - data_minimum)/scale
        # scaled_data = (input_data - data_minimum) / (data_maximum - data_minimum)
        data_minimum = data_minimum.reshape(1, data_minimum.shape[0])
        data_maximum = data_maximum.reshape(1, data_maximum.shape[0])
        return scaled_data, data_minimum, data_maximum

    @staticmethod
    def data_unscaling_minmax(x_scaled, x_min, x_max):
        """
        This function performs column-wise un-scaling on the a minmax-scaled input dataset.
            Args:
                x_scaled(NumPy Array): The input data set to be un-scaled. Data values should be between 0 and 1.
                x_min(NumPy Array): 1-D or 2-D (n-by-1) vector containing the actual minimum value for each column. Must contain same number of elements as the number of columns in x_scaled.
                x_max(NumPy Array): 1-D or 2-D (n-by-1) vector containing the actual maximum value for each column. Must contain same number of elements as the number of columns in x_scaled.
            Returns:
                unscaled_data(NumPy Array): A 2-D numpy array containing the scaled data, unscaled_data = x_min + x_scaled * (x_max - x_min)
            Raises:
                IndexError: Function raises index error when the dimensions of the arrays are inconsistent.
        """
        # Check if it can be evaluated. Will return index error if dimensions are wrong
        if x_scaled.ndim == 1:  # Check if 1D, and convert to 2D if required.
            x_scaled = x_scaled.reshape(len(x_scaled), 1)
        if (x_scaled.shape[1] != x_min.size) or (x_scaled.shape[1] != x_max.size):
            raise IndexError('Dimensionality problems with data for un-scaling.')
        unscaled_data = x_min + x_scaled * (x_max - x_min)
        return unscaled_data


class SamplingMethods:

    def nearest_neighbour(self, full_data, a):
        """
        Function determines the closest point to a in data_input (user provided data).
        This is done by determining the input data with the smallest L2 distance from a.
        The function:
        1. Calculates the L2 distance between all the input data points and a,
        2. Sorts the input data based on the calculated L2-distances, and
        3. Selects the sample point in the first row (after sorting) as the closest sample point.
        Args:
            self: contains, among other things, the input data.
            full_data: refers to the input dataset supplied by the user.
            a: a single row vector containing the sample point we want to find the closest sample to.
        Returns:
            closest_point: a row vector containing the closest point to a in self.x_data
        """

        dist = full_data[:, :-1] - a
        l2_norm = np.sqrt(np.sum((dist ** 2), axis=1))
        l2_norm = l2_norm.reshape(l2_norm.shape[0], 1)
        distances = np.append(full_data, l2_norm, 1)
        sorted_distances = distances[distances[:, -1].argsort()]
        closest_point = sorted_distances[0, :-1]
        return closest_point

    def points_selection(self, full_data, generated_sample_points):
        """
        Uses L2-distance evaluation (implemented in nearest_neighbour) to find closest available points in original data to those generated by the sampling technique.
        Calls the nearest_neighbour function for each row in the input data.
        Args:
            full_data: refers to the input dataset supplied by the user.
            generated_sample_points(NumPy Array): The vector of points (number_of_sample rows) for which the closest points in the original data are to be found. Each row represents a sample point.
        Returns:
            equivalent_points: Array containing the points (in rows) most similar to those in generated_sample_points
        """

        equivalent_points = np.zeros((generated_sample_points.shape[0], generated_sample_points.shape[1] + 1))
        for i in range(0, generated_sample_points.shape[0]):
            closest_point = self.nearest_neighbour(full_data, generated_sample_points[i, :])
            equivalent_points[i, :] = closest_point
        return equivalent_points

    def sample_point_selection(self, full_data, sample_points, sampling_type):
        if sampling_type == 'selection':
            sd = FeatureScaling()
            scaled_data, data_min, data_max = sd.data_scaling_minmax(full_data)
            points_closest_scaled = self.points_selection(scaled_data, sample_points)
            points_closest_unscaled = sd.data_unscaling_minmax(points_closest_scaled, data_min, data_max)

            unique_sample_points = np.unique(points_closest_unscaled, axis=0)
            if unique_sample_points.shape[0] < points_closest_unscaled.shape[0]:
                warnings.warn(
                    'The returned number of samples is less than the requested number due to repetitions during nearest neighbour selection.')
            print('\nNumber of unique samples returned by sampling algorithm:', unique_sample_points.shape[0])

        elif sampling_type == 'creation':
            sd = FeatureScaling()
            unique_sample_points = sd.data_unscaling_minmax(sample_points, full_data[0, :], full_data[1, :])

        return unique_sample_points

    def prime_number_generator(self, n):
        """
        Function generates a list of the first n prime numbers
            Args:
                n(int): Number of prime numbers required
            Returns:
                prime_list(list): A list of the first n prime numbers
        Example: Generate first three prime numbers
            >>  prime_number_generator(3)
            >> [2, 3, 5]
        """
        # Alternative way of generating primes using list generators
        # prime_list = []
        # current_no = 2
        # while len(prime_list) < n:
        #     matching_objs = next((o for o in range(2, current_no) if current_no % o == 0), 0)
        #     if matching_objs==0:
        #         prime_list.append(current_no)
        #     current_no += 1

        prime_list = []
        current_no = 2
        while len(prime_list) < n:
            for i in range(2, current_no):
                if (current_no % i) == 0:
                    break
            else:
                prime_list.append(current_no)
            current_no += 1
        return prime_list

    def base_conversion(self, a, b):
        """
        Function converts integer a from base 10 to base b
            Args:
                a(int): Number to be converted, base 10
                b(int): Base required
            Returns:
                string_representation(list): List containing strings of individual digits of "a" in the new base "b"
        Examples: Convert (i) 5 to base 2 and (ii) 57 to base 47
            >>  base_conversion(5, 2)
            >> ['1', '0', '1']
            >>  base_conversion(57, 47)
            >> ['1', '10']
        """

        string_representation = []
        if a < b:
            string_representation.append(str(a))
        else:
            while a > 0:
                a, c = (a // b, a % b)
                string_representation.append(str(c))
            string_representation = (string_representation[::-1])
        return string_representation

    def prime_base_to_decimal(self, num, base):
        """
        ===============================================================================================================
        Function converts a fractional number "num" in base "base" to base 10. Reverses the process in base_conversion
        Note: The first string element is ignored, since this would be zero for a fractional number.
            Args:
                num(list): Number in base b to be converted. The number must be represented as a list containing individual digits of the base, with the first entry as zero.
                b(int): Original base
            Returns:
                decimal_equivalent(float): Fractional number in base 10
        Examples:
        Convert 0.01 (base 2) to base 10
            >>  prime_base_to_decimal(['0', '0', '1'], 2)  # Represents 0.01 in base 2
            >> 0.25
        Convert 0.01 (base 20) to base 10
            >>  prime_base_to_decimal(['0', '0', '1'], 20)  # Represents 0.01 in base 20
            >> 0.0025
        ================================================================================================================
        """
        binary = num
        decimal_equivalent = 0
        # Convert fractional part decimal equivalent
        for i in range(1, len(binary)):
            decimal_equivalent += int(binary[i]) / (base ** i)
        return decimal_equivalent

    def data_sequencing(self, no_samples, prime_base):
        """
        ===============================================================================================================
        Function which generates the first no_samples elements of the Halton or Hammersley sequence based on the prime number prime_base
        The steps for generating the first no_samples of the sequence are as follows:
        1. Create a list of numbers between 0 and no_samples --- nums = [0, 1, 2, ..., no_samples]
        2. Convert each element in nums into its base form based on the prime number prime_base, reverse the base digits of each number in num
        3. Add a decimal point in front of the reversed number
        4. Convert the reversed numbers back to base 10
            Args:
                no_samples(int): Number of Halton/Hammersley sequence elements required
                prime_base(int): Current prime number to be used as base
            Returns:
                sequence_decimal(NumPy Array): 1-D array containing the first no_samples elements of the sequence based on prime_base
        Examples:
        First three elements of the Halton sequence based on base 2
            >>  data_sequencing(self, 3, 2)
            >> [0, 0.5, 0.75]
        ================================================================================================================
        """
        pure_numbers = np.arange(0, no_samples)
        bitwise_rep = []
        reversed_bitwise_rep = []
        sequence_bitwise = []
        sequence_decimal = np.zeros((no_samples, 1))
        for i in range(0, no_samples):
            base_rep = self.base_conversion(pure_numbers[i], prime_base)
            bitwise_rep.append(base_rep)
            reversed_bitwise_rep.append(base_rep[::-1])
            sequence_bitwise.append(['0.'] + reversed_bitwise_rep[i])
            sequence_decimal[i, 0] = self.prime_base_to_decimal(sequence_bitwise[i], prime_base)
        sequence_decimal = sequence_decimal.reshape(sequence_decimal.shape[0], )
        return sequence_decimal


class CVTSampling(SamplingMethods):
    """
    A class that constructs Centroidal Voronoi Tessellation (CVT) samples.
    CVT sampling is based on the generation of samples in which the generators of the Voronoi tessellations and the mass centroids coincide.
    To use: call class with inputs, and then ``sample_points`` function.
    **Example:**

    .. code-block:: python

        # For the first 10 CVT samples in a 2-D space:
        # >>> b = rbf.CVTSampling(data_bounds, 10, tolerance = 1e-5, sampling_type="creation")
        # >>> samples = b.sample_points()
    """

    def __init__(self, data_input, number_of_samples=None, tolerance=None, sampling_type=None):
        """
        Initialization of CVTSampling class. Two inputs are required, while an optional option to control the solution accuracy may be specified.
        Args:
            data_input (NumPy Array, Pandas Dataframe or list): The input data set or range to be sampled.
                - When the aim is to select a set of samples from an existing dataset, the dataset must be a NumPy Array or a Pandas Dataframe and **sampling_type** option must be set to "selection". The output variable (Y) is assumed to be supplied in the last column.
                - When the aim is to generate a set of samples from a data range, the dataset must be a list containing two lists of equal lengths which contain the variable bounds and **sampling_type** option must be set to "creation". It is assumed that no range contains no output variable information  in this case.
            number_of_samples(int): The number of samples to be generated. Should be a positive integer less than or equal to the number of entries (rows) in **data_input**.
            sampling_type(str) : Option which determines whether the algorithm selects samples from an existing dataset ("selection") or attempts to generate sample from a supplied range ("creation"). Default is "creation".
        Keyword Args:
            tolerance(float): Maximum allowable Euclidean distance between centres from consectutive iterations of the algorithm. Termination condition for algorithm.
                - The smaller the value of tolerance, the better the solution but the longer the algorithm requires to converge. Default value is :math:`10^{-7}`.
        Returns:
                **self** function containing the input information.
        Raises:
                ValueError: When **data_input** is the wrong type.
                Exception: When the **number_of_samples** is invalid (not an integer, too large, zero, negative)
                Exception: When the tolerance specified is too loose (tolerance > 0.1) or invalid
                warnings.warn: when the tolerance specified by the user is too tight (tolerance < :math:`10^{-9}`)
        """
        if sampling_type is None:
            sampling_type = 'creation'
            self.sampling_type = sampling_type
        else:
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
            self.number_of_centres = number_of_samples

            x_data = data_input  # Only x data will be present in this case
            if x_data.ndim == 1:
                x_data = x_data.reshape(len(x_data), 1)
            self.x_data = x_data
            self.y_data = []

        if tolerance is None:
            tolerance = 1e-7
        elif tolerance > 0.1:
            raise Exception('Tolerance must be less than 0.1 to achieve good results')
        elif tolerance < 1e-9:
            warnings.warn('Tolerance too tight. CVT algorithm may take long time to converge.')
        elif (tolerance < 0.1) and (tolerance > 1e-9):
            tolerance = tolerance
        else:
            raise Exception('Invalid tolerance input')
        self.eps = tolerance

    @staticmethod
    def random_sample_selection(no_samples, no_features):
        """
        Function generates a the required number of samples (no_samples) within an no_features-dimensional space.
        This is achieved by generating an m x n 2-D array using numpy's random.rand function, where
            - m = number of training samples to be generated, and'
            - n = number of design features/variables (dimensionality of the problem).
        Args:
            no_samples(int): The number of samples to be generated.
            no_features(int): Number of design features/variables in the input data.
        Returns:
            random_points(NumPy Array): 2-D array of size no_samples x no_features generated from a uniform distribution.
        Example: Generate three samples for a two-dimensional problem
            >>  rbf.CVTSampling.random_sample_selection(3, 2)
            >> array([[0.03149075, 0.70566624], [0.48319597, 0.03810093], [0.19962214, 0.57641408]])
        """
        random_points = np.random.rand(no_samples, no_features)
        return random_points

    @staticmethod
    def eucl_distance(u, v):
        """
        The function eucl_distance(u,v) calculates Euclidean distance between two points or arrays u and v.
        Args:
            u, v (NumPy Array): Two points or arrays with the same number of features (same second dimension)
        Returns:
            euc_d(NumPy Array): Array of size (u.shape[0] x 1) containing Euclidean distances.
        """
        d = u - v
        d_sq = d ** 2
        euc_d = np.sqrt(np.sum(d_sq, axis=1))
        return euc_d

    @staticmethod
    def create_centres(initial_centres, current_random_points, current_centres, counter):
        """
        The function create_centres generates new mass centroids for the design space based on McQueen's method.
        The mass centroids are created based on the previous mass centroids and the mean of random data sampling the design space.
            Args:
                initial_centres(NumPy Array): A 2-D array containing the current mass centroids, size no_samples x no_features.
                current_random_points(NumPy Array): A 2-D array containing several points generated randomly from within the design space.
                current_centres(NumPy Array): Array containing the index number of the closest mass centroid of each point in current_random_points, representing its class.
                counter(int): current iteration number
            Returns:
                centres(NumPy Array): A 2-D array containing the new mass centroids, size no_samples x no_features.
        The steps carried out in the function at each iteration are:
        (1) Classify the current random points in current_random_points based on their centres
        (2) Evaluate the mean of the random points in each class
        (3) Create the new centres as the weighted average of the current centres (initial_centres) and the mean data calculated in the second step. The weighting is done based on the number of iterations (counter).
        """
        centres = np.zeros((initial_centres.shape[0], initial_centres.shape[1]))
        current_centres = current_centres.reshape(current_centres.shape[0], 1)
        for i in range(0, initial_centres.shape[0]):
            data_matrix = current_random_points[current_centres[:, 0] == i]
            m_prime, n_prime = data_matrix.shape
            if m_prime == 0:
                centres[i, :] = np.mean(initial_centres, axis=0)
            else:
                centres[i, :] = np.mean(data_matrix, axis=0)

        # Weighted average based on previous number of iterations
        centres = ((counter * initial_centres) + centres) / (counter + 1)
        return centres

    def sample_points(self):
        """
        The ``sample_points`` method determines the best/optimal centre points (centroids) for a data set based on the minimization of the total distance between points and centres.
        Procedure based on McQueen's algorithm: iteratively minimize distance, and re-position centroids.
        Centre re-calculation done as the mean of each data cluster around each centre.
        Returns:
            NumPy Array or Pandas Dataframe:     A numpy array or Pandas dataframe containing the final **number_of_samples** centroids obtained by the CVT algorithm.
        """
        _, n = self.x_data.shape
        size_multiple = 1000
        initial_centres = self.random_sample_selection(self.number_of_centres, n)
        # Iterative optimization process
        cost_old = 0
        cost_new = 0
        cost_change = float('Inf')
        counter = 1
        while (cost_change > self.eps) and (counter <= 1000):
            cost_old = cost_new
            current_random_points = self.random_sample_selection(self.number_of_centres * size_multiple, n)
            distance_matrix = np.zeros(
                (current_random_points.shape[0], initial_centres.shape[0]))  # Vector to store distances from centroids
            current_centres = np.zeros(
                (current_random_points.shape[0], 1))  # Vector containing the centroid each point belongs to

            # Calculate distance between random points and centres, sort and estimate new centres
            for i in range(0, self.number_of_centres):
                distance_matrix[:, i] = self.eucl_distance(current_random_points, initial_centres[i, :])
            current_centres = (np.argmin(distance_matrix, axis=1))
            new_centres = self.create_centres(initial_centres, current_random_points, current_centres, counter)

            # Estimate distance between new and old centres
            distance_btw_centres = self.eucl_distance(new_centres, initial_centres)
            cost_new = np.sqrt(np.sum(distance_btw_centres ** 2))
            cost_change = np.abs(cost_old - cost_new)
            counter += 1
            # print(counter, cost_change)
            if cost_change >= self.eps:
                initial_centres = new_centres

        sample_points = new_centres

        unique_sample_points = self.sample_point_selection(self.data, sample_points, self.sampling_type)
        if len(self.data_headers) > 0:
            unique_sample_points = pd.DataFrame(unique_sample_points, columns=self.data_headers)
        return unique_sample_points


if __name__ == '__main__':
    dim = 4
    n_samples = 20
    cvt = CentroidalVoronoiTessellationSampling(tolerance=1e-2)
    cvt._do(dim, n_samples)
    x_lhs = cvt.x
    print(x_lhs)
    print(x_lhs.shape)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter(x_lhs[:, 0], x_lhs[:, 1], c='k')
    plt.show()
