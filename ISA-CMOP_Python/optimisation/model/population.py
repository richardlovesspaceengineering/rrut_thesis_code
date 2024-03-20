import numpy as np
import multiprocessing
from multiprocessing_util import *

from optimisation.model.individual import Individual

from optimisation.util.non_dominated_sorting import NonDominatedSorting
from optimisation.util.calculate_crowding_distance import calculate_crowding_distance
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.ticker import FuncFormatter
import time
import math


class Population(np.ndarray):
    def __new__(cls, problem, n_individuals=0):
        obj = super(Population, cls).__new__(cls, n_individuals, dtype=cls).view(cls)
        for i in range(n_individuals):
            obj[i] = Individual(problem)

        return obj

    @classmethod
    def merge(cls, a, b):
        if isinstance(a, Population) and isinstance(b, Population):
            if len(a) == 0:
                return b
            elif len(b) == 0:
                return a
            else:
                obj = np.concatenate([a, b]).view(Population)
                return obj
        else:
            raise Exception("Both a and b must be Population instances")

    @classmethod
    def merge_multiple(cls, *args):
        base = None
        for item in args:
            if isinstance(item, Population):
                if isinstance(base, Population):
                    base = np.concatenate([base, item]).view(Population)
                else:
                    base = item

        return base

    def get_single_pop(self, index):
        result = self[index]  # Utilize the built-in __getitem__ for extraction
        # Ensure the result is a Population instance, especially for single item extraction
        if not isinstance(result, Population):
            result = np.array([result])
            result = result.view(Population)

        return result

    def remove_nan_inf_rows(self, pop_type):
        """
        pop_type is "neig" or "walk" or "global"
        """

        # Extract evaluated population values.
        var = self.extract_var()
        obj = self.extract_obj()
        cons = self.extract_cons()

        # Get indices of rows without NaN or infinity in the objective array
        nan_inf_idx = self.get_nan_inf_idx()
        num_rows_removed = len(nan_inf_idx)

        # If there are rows to remove, slice directly to exclude them
        if num_rows_removed != 0:

            # Calculate indices to keep
            all_indices = np.arange(len(self))
            keep_indices = np.setdiff1d(all_indices, nan_inf_idx)

            # Directly slice the Population instance to keep only valid individuals
            new_pop = self[keep_indices]

            return new_pop, num_rows_removed
        else:
            return self, 0

    ### GETTERS
    def extract_attribute(self, attr_name):
        """
        General-use function to extract a specified attribute from each individual in the population.

        Parameters:
        - attr_name (str): The name of the attribute to extract.

        Returns:
        - np.ndarray: An array of the extracted attribute values. Returns an empty numpy array if the population is empty.
        """
        # Check if the population is empty before proceeding
        if len(self) == 0:
            return np.array([])  # Return an empty numpy array

        # Use list comprehension to extract the attribute from each individual
        attr_list = [getattr(ind, attr_name) for ind in self]

        # If the first attribute is single-valued (e.g., a float or int), we assume all are, and use np.array directly
        if np.ndim(attr_list[0]) == 0:
            return np.atleast_2d(np.array(attr_list))
        # Otherwise, we handle multi-valued attributes (e.g., arrays) differently
        else:
            return np.atleast_2d(np.vstack(attr_list))

    def extract_var(self):
        return self.extract_attribute("var")

    def extract_obj(self):
        return self.extract_attribute("obj")

    def extract_cons(self):
        return self.extract_attribute("cons")

    def extract_cv(self):
        return self.extract_attribute("cv").reshape(
            -1, 1
        )  # Ensure n x 1 shape for consistency

    def extract_rank(self):
        return self.extract_attribute("rank").flatten()

    def extract_uncons_rank(self):
        return self.extract_attribute("rank_uncons").flatten()

    def extract_crowding(self):
        return self.extract_attribute("crowding_distance")

    def extract_pf(self, max_points=None):
        pareto_front = self[0].problem._calc_pareto_front()

        if max_points is not None and 0 < max_points < len(pareto_front):
            interval = math.ceil(len(pareto_front) / max_points)
            pareto_front = pareto_front[::interval]

        return pareto_front

    def extract_bounds(self):
        return self[0].bounds

    def extract_nondominated(self, constrained=True):
        """
        Extract non-dominated solutions from the population.

        Can only run once all objectives, constraints have been evaluated i.e after a call to self.evaluate(x).

        Creates a new population which is a subset of the original.
        """
        # Number of best-ranked solutions.
        if constrained:
            rank_array = self.extract_rank()
        else:
            rank_array = self.extract_uncons_rank()

        best_rank = np.min(
            rank_array
        )  # usually 1; could be other than 1 if we have trimmed some points.

        # Find indices of individuals with the best rank (non-dominated)
        best_indices = np.where(rank_array == best_rank)[0]

        # Use boolean indexing to directly select non-dominated solutions from the population
        nondominated_population = self[best_indices].view(Population)

        return nondominated_population

    def extract_feasible(self):
        """
        Extract feasible solutions from the population using a vectorized approach.
        """
        # Assuming `extract_cv()` is a method that returns a numpy array of cv values for the entire population
        cv_values = self.extract_cv()

        # Find indices of feasible solutions (where cv <= 0)
        feasible_indices = np.where(cv_values <= 0)[0]

        # Directly use feasible_indices to select feasible solutions from the population
        feasible_population = self[feasible_indices].view(Population)

        return feasible_population

    def eval_fronts(self, constrained):
        """
        Place each individual on a front.
        """
        obj_array = self.extract_obj()

        if constrained:
            cons_val = self.extract_cv()
        else:
            cons_val = None

        fronts = NonDominatedSorting().do(
            obj_array,
            cons_val=cons_val,
            n_stop_if_ranked=obj_array.shape[0],
            return_rank=False,
        )

        return fronts

    def eval_rank_and_crowding(self):
        """
        Evaluate the rank and crowding of each individual within the population.
        """

        # Constrained fronts.
        fronts_cons = self.eval_fronts(constrained=True)

        # Cycle through fronts
        for k, front in enumerate(fronts_cons):
            # Calculate crowding distance of the front

            # LEFT OUT TO SPEED UP COMPUTATION
            # front_crowding_distance = calculate_crowding_distance(obj_array[front, :])

            # Save rank and crowding to the individuals
            for j, i in enumerate(front):
                self[i].rank = k + 1  # lowest rank is 1
                # self[i].crowding_distance = front_crowding_distance[j]
                self[i].crowding_distance = None

    def eval_unconstrained_rank(self):
        # Unconstrained fronts.
        fronts_uncons = self.eval_fronts(constrained=False)

        # Cycle through fronts
        for k, front in enumerate(fronts_uncons):
            # Save to the individuals
            for j, i in enumerate(front):
                self[i].rank_uncons = k + 1  # lowest rank is 1

    def evaluate_individual(self, individual, var_array):
        """
        Function to evaluate a single individual.
        Arguments:
            individual: The individual to evaluate.
            var_array: The array of variables for evaluation.
        Returns:
            Tuple containing the individual's index, evaluated variables, objectives, constraints, and constraint violation.
        """
        # Assign decision variables.
        individual.var = var_array

        # Run evaluation of objectives, constraints, and CV.
        individual.eval_instance()

        return individual

    def evaluate_chunks(self, individuals, var_array):

        for i, individual in enumerate(individuals):
            individuals[i] = self.evaluate_individual(individual, var_array[i, :])
            print(f"Evaluated individual {i + 1} of {len(var_array)}.")

        return individuals

    @handle_ctrl_c
    def evaluate(self, var_array, eval_fronts, num_processes=1, show_msg=False):

        vectorized = True

        if (
            self[0]
            .problem.problem_name.lower()
            .startswith(("xa", "lircmop", "ct", "cs"))
            and "ctp" not in self[0].problem.problem_name.lower()
        ):
            vectorized = False

        start_time = time.time()
        if vectorized:
            # Evaluate vectorized.
            obj, cons = self[0].problem.evaluate(var_array)

            # Assign to individuals.
            for i in range(len(self)):
                self[i].var = var_array[i, :]
                self[i].obj = obj[i, :]
                self[i].cons = cons[i, :]
                self[i].cv = self[i].eval_cv()

            # Clear all unnecessary variables from memory.
            del var_array, obj, cons
        else:
            if num_processes > 1:
                # Parallel evaluation
                print(
                    f"Evaluating population of size {len(self)} in parallel with {num_processes} processes."
                )
                with multiprocessing.Pool(
                    processes=num_processes, initializer=init_pool
                ) as pool:

                    if self[0].problem.problem_name.lower().startswith(("xa")):

                        # Parallel processing
                        results = pool.starmap(
                            self.evaluate_individual,
                            [(self[i], var_array[i, :]) for i in range(len(self))],
                        )

                        # Merge back
                        for i, result in enumerate(results):
                            self[i] = result
                    else:
                        print("Using chunked evaluation...")
                        # Split var_array into chunks
                        var_array_chunks = np.array_split(var_array, num_processes)

                        # Split self into chunks; since self is a list, we use array_split from numpy and then convert each chunk back to a list
                        self_chunks = np.array_split(self, num_processes)

                        # Create a list of tuples where each tuple contains a chunk of self and the corresponding chunk of var_array
                        # Here, each chunk is zipped together since they are of equal length
                        args = [
                            (self_chunk, var_chunk)
                            for self_chunk, var_chunk in zip(
                                self_chunks, var_array_chunks
                            )
                        ]

                        # Use pool.starmap to parallelize
                        results = pool.starmap(self.evaluate_chunks, args)

                        # Instead of self = Population.merge_multiple(*results)
                        merged_population = Population.merge_multiple(*results)

                        # Update the original population with the merged results
                        for i in range(len(merged_population)):
                            self[i] = merged_population[i]

            else:
                for i in range(len(self)):
                    # Assign decision variables.
                    self[i].var = var_array[i, :]

                    # Run evaluation of objectives, constraints and CV.
                    self[i].eval_instance()

        end_time = time.time()
        if show_msg:
            print(
                f"Evaluated population of size {len(self)} in {end_time - start_time:.2f} seconds."
            )

        if eval_fronts:
            self.evaluate_fronts()

    def evaluate_fronts(self, show_time=False):
        start_time = time.time()

        # Now can find rank and crowding of each individual.
        self.eval_rank_and_crowding()

        # Unconstrained ranks
        self.eval_unconstrained_rank()
        end_time = time.time()

        if show_time:
            print(
                f"Evaluated ranks (size {len(self)}) in {end_time - start_time:.2f} seconds."
            )

    def is_ranks_evaluated(self):
        ranks = self.extract_rank()
        uncons_ranks = self.extract_uncons_rank()

        # Check if the length of the arrays matches the length of self
        # and if there are no NaN values in the arrays
        # TODO: might need to make this more robust for aerofoils.
        return (len(ranks) == len(self) and not np.all(np.isnan(ranks))) and (
            len(uncons_ranks) == len(self) and not np.all(np.isnan(uncons_ranks))
        )

    # Plotters
    def var_scatterplot_matrix(self, bounds=None):
        """
        Create a scatterplot matrix for each pairwise combination of decision variables, all stored in one numpy array.

        Parameters:
        - data: numpy array (n x n) containing the data for the scatterplot matrix.
        - bounds: 2D array (2 x n) specifying the upper and lower bounds for each axis (default is None).

        Returns:
        - None (displays the scatterplot matrix).
        """
        # Convert the numpy array to a Pandas DataFrame for Seaborn
        data = self.extract_var()
        df = pd.DataFrame(data)

        # Create the scatterplot matrix using Seaborn
        sns.set(style="ticks")
        g = sns.pairplot(df, diag_kind="kde", markers="o")

        if bounds is None:
            bounds = self.extract_bounds()

        # Set the x-axis and y-axis limits for each pair plot.
        for i in range(len(bounds[0])):
            for j in range(len(bounds[0])):
                g.axes[i, j].set_xlim(bounds[0][j], bounds[1][j])
                g.axes[i, j].set_ylim(bounds[0][i], bounds[1][i])

                # Add boxes
                g.axes[i, j].spines["top"].set_visible(True)
                g.axes[i, j].spines["right"].set_visible(True)
                g.axes[i, j].spines["bottom"].set_visible(True)
                g.axes[i, j].spines["left"].set_visible(True)
                if i != j:
                    # Format the axis labels to one decimal place
                    g.axes[i, j].xaxis.set_major_formatter(
                        FuncFormatter(lambda x, _: f"{x:.1f}")
                    )
                    g.axes[i, j].yaxis.set_major_formatter(
                        FuncFormatter(lambda x, _: f"{x:.1f}")
                    )

                    # Add grid and boxes to all off-diagonal plots
                    g.axes[i, j].grid(True)

        # Display the plot
        plt.show()

    # Setters.
    def set_obj(self, obj):
        """
        Set objective values. Useful when an external transformation has been applied.
        """

        assert obj.shape[0] == len(
            self
        ), "The shape of 'obj' must match the number of elements in the Population list."

        for i in range(len(self)):
            self[i].set_obj(obj[i, :])

    def get_nan_inf_idx(self):
        # Extract evaluated population values.
        obj = self.extract_obj()

        # Find indices of rows with NaN or infinity in the objective array
        nan_inf_idx = np.logical_or(
            np.isnan(obj).any(axis=1), np.isinf(obj).any(axis=1)
        )

        return np.where(nan_inf_idx)[0]

    def get_infeas_idx(self):
        cv = self.extract_cv()
        infeasible_indices = np.where(cv > 0)[0]
        return infeasible_indices

    # CSV writers.
    def write_var_to_csv(self, filename):
        np.savetxt(filename, self.extract_var())

    def write_obj_to_csv(self, filename):
        np.savetxt(filename, self.extract_obj())

    def write_cons_to_csv(self, filename):
        np.savetxt(filename, self.extract_cons())

    def write_cv_to_csv(self, filename):
        np.savetxt(filename, self.extract_cv())

    def write_rank_to_csv(self, filename):
        np.savetxt(filename, self.extract_rank())

    def write_rank_uncons_to_csv(self, filename):
        np.savetxt(filename, self.extract_uncons_rank())

    def write_pf_to_csv(self, filename):
        np.savetxt(filename, self.extract_pf())

    def save_population_attributes(self, file_path):
        """
        Save the population's variables, objectives, ranks, and constraint violations to a numpy file.

        Parameters:
        - sample_number: The identifier for the sample, used in naming the saved file.
        - directory: The directory where the file will be saved. Defaults to "./population_data".
        """

        # Extract attributes
        var = self.extract_var()
        obj = self.extract_obj()
        cons = self.extract_cons()
        rank = self.extract_rank()
        rank_uncons = self.extract_uncons_rank()
        cv = self.extract_cv()

        # Save the attributes to a numpy file
        np.savez(
            file_path,
            var=var,
            obj=obj,
            cons=cons,
            rank=rank,
            rank_uncons=rank_uncons,
            cv=cv,
        )

    @classmethod
    def from_saved_attributes(cls, file_path, problem):
        """
        Create a new Population instance from saved variables, objectives, ranks, and constraint violations.

        Parameters:
        - file_path: Path to the .npz file containing the saved attributes.
        - problem: The problem instance associated with the population. It's necessary for creating Individual instances.

        Returns:
        - A new Population instance with individuals having attributes set as per the saved data.
        """

        # Load the attributes from the .npz file
        data = np.load(file_path)
        var = data["var"]
        obj = data["obj"]
        cons = data["cons"]
        rank = data["rank"]
        rank_uncons = data["rank_uncons"]
        cv = data["cv"]

        # Initialize a new Population instance with the appropriate size
        new_population = cls(problem, n_individuals=var.shape[0])

        # Assign the loaded attributes to the individuals in the new population
        for i, individual in enumerate(new_population):

            # Saving values
            individual.var = var[i]
            individual.obj = obj[i]
            individual.cons = cons[i]
            individual.rank = rank[i]
            individual.rank_uncons = rank_uncons[i]
            individual.cv = cv[i]

        return new_population
