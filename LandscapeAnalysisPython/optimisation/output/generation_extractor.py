import numpy as np
from pathlib import Path


class GenerationExtractor(object):
    def __init__(self, filename, base_path='./results/'):
        self.filename = filename   
        self.base_path = base_path
        self.full_path = str(Path(self.base_path).joinpath(self.filename + '.npy'))

        self.data_array = None
        self.population_array = None
        self.indicator_array = None
        self.constraints_array = None

    def add_generation(self, population, n_gen):
        # Extract variables and objectives from population
        n_pop = len(population)
        x_vars = np.atleast_2d(population.extract_var())
        obj_array = np.atleast_2d(population.extract_obj())
        cons_array = np.atleast_2d(population.extract_cons())

        # Check if population has constraints
        has_constraints = not np.all(cons_array.flatten() == None)

        # Store front data internally
        gen_arr = n_gen * np.ones(n_pop)[:, None]
        if has_constraints:
            new_data = np.hstack((gen_arr, x_vars, obj_array, cons_array))
        else:
            new_data = np.hstack((gen_arr, x_vars, obj_array))

        if self.population_array is None:
            self.population_array = new_data
        else:
            self.population_array = np.vstack((self.population_array, new_data))

    def add_front(self, population, n_gen, indicator_values=None, constraints=None):
        # Extract variables and objectives from population
        n_pop = len(population)
        x_vars = np.atleast_2d(population.extract_var())
        obj_array = np.atleast_2d(population.extract_obj())

        # Store front data internally
        gen_arr = n_gen * np.ones(n_pop)[:, None]
        new_data = np.hstack((gen_arr, x_vars, obj_array))

        if self.data_array is None:
            self.data_array = new_data
        else:
            self.data_array = np.vstack((self.data_array, new_data))

        if indicator_values is not None:
            indicator_values = np.atleast_2d(indicator_values)
            if self.indicator_array is None:
                self.indicator_array = indicator_values
            else:
                self.indicator_array = np.vstack((self.indicator_array, indicator_values))

        if constraints is not None:
            constraints = np.atleast_2d(constraints)
            if self.constraints_array is None:
                self.constraints_array = constraints
            else:
                self.constraints_array = np.vstack((self.constraints_array, constraints))

    def finalise_output(self):
        # Save data to file as a pickle in .npy format
        with open(self.full_path, 'wb') as f:
            np.save(f, self.population_array)   # Full population first
            np.save(f, self.data_array)         # Fronts next
            np.save(f, self.indicator_array)    # Indicator metric next
            np.save(f, self.constraints_array)  # Constraints last

        # with open(self.full_path, 'rb') as f:
        #     population = np.load(f)          # Full population first
        #     fronts = np.load(f)              # Fronts next
        #     indicator_values = np.load(f)    # Indicator metric next

    def load_output(self):
        with open(self.full_path, 'rb') as f:
            population = np.load(f)          # Full population first
            fronts = np.load(f)              # Fronts next
            indicator_values = np.load(f)    # Indicator metric next

        return population, fronts, indicator_values

    # Compatibility methods for external use (i.e. from tdeadp framework)
    def _add_generation(self, n_pop, vars, objs, n_gen, cons=None):
        # Extract variables and objectives from population
        x_vars = np.atleast_2d(vars)
        obj_array = np.atleast_2d(objs)
        cons_array = np.atleast_2d(cons)

        # Check if population has constraints
        has_constraints = not np.all(cons_array.flatten() == None)

        # Store front data internally
        gen_arr = n_gen * np.ones(n_pop)[:, None]
        if has_constraints:
            new_data = np.hstack((gen_arr, x_vars, obj_array, cons_array))
        else:
            new_data = np.hstack((gen_arr, x_vars, obj_array))

        if self.population_array is None:
            self.population_array = new_data
        else:
            self.population_array = np.vstack((self.population_array, new_data))

    def _add_front(self, n_pop, vars, objs, n_gen, indicator_values=None, constraints=None):
        # Extract variables and objectives from population
        x_vars = np.atleast_2d(vars)
        obj_array = np.atleast_2d(objs)

        # Store front data internally
        gen_arr = n_gen * np.ones(n_pop)[:, None]
        new_data = np.hstack((gen_arr, x_vars, obj_array))

        if self.data_array is None:
            self.data_array = new_data
        else:
            self.data_array = np.vstack((self.data_array, new_data))

        if indicator_values is not None:
            indicator_values = np.atleast_2d(indicator_values)
            if self.indicator_array is None:
                self.indicator_array = indicator_values
            else:
                self.indicator_array = np.vstack((self.indicator_array, indicator_values))

        if constraints is not None:
            constraints = np.atleast_2d(constraints)
            if self.constraints_array is None:
                self.constraints_array = constraints
            else:
                self.constraints_array = np.vstack((self.constraints_array, constraints))
