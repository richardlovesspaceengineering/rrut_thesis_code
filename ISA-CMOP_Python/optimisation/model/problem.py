import numpy as np
from mpi4py import MPI

from collections import OrderedDict

from optimisation.variable import Variable
from optimisation.constraint import Constraint
from optimisation.objective import Objective


class Problem(object):

    def __init__(self, name, obj_func=None,
                 map_internally=False, n_processors=1, comm=None):

        """
        :param name:
        :param obj_fun:
        :param surrogate_obj_fun:
        :param comm:
        """

        self.name = name

        # Objective & constraint functions
        self.obj_func = obj_func
        self.obj_func_specific = None
        self.cons_func_specific = None

        # Parallel mapping
        self.map_internally = map_internally
        self.n_processors = n_processors

        # Internal mapping
        self.pool = None

        # External mapping
        if comm is None:
            self.comm = MPI.COMM_WORLD
        else:
            self.comm = comm

        # Use ordered dictionaries to keep track of variables, constraints and objectives
        self.variables = OrderedDict()
        self.constraints = OrderedDict()
        self.objectives = OrderedDict()

        # Basic parameters
        self.n_var = 0
        self.n_con = 0
        self.n_obj = 0

        # Design variable limits
        self.x_lower = []
        self.x_upper = []

        # Specified variable set
        self.x_value = []
        self.x_value_additional = []

    def add_specific_funcs(self, setup):

        self.obj_func_specific = setup.obj_func_specific
        self.cons_func_specific = setup.cons_func_specific

    def add_var(self, name, *args, **kwargs):

        self.add_var_group(name, 1, *args, **kwargs)

    def add_var_group(self, name, n_vars, var_type='c', value=0.0, lower=None, upper=None, scale=1.0, choices=None,
                      additional_values=None):

        """
        :param name:
        :param n_vars:
        :param var_type:
        :param value:
        :param lower:
        :param upper:
        :param scale:
        :param additional_values:
        :param choices:
        :return:
        """

        # Check that n_vars > 0
        if n_vars < 1:
            raise Exception('Number of variables n_var must be greater than or equal to 1. '
                            'Variable in question is %s.' % name)

        # Check variable type
        if var_type not in ['c', 'i', 'd']:
            raise Exception('Variable type must be ''c'' (continuous), ''i'' (integer) or ''d'' (discrete)')

        # Process value argument
        value = np.atleast_1d(value).real
        if len(value) == 1:
            value = value[0]*np.ones(n_vars)
        elif len(value) == n_vars:
            pass
        else:
            raise Exception('The length of the "value" argument to '
                            'addVarGroup is %d, but the number of '
                            'variables in nVars is %d.' % (len(value), n_vars))

        # Process additional values argument
        if additional_values is None:
            additional_values = None
        else:
            for idx in range(len(additional_values)):
                additional_values[idx] = np.array(np.atleast_1d(additional_values))
                if len(additional_values[idx]) == 1:
                    additional_values[idx] = additional_values[idx][0]*np.ones(n_vars)
                elif len(additional_values[idx]) == n_vars:
                    pass
                else:
                    raise Exception('The length of the "additional value" argument to '
                                    'addVarGroup is %d, but the number of '
                                    'variables in nVars is %d.' % (len(additional_values[idx]), n_vars))

        # Process lower argument
        if lower is None:
            lower = [None for i in range(n_vars)]
        elif np.isscalar(lower):
            lower = lower*np.ones(n_vars)
        elif len(lower) == n_vars:
            lower = np.atleast_1d(lower).real
        else:
            raise Exception('The "lower" argument to addVarGroup is '
                            'invalid. It must be None, a scalar, or a '
                            'list/array or length nVars=%d.' % n_vars)

        # Process upper argument
        if upper is None:
            upper = [None for i in range(n_vars)]
        elif np.isscalar(upper):
            upper = upper*np.ones(n_vars)
        elif len(upper) == n_vars:
            upper = np.atleast_1d(upper).real
        else:
            raise Exception('The "upper" argument to addVarGroup is '
                            'invalid. It must be None, a scalar, or a '
                            'list/array or length nVars=%d.' % n_vars)

        # Process scale argument
        if scale is None:
            scale = np.ones(n_vars)
        else:
            scale = np.atleast_1d(scale)
            if len(scale) == 1:
                scale = scale[0]*np.ones(n_vars)
            elif len(scale) == n_vars:
                pass
            else:
                raise Exception('The length of the "scale" argument to '
                                'addVarGroup is %d, but the number of '
                                'variables in nVars is %d.' % (len(scale), n_vars))

        # Create a list of all the variable objects
        var_list = []
        for idx in range(n_vars):
            var_name = name + '_' + str(idx)
            if additional_values is None:
                _additional_values = additional_values
            else:
                _additional_values = np.asarray(additional_values)[:, idx]
            var_list.append(Variable(name=var_name, type=var_type, value=value[idx],
                                     lower=lower[idx], upper=upper[idx], scale=scale[idx], choices=choices,
                                     additional_values=_additional_values))

        # Check that variable name does not already exist
        if name in self.variables:
            # Check that the variables happen to be the same
            err = False
            if not len(self.variables[name]) == len(var_list):
                raise Exception('The supplied name "%s" for a variable group '
                                'has already been used!' % name)
            for i in range(len(var_list)):
                if not var_list[i] == self.variables[name][i]:
                    raise Exception('The supplied name "%s" for a variable group '
                                    'has already been used!' % name)
        else:
            # Finally we set the dict entry using variable list
            self.variables[name] = var_list

    def add_con(self, name, *args, **kwargs):

        self.add_con_group(name, 1, *args, **kwargs)

    def add_con_group(self, name, n_con, lower=None, upper=None):

        # Check that constraint name does not already exist
        if name in self.constraints:
            raise Exception('The supplied name "%s" for a constraint group '
                            'has already been used.' % name)

        # Add constraint object to dict entry
        self.constraints[name] = Constraint(name=name, n_con=n_con, lower=lower, upper=upper)

    def add_obj(self, name, *args, **kwargs):

        # Add objective object to dict entry
        self.objectives[name] = Objective(name=name, *args, **kwargs)

    def finalise_design_variables(self):

        _key = list(self.variables.keys())[0]
        if self.variables[_key][0].additional_values is not None:
            n_additional_var_sets = len(self.variables[_key][0].additional_values)
            self.x_value_additional = [[] for _ in range(n_additional_var_sets)]
        else:
            self.x_value_additional = None

        for var_group in self.variables:

            n = len(self.variables[var_group])
            self.n_var += n

            for var_idx in range(n):
                self.x_lower.append(self.variables[var_group][var_idx].lower)
                self.x_upper.append(self.variables[var_group][var_idx].upper)
                self.x_value.append(self.variables[var_group][var_idx].value)
                if self.variables[var_group][var_idx].additional_values is not None:
                    for i in range(len(self.variables[var_group][var_idx].additional_values)):
                        self.x_value_additional[i].append(self.variables[var_group][var_idx].additional_values[i])

        # Convert lower and upper limit lists to numpy arrays
        self.x_lower = np.asarray(self.x_lower)
        self.x_upper = np.asarray(self.x_upper)

        # Convert specified design variable values to numpy array
        self.x_value = np.asarray(self.x_value)

        # Convert additional design variable sets to single numpy array
        if self.x_value_additional is not None:
            self.x_value_additional = np.asarray(self.x_value_additional)
            if len(self.x_value_additional[0]) != len(self.x_value):
                raise Exception('The size of the additional value variable array(s) '
                                'does not match the number of variables')

    def finalise_constraints(self):

        for cons_group in self.constraints:
            self.n_con += self.constraints[cons_group].n_con

    def finalise_objectives(self):

        for obj in self.objectives:
            self.n_obj += 1

    def finalise(self):

        self.finalise_design_variables()
        self.finalise_constraints()
        self.finalise_objectives()

