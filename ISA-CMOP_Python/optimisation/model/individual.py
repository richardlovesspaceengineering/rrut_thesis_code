import numpy as np
from collections import OrderedDict


class Individual(object):

    def __init__(self, problem):

        # Design variables
        self.var = np.zeros(problem.n_var)
        self.var_dict = OrderedDict()

        # Objective and constraint values
        self.obj = 100.0*np.ones(problem.n_obj)
        self.cons = np.zeros(problem.n_con)
        self.cons_sum = 0.0
        self.cons_viol = 0.0 # based on Shi2019

        # Rank, crowding distance & hypervolume
        self.rank = np.nan
        self.crowding_distance = 0.0
        self.hypervolume = 0.0

        # Performance
        self.performance = []

    def set_var(self, problem, var):

        # Set variable values
        self.var = var

        # Set ordered dict according to design variable sequence in problem
        current_var_idx = 0
        for key in problem.variables.keys():
            n = len(problem.variables[key])
            self.var_dict[key] = np.copy(self.var[current_var_idx:current_var_idx+n])
            current_var_idx += n

    def descale_var(self, problem):

        # De-scale variables in ordered dict according to design variable sequence in problem
        for key in problem.variables.keys():
            temp = []
            for var_idx in range(len(problem.variables[key])):
                if problem.variables[key][var_idx].type == 'c':
                    self.var_dict[key][var_idx] = self.var_dict[key][var_idx]/problem.variables[key][var_idx].scale
                elif problem.variables[key][var_idx].type == 'i':
                    self.var_dict[key][var_idx] = round(self.var_dict[key][var_idx]/problem.variables[key][var_idx].scale)
                elif problem.variables[key][var_idx].type == 'd':
                    idx = int(round(self.var_dict[key][var_idx]/problem.variables[key][var_idx].scale))
                    temp.append(problem.variables[key][var_idx].choices[idx])
                    if var_idx == len(problem.variables[key]) - 1:
                        self.var_dict[key] = temp
                else:
                    raise Exception('Unknown variable type: ', problem.variables[key][var_idx].type)

    def scale_var(self, problem):

        # Scale variables in ordered dict according to design variable sequence in problem
        for key in problem.variables.keys():
            temp = []
            for var_idx in range(len(problem.variables[key])):
                if problem.variables[key][var_idx].type == 'c':
                    self.var_dict[key][var_idx] = self.var_dict[key][var_idx]*problem.variables[key][var_idx].scale
                elif problem.variables[key][var_idx].type == 'i':
                    self.var_dict[key][var_idx] = np.double(self.var_dict[key][var_idx])*problem.variables[key][var_idx].scale
                elif problem.variables[key][var_idx].type == 'd':
                    temp.append(problem.variables[key][var_idx].choices.index(self.var_dict[key][var_idx]))
                    if var_idx == len(problem.variables[key]) - 1:
                        self.var_dict[key] = np.asarray(np.double(temp))*problem.variables[key][var_idx].scale
                else:
                    raise Exception('Unknown variable type: ', problem.variables[key][var_idx].type)

