import numpy as np
from optimisation.setup import Setup
from optimisation.util.reference_directions import UniformReferenceDirection

import modact.problems as pb
import os

"""
A wrapper for the MODAct real-world problem sets
"""

AVAILABLE_CASES = [
    "cs1",
    "cs2",
    "cs3",
    "cs4",
    "ct1",
    "ct2",
    "ct3",
    "ct4",
    "cts1",
    "cts2",
    "cts3",
    "cts4",
    "ctse1",
    "ctse2",
    "ctse3",
    "ctse4",
    "ctsei1",
    "ctsei2",
    "ctsei3",
    "ctsei4",
]


class MODAct(Setup):
    """
    Multi-Objective Design of Actuators
    MODAct is a framework for real-world constrained multi-objective optimization.
    Refer to the python package https://github.com/epfl-lamd/modact from requirements.
    Best-known Pareto fronts must be downloaded from here: https://doi.org/10.5281/zenodo.3824302
    Parameters
    References:
    --------------------
    C. Picard and J. Schiffmann, “Realistic Constrained Multi-Objective Optimization Benchmark Problems from Design,”
    IEEE Transactions on Evolutionary Computation, pp. 1–1, 2020.
    """

    def __init__(self, problem_name="cs1"):
        problem_name = problem_name.lower()
        assert problem_name in AVAILABLE_CASES
        super().__init__()

        # List of available modact problems
        self.problem_name = problem_name
        self.avail_cases = AVAILABLE_CASES

        try:
            # Obtain problem from modact
            self.prob = pb.get_problem(self.problem_name)
        except Exception as e:
            print(e)

        # Extract problem info
        self.lb, self.ub = self.prob.bounds()
        self.lb, self.ub = np.array(self.lb), np.array(self.ub)

        self.dim = len(self.lb)
        self.n_objectives = len(self.prob.weights)
        self.n_constraints = len(self.prob.c_weights)
        self.n_int = 0

        # Minimisation weights
        self.weights = np.array(self.prob.weights)
        self.c_weights = np.array(self.prob.c_weights)

        # Initial guess
        self.initial_value = self.lb + 0.5 * (self.ub - self.lb)

        # Optimum Pareto Front
        # self.f_opt = -1.0 * np.genfromtxt(f"./cases/MODAct_files/{self.problem_name}_PF.dat", delimiter="") * self.weights
        self.f_opt = None
        self.var_opt = None

        if self.n_int == 0:
            self.int_var = np.array([])
            self.cont_var = np.arange(0, self.dim)
        else:
            raise Exception("Not setup to handle discrete variables!")

    def _calc_pareto_front(self):
        # New method added by Richard - had to slightly alter filepath.
        print(os.getcwd())
        return (
            -1.0
            * super()._calc_pareto_front(
                f"cases/MODAct_files/{self.problem_name.lower()}_PF.dat"
            )
            * self.weights
        )

    def evaluate(self, var):
        obj = self.obj_func_specific(var)
        cons = self.cons_func_specific(var)
        return obj, cons

    def set_variables(self, prob, **kwargs):
        prob.add_var_group(
            "x_vars",
            self.dim,
            "c",
            lower=self.lb,
            upper=self.ub,
            value=self.initial_value,
            scale=1.0,
        )  # TODO: initial value check

    def set_constraints(self, prob, **kwargs):
        prob.add_con_group(
            "con", self.n_constraints, lower=None, upper=None
        )  # TODO: check bounds

    def set_objectives(self, prob, **kwargs):
        for i in range(self.n_objectives):
            prob.add_obj(f"f_{i}")

    def set_pareto(self, prob, **kwargs):
        prob.add_pareto_set(prob, self.f_opt)

    def obj_func(self, x_dict, **kwargs):
        x = x_dict["x_vars"]
        obj, cons = self.obj_func_specific(x)
        performance = None

        return obj, cons, performance

    def obj_func_specific(self, x):
        f, g = self.prob(x)
        obj = np.array(f) * -1.0 * self.weights
        cons = np.array(g) * self.c_weights

        return obj, cons

    def cons_func_specific(self, x):
        _, g = self.prob(x)
        cons = np.array(g) * self.c_weights

        return cons


if __name__ == "__main__":
    case = "cs1"
    prob = MODAct(case)
    print(prob.lb, prob.ub)
