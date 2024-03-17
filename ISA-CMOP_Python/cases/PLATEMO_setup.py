import numpy as np
import matlab.engine
from optimisation.setup import Setup
import os


class PlatEMOSetup(Setup):

    def __init__(self, problem_name, n_var):

        self.start_matlab_session()

        self.problem_name = problem_name.upper()

        self.initialize_matlab_objects(n_var)
        self.generate_pareto_front()

        # Save attributes.
        self.n_var = n_var
        self.n_obj = int(self.get_attribute_from_matlab_object(self.prob, "M"))
        self.n_ieq_constr = 1  # only constraint violation is output
        self.n_constr = self.n_ieq_constr  # only constraint violation is output
        self.xl, self.xu = self.get_attribute_from_matlab_object(
            self.prob, "lower"
        ), self.get_attribute_from_matlab_object(self.prob, "upper")

    def initialize_matlab_objects(self, n_var):
        # Instantiate the MATLAB object for the problem.
        problem_function = getattr(self.matlab_engine, self.problem_name)
        self.prob = problem_function("D", n_var)

    def insert_matlab_objects(self, matlab_prob, matlab_engine):
        self.matlab_engine = matlab_engine
        self.prob = matlab_prob

    def pop_matlab_objects(self):
        prob = self.prob
        matlab_engine = self.matlab_engine
        self.prob = None
        self.matlab_engine = None
        return prob, matlab_engine

    def get_attribute_from_matlab_object(self, obj, attribute_name):
        return np.array(self.matlab_engine.getfield(obj, attribute_name))

    def generate_pareto_front(self, n_points=1000):
        file_path = f"./cases/PlatEMO_files/{self.problem_name}.pf"

        # Check if the file already exists
        if not os.path.exists(file_path):
            pf = np.array(self.matlab_engine.GetOptimum(self.prob, float(n_points)))
            np.savetxt(file_path, pf)
            print(f"Generated PF for {self.problem_name}")
        else:
            print(f"PF file for {self.problem_name} already exists.")

    def _calc_pareto_front(
        self,
    ):
        return super()._calc_pareto_front(
            f"./cases/PlatEMO_files/{self.problem_name}.pf"
        )

    def evaluate(self, x):

        x = matlab.double(x.tolist())
        pop_obj = self.matlab_engine.Evaluation(self.prob, x)

        return self.get_attribute_from_matlab_object(pop_obj, "objs"), np.atleast_2d(
            self.get_attribute_from_matlab_object(pop_obj, "cons")
        )

    def end_matlab_session(self):
        print(f"Closing MATLAB engine for PlatEMO instance.")
        self.matlab_engine.quit()
        self.matlab_engine = None
        self.is_matlab_on = False

    def start_matlab_session(self):
        print(f"Starting MATLAB engine for PlatEMO instance.")
        self.matlab_engine = matlab.engine.start_matlab()

        # Add PlatEMO cases to path.
        self.matlab_engine.addpath(
            self.matlab_engine.genpath(
                "~/Documents/Richard/rrut_thesis_code/PlatEMO/PlatEMO_Problems"
            ),
            nargout=0,
        )
        self.is_matlab_on = True
