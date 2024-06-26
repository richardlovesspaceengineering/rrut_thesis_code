import json
import re
import sys
from ProblemEvaluator import ProblemEvaluator
from PreSampler import PreSampler

# Import the get_problem method from pymoo.problems
from pymoo.problems import get_problem
import cases.LIRCMOP_setup
from cases.PLATEMO_setup import PlatEMOSetup, RWCMOPSetup
from pathlib import Path
import numpy as np
import socket

import matlab.engine


def append_aerofoil_path():

    hostname = socket.gethostname()
    if hostname == "megatron2":
        sys_path = str(
            Path("/home/kj66/Documents/Richard/AirfoilBenchmarkSuite/").expanduser()
        )
    else:
        sys_path = str(
            Path("C:/Users/richa/Documents/Thesis/AirfoilBenchmarkSuite/").expanduser()
        )
    sys.path.append(sys_path)


# Load the JSON configuration from the file
def load_json_config(json_file):
    with open(json_file, "r") as file:
        json_config = json.load(file)
    return json_config


# Function to generate instances
def generate_instances_from_config(json_config):
    instances = []

    for suite_name, suite_data in json_config["suites"].items():
        for problem_num, problem_data in suite_data.items():
            n_var_list = problem_data["n_var"]

            # Construct the complete class name (e.g., "MW1" or "MW2")
            problem_name = f"{suite_name}{problem_num}"

            # Create instances with updated n_var
            for n_var in n_var_list:
                # Get the problem class. Needs to happen in the loop or else each step here will overwrite existing entries in instances.
                problem = get_problem(problem_name, n_var=n_var)

                instance_string = f"{problem_name}_d{n_var}"
                instances.append((instance_string, problem))

    return instances


def generate_suite_structure(benchmark_problem_names, dimensions):
    suite_structure = {}

    for problem_name in benchmark_problem_names:
        suite_name, problem_number = re.split("(\d+)", problem_name)[:-1]

        if suite_name not in suite_structure:
            suite_structure[suite_name] = {}

        for dimension in dimensions:
            if problem_number not in suite_structure[suite_name]:
                suite_structure[suite_name][problem_number] = {"n_var": []}
            suite_structure[suite_name][problem_number]["n_var"].append(dimension)

    print("Generated JSON file for the problem configurations")

    return {"suites": suite_structure}


def generate_platemo_instance(problem_name, n_var):
    problem_name = problem_name


# Function to generate a single problem instance
def generate_instance(problem_name, n_var):
    problem_name = problem_name.lower()

    # Check if problem_name contains 'DASCMOP'
    if problem_name.startswith(("cs", "ct")) and "ctp" not in problem_name:
        from cases.MODAct_setup import MODAct

        problem = MODAct(problem_name)
        print("MODAct problem selected - note that these are fixed 20D problems.")

        problem.n_var = problem.dim
        problem.n_obj = problem.n_objectives
        problem.n_constr = problem.n_constraints
        problem.xl = problem.lb
        problem.xu = problem.ub

    elif "lircmop" in problem_name:
        problem = getattr(cases.LIRCMOP_setup, problem_name.upper())(n_dim=n_var)

        # Helps with downstream naming issues
        problem.n_var = problem.dim
        problem.n_obj = problem.n_objectives
        problem.n_constr = problem.n_constraints
        problem.xl = problem.lb
        problem.xu = problem.ub

    elif problem_name.startswith("xa"):
        append_aerofoil_path()

        from test_problems.xairfoil import get_problem as get_problem_airfoil

        problem = get_problem_airfoil(
            instance=problem_name.upper(),
            dimension=n_var,
            solver="xfoil",
            impute_values=False,
        )

        # Helps with downstream naming issues
        problem.n_constr = problem.n_con
        problem.xl = problem.lb
        problem.xu = problem.ub

    elif problem_name.startswith(("cf", "sdc", "rwmop")):

        # Instantiate
        if n_var:
            problem = PlatEMOSetup(problem_name, n_var)
        else:
            problem = RWCMOPSetup(problem_name)

        n_var = problem.n_var

    else:

        if "dascmop" in problem_name:
            problem = get_problem(problem_name, n_var=n_var, difficulty=8)
        else:
            problem = get_problem(problem_name, n_var=n_var)

    problem.problem_name = problem_name.upper()

    instance_string = f"{problem_name.upper()}_d{n_var}"
    return problem, instance_string


def main():
    if len(sys.argv) != 9:
        print(
            "Usage: python runner.py problem_name n_dimensions num_samples mode save_features_array results_dir temp_pops_dir num_cores"
        )
        return

    problem_name = sys.argv[1].replace(",", "")
    if sys.argv[2]:
        n_var = int(sys.argv[2])
    else:
        n_var = None
    num_samples = int(sys.argv[3])
    mode = sys.argv[4].replace(",", "")
    results_dir = str(sys.argv[6])
    temp_pops_dir = str(sys.argv[7])
    num_cores = int(sys.argv[8])

    if sys.argv[5].lower() == "true":
        save_arrays = True
    else:
        save_arrays = False

    problem, instance_string = generate_instance(problem_name, n_var)
    evaluator = ProblemEvaluator(
        problem, instance_string, mode, results_dir, num_samples, num_cores
    )
    evaluator.do(save_arrays, temp_pops_dir)


if __name__ == "__main__":
    main()
