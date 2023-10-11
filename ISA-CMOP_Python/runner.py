import json
from ProblemEvaluator import ProblemEvaluator

# Import the get_problem method from pymoo.problems
from pymoo.problems import get_problem


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


if __name__ == "__main__":
    json_file_path = "problems_to_run.json"

    # Load the JSON configuration
    json_config = load_json_config(json_file_path)

    # Generate instances
    instances = generate_instances_from_config(json_config)

    # You now have a list of tuples with (instance name, problem instance)
    for instance_name, problem_instance in instances:
        print(f"Instance Name: {instance_name}")
        print(f"Problem:\n{problem_instance}")
        print("\n")

    evaluator = ProblemEvaluator(instances)
    evaluator.do(num_samples=3)
