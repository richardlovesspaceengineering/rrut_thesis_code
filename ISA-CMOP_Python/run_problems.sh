# Problem suites
problems=("MW1", "MW2", "MW3", "MW4", "MW5", "MW6", "MW7" "MW11")

# Dimensions to consider for each of the above.
n_dim=(2 5 10)

# Number of samples to run.
num_samples=30

# Path info. @Juan if you could add heuristics so it can easily swap between Richard's computer and the megatrons that'd be great!
PYTHON_SCRIPT="D:/richa/anaconda3/envs/thesis_env_windows/python.exe"
SCRIPT_PATH="d:/richa/Documents/Thesis/rrut_thesis_code/ISA-CMOP_Python/runner.py"

# Convert config inputs to a single string
problem_str=$(printf "%s," "${problems[@]}")
problem_str=${problem_str%,}  # Remove the trailing comma
n_dim_str=$(printf "%s," "${n_dim[@]}")
n_dim_str=${n_dim_str%,}  # Remove the trailing comma

# Run. Note that all logging is done within Python.
"$PYTHON_SCRIPT" "$SCRIPT_PATH" "$problem_str" "$n_dim_str" "$num_samples"