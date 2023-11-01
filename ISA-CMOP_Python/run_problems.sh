#!/bin/bash

# Problem suites
problems=("MW1", "MW2", "MW3", "MW4", "MW5", "MW6", "MW7" "MW11")

# Dimensions to consider for each of the above.
n_dim=(2 5 10)

# Number of samples to run.
num_samples=30

# Get hostname
pc1="megatron"
host="$(hostname)"
echo "Host is: $host"

# Path info. @Juan if you could add heuristics so it can easily swap between Richard's computer and the megatrons that'd be great!
if [[ "$host" == *"$pc1"* ]]; then # megatrons
  PYTHON_SCRIPT="/home/kj66/Documents/Richard/venv/bin/python3"
  SCRIPT_PATH="/home/kj66/Documents/Richard/rrut_thesis_code/"
else # richard's pc
  PYTHON_SCRIPT="D:/richa/anaconda3/envs/thesis_env_windows/python.exe"
  SCRIPT_PATH="d:/richa/Documents/Thesis/rrut_thesis_code/"
fi
echo "Using interpreter: $PYTHON_SCRIPT"

# Create unique temporary directory
temp_dir=$(mktemp -d -t ci-XXXXXXXXXX --tmpdir=$SCRIPT_PATH)
# echo "new dir: $temp_dir"

# Copy framework to temporary directory
copy_dir="$SCRIPT_PATH"
cd_dir="$SCRIPT_PATH"
cd_dir+="ISA-CMOP_Python/"
copy_dir+="ISA-CMOP_Python/*"
cp -R $copy_dir "$temp_dir"

# Handle CTRL+C event clean up
trap ctrl_c INT
function ctrl_c() {
        if [ -d "$temp_dir" ]; then
                # Clean up temp dir
                rm -rf "$temp_dir"
                echo "Cleaning up $temp_dir"
                exit 0
        fi
}

# Set cwd if not already
cd "$cd_dir" || { echo "cd $cd_dir failed"; }

# Path to execute python script
run_dir="$temp_dir"
run_dir+="/runner.py"    # Main script to execute (runner.py)
echo $"File running inside: $run_dir"

# Convert config inputs to a single string
problem_str=$(printf "%s," "${problems[@]}")
problem_str=${problem_str%,}  # Remove the trailing comma
n_dim_str=$(printf "%s," "${n_dim[@]}")
n_dim_str=${n_dim_str%,}  # Remove the trailing comma

# Run. Note that all logging is done within Python.
# TODO: This might have to change to a for loop call
"$PYTHON_SCRIPT" "$run_dir" "$problem_str" "$n_dim_str" "$num_samples"
