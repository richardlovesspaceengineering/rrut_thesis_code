#!/bin/bash

# Message describing the experimental setup.
desc_msg="Added adaptive walk features; using 95th percentile values for upper bound, min for lower bound. Adaptive walk normalisation values are just taken from the RW. Global normalisation values are separately computed."

# Problem suites
problemsCTP=("CTP1", "CTP2", "CTP3", "CTP4", "CTP5", "CTP6", "CTP7", "CTP8")
problemsMW=("MW1", "MW2", "MW3", "MW4", "MW5", "MW6", "MW7", "MW8", "MW9", "MW10", "MW11", "MW12", "MW13", "MW14")

# Dimensions to consider for each of the above.
n_dim=(2 5 10)

# Number of samples to run.
num_samples=30

# Modes are debug or eval.
mode="eval"

# Save full feature arrays. Aggregated feature arrays are always saved.
save_feature_arrays=true

# Define the log file and wipe it
log_file="features_evaluation.log"
> "$log_file"

# Print experimental setup.
echo -e "Experimental setup:\n$desc_msg\n" | tee -a "$log_file"

# Get hostname
pc1="megatron"
host="$(hostname)"
echo "Host is: $host" | tee -a "$log_file"

# Path info.
if [[ "$host" == *"$pc1"* ]]; then # megatrons
  PYTHON_SCRIPT="/home/kj66/Documents/Richard/venv/bin/python3"
  SCRIPT_PATH="/home/kj66/Documents/Richard/rrut_thesis_code/"
else # richard's pc
  PYTHON_SCRIPT="C:/Users/richa/anaconda3/envs/thesis_env_windows/python.exe"
  SCRIPT_PATH="C:/Users/richa/Documents/Thesis/rrut_thesis_code/"
fi
echo "Using interpreter: $PYTHON_SCRIPT" | tee -a "$log_file"

# Create unique temporary directory
temp_dir=$(mktemp -d -t ci-XXXXXXXXXX --tmpdir="$SCRIPT_PATH")
echo "New directory: $temp_dir" | tee -a "$log_file"

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
                echo "Cleaning up $temp_dir" | tee -a "$log_file"
                exit 0
        fi
}

# Set cwd if not already
cd "$cd_dir" || { echo "cd $cd_dir failed" | tee -a "$log_file"; }

# Path to execute python script
run_dir="$temp_dir"
run_dir+="/runner.py"    # Main script to execute (runner.py)
echo "File running inside: $run_dir" | tee -a "$log_file"

# Determine which set of problems to run based on user input
if [ "$1" == "MW" ]; then
  problems=("${problemsMW[@]}")
elif [ "$1" == "CTP" ]; then
  problems=("${problemsCTP[@]}")
else
  echo "Invalid argument. Use 'MW' or 'CTP'." | tee -a "$log_file"
  exit 1
fi

# Run. Note that all logging is done within Python.
for problem in "${problems[@]}"; do
  problem=$(echo "$problem" | sed 's/,$//')  # Remove trailing comma if it exists
  for dim in "${n_dim[@]}"; do
    echo "Running problem: $problem, dimension: $dim" | tee -a "$log_file"  # Print message to the terminal and log file
    "$PYTHON_SCRIPT" -u "$run_dir" "$problem" "$dim" "$num_samples" "$mode" "$save_feature_arrays" 2>&1 | tee -a "$log_file"
  done
done

# Clean up temp dir
rm -rf "$temp_dir"

# Exit
exit 0
