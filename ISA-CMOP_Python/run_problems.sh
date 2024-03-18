#!/bin/bash

# Message describing the experimental setup.
desc_msg="Running all 5D, then 10D and so on from JSON file."

# problemsRW=("Truss2D", "WeldedBeam") # not scalable in Pymoo

# Number of samples to run. Will max out at 10 for aerofoil problems (handled in Python).
num_samples=30

# Modes are debug or eval.
mode="eval"
# mode="debug"

# Use pre-generated samples?
regenerate_samples=false #@JUAN set to true if you need to generate/can't see the pregen_samples folder as a sibling folder.

# Save full feature arrays. Aggregated feature arrays are always saved.
save_feature_arrays=true

# Create unique folder for the results of this run.
results_dir="instance_results/$(date +'%b%d_%H%M')"
mkdir -p "$results_dir"

# Define the log file and wipe it
log_file="${results_dir}/features_evaluation.log"
> "$log_file"

# Print experimental setup.
echo "Run folder: $results_dir" | tee -a "$log_file"
echo -e "Experimental setup:\n$desc_msg\n" | tee -a "$log_file"

# Get hostname
pc1="megatron"
pc2="richards-air.staff.sydney.edu.au"
pc3="Richards-MacBook-Air.local"
pc4="RUTHERFORD"
host="$(hostname)"
echo "Host is: $host" | tee -a "$log_file"

# Path info.
if [[ "$host" == *"$pc1"* ]]; then # megatrons
  PYTHON_SCRIPT="/home/kj66/Documents/Richard/venv/bin/python3"
  SCRIPT_PATH="/home/kj66/Documents/Richard/rrut_thesis_code/"
  num_cores=30 # will revert to lower values for higher dims.
elif [[ "$host" == *"$pc4"* ]]; then
    PYTHON_SCRIPT="D:/richa/anaconda3/envs/thesis_env_windows/python.exe"
    SCRIPT_PATH="d:/richa/Documents/Thesis/rrut_thesis_code/"
    num_cores=6
elif [[ "$host" == *"$pc2"* ]] || [[ "$host" == *"$pc3"* ]]; then
  # This checks if $host matches $pc2 or $pc3
  PYTHON_SCRIPT="$HOME/anaconda3/envs/thesis_env_py3.8/bin/python"
  SCRIPT_PATH="/Users/richardrutherford/Documents/Thesis Code/rrut_thesis_code/"
  num_cores=3 # Specify the number of cores for pc2 or pc3
else # richard's pc
  PYTHON_SCRIPT="C:/Users/richa/anaconda3/envs/thesis_env_windows/python.exe"
  SCRIPT_PATH="C:/Users/richa/Documents/Thesis/rrut_thesis_code/"
  num_cores=10 # 10 cores keeps i7-14700KF at 70-80% avg. utilisation, max 22 GB memory usage
fi
echo "Using interpreter: $PYTHON_SCRIPT" | tee -a "$log_file"

# Create unique temporary directory
temp_dir=$(mktemp -d -t ci-XXXXXXXXXX --tmpdir="$SCRIPT_PATH")
echo "New directory: $temp_dir" | tee -a "$log_file"

# Create sibling temp_pops directory and ensure it's cleaned up properly
temp_pops_dir="../temp_pops/" # Create temp_pops as a sibling directory.
mkdir -p "$temp_pops_dir"
echo "Created temp_pops directory: $temp_pops_dir" | tee -a "$log_file"

clean_temp_pops_dir() {
    local dir_to_clean="$1"

    # Check if cygpath exists and use it to normalize the directory path on Cygwin
    if command -v cygpath &> /dev/null; then
        dir_to_clean=$(cygpath -w "$dir_to_clean" | tr '\\' '/')
    fi

    # Find and print top-level directories not containing 'XA', not starting with 'CS' or 'CT' (excluding 'CTP')
    echo "Directories to be removed from $dir_to_clean:"
    local dirs_to_remove=$(find "$dir_to_clean" -mindepth 1 -maxdepth 1 -type d ! -name '*XA*' ! -name '*RWMOP*' ! -name '*CF*' ! -name 'CS*' ! -name 'CT*' -o -name 'CTP*' | tr '\\' '/')
    if [ -z "$dirs_to_remove" ]; then
        echo "None"
    else
        echo "$dirs_to_remove"
        # Remove directories based on the specified criteria
        echo "$dirs_to_remove" | xargs -I {} rm -rf "{}"
        echo "Specified directories removed."
    fi

    # Find and print top-level directories that are being kept
    echo "Directories being kept in $dir_to_clean:"
    local dirs_kept=$(find "$dir_to_clean" -mindepth 1 -maxdepth 1 -type d | tr '\\' '/')
    if [ -z "$dirs_kept" ]; then
        echo "None"
    else
        echo "$dirs_kept"
    fi
}


# Copy framework to temporary directory
echo "Writing ISA-CMOP_Python to temporary directory." | tee -a "$log_file"
copy_dir="$SCRIPT_PATH"
cd_dir="$SCRIPT_PATH"
cd_dir+="ISA-CMOP_Python/"
copy_dir+="ISA-CMOP_Python/*"
cp -R $copy_dir "$temp_dir"

# Handle CTRL+C event clean up
trap ctrl_c INT
function ctrl_c() {
    echo "Terminating program..." | tee -a "$log_file"

    # Clean up temp_pops dir, excluding directories with 'XA' in the name
    clean_temp_pops_dir "$temp_pops_dir"

    # Clean up temp dir
    rm -rf "$temp_dir"
    echo "Removed temporary directory: $temp_dir" | tee -a "$log_file"

    exit 0
}


# Set cwd if not already
cd "$cd_dir" || { echo "cd $cd_dir failed" | tee -a "$log_file"; }

# Path to execute python script
run_dir="$temp_dir"
run_dir+="/runner.py"    # Main script to execute (runner.py)
pre_sampler_script="PreSampler.py"  # PreSampler script

echo "File running inside: $run_dir" | tee -a "$log_file"

# Check if regenerate_samples is true
if [ "$regenerate_samples" = true ]; then
  # Run PreSampler.py for each dimension and number of samples
  for dim in "${dimensions[@]}"; do
    echo "Running PreSampler.py for dimension: $dim" | tee -a "$log_file"
    "$PYTHON_SCRIPT" -u "$pre_sampler_script" "$dim" "$num_samples" "$mode" 2>&1 | tee -a "$log_file"
  done
fi

# Define the JSON file path
config_file="problems_to_run.json"

# Depending on the mode, extract the appropriate section from the JSON file for problem definitions
jq_filter=".$mode | to_entries|map(\"\(.key) \(.value)\")|.[]"

# Read and execute based on the selected section of the JSON file
jq -r "$jq_filter" $config_file | while read line; do
    problem_dim=$(echo $line | cut -d ' ' -f 1)
    should_run=$(echo $line | cut -d ' ' -f 2)
    
    # Extract problem name and dimension
    problem=$(echo $problem_dim | sed -E 's/_d[0-9]+//')
    dim=$(echo $problem_dim | grep -oE '_d[0-9]+' | sed 's/_d//')

    # Check if dimension is empty and handle accordingly
    if [ -z "$dim" ]; then
        echo "No dimension specified for $problem."
    fi
    
    if [[ "$should_run" == "true" ]]; then
        echo -e "\nRunning problem: $problem, dimension: ${dim:-[none]}" | tee -a "$log_file"
        
        # Your existing code to run the problem, handling the case where dim might be empty
        "$PYTHON_SCRIPT" -u "$run_dir" "$problem" "${dim:-}" "$num_samples" "$mode" "$save_feature_arrays" "$results_dir" "$temp_pops_dir" "$num_cores" 2>&1 | tee -a "$log_file"

        # Clean up, etc.
        echo "Cleaning temp_pops directory for next run (other than XA)..." | tee -a "$log_file"
        clean_temp_pops_dir "$temp_pops_dir"
        echo "temp_pops directory cleaned." | tee -a "$log_file"
        # echo "Currently not cleaning directories."


        # Update the JSON to mark this problem-dimension as false, indicating it's been run, only if mode is "eval"
        # if [[ "$mode" == "eval" ]]; then
        #     jq ".[\"$mode\"][\"$problem_dim\"] = \"false\"" $config_file > temp.json && mv temp.json $config_file
        #     echo "Updated $problem_dim in JSON file to false."
        # fi

    else
        echo "Skipping problem: $problem, dimension: ${dim:-[none]} as per config."
    fi
done



# At the end of the program, clean up temp_pops_dir except XA directories
# clean_temp_pops_dir "$temp_pops_dir"
echo "Final cleanup of temp_pops directory, excluding XA folders." | tee -a "$log_file"
# echo "Currently not cleaning directories."

# Clean up temp dir
rm -rf "$temp_dir"
echo "Removed temporary directory: $temp_dir" | tee -a "$log_file"

exit 0

# Exit
exit 0
