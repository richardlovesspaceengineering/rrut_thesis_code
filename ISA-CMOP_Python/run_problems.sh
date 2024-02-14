#!/bin/bash

# Message describing the experimental setup.
desc_msg="Increased dimensionality, now running full benchmark suites (other than aerofoils)."

# Problem suites
problemsCTP=("CTP1", "CTP2", "CTP3", "CTP4", "CTP5", "CTP6", "CTP7", "CTP8")
problemsMW=("MW1", "MW2", "MW3", "MW4", "MW5", "MW6", "MW7", "MW8", "MW9", "MW10", "MW11", "MW12", "MW13", "MW14")
problemsDASCMOP=("DASCMOP1", "DASCMOP2", "DASCMOP3", "DASCMOP4", "DASCMOP5", "DASCMOP6", "DASCMOP7", "DASCMOP8", "DASCMOP9")
problemsDCDTLZ=("DC1DTLZ1" "DC1DTLZ3" "DC2DTLZ1" "DC2DTLZ3" "DC3DTLZ1" "DC3DTLZ3")
problemsCDTLZ=("C1DTLZ1" "C1DTLZ3" "C2DTLZ2" "C3DTLZ1" "C3DTLZ4")
# problemsRW=("Truss2D", "WeldedBeam") # not scalable in Pymoo
# problemsMODACT=("MODACT") # requires extra package

# Dimensions to consider
dimensions=(5 10 15 20 30)

# Number of samples to run.
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
host="$(hostname)"
echo "Host is: $host" | tee -a "$log_file"

# Path info.
if [[ "$host" == *"$pc1"* ]]; then # megatrons
  PYTHON_SCRIPT="/home/kj66/Documents/Richard/venv/bin/python3"
  SCRIPT_PATH="/home/kj66/Documents/Richard/rrut_thesis_code/"
  num_cores=30 # @JUAN NEED TO SPECIFY.
else # richard's pc
  PYTHON_SCRIPT="C:/Users/richa/anaconda3/envs/thesis_env_windows/python.exe"
  SCRIPT_PATH="C:/Users/richa/Documents/Thesis/rrut_thesis_code/"
  num_cores=10 # 10 cores keeps i7-14700KF at 70-80% avg. utilisation, max 22 GB memory usage
fi
echo "Using interpreter: $PYTHON_SCRIPT" | tee -a "$log_file"

# Create unique temporary directory
temp_dir=$(mktemp -d -t ci-XXXXXXXXXX --tmpdir="$SCRIPT_PATH")
echo "New directory: $temp_dir" | tee -a "$log_file"

# Copy framework to temporary directory
echo "Writing ISA-CMOP_Python to temporary directory. This may take some time if pregenerated samples are being used." | tee -a "$log_file"
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

# Check command line argument for problem set selection
if [ "$1" = "MW" ]; then
    selected_problems=("${problemsMW[@]}")
elif [ "$1" = "CTP" ]; then
    selected_problems=("${problemsCTP[@]}")
elif [ "$1" = "DASCMOP" ]; then
    selected_problems=("${problemsDASCMOP[@]}")
elif [ "$1" = "DCDTLZ" ]; then
    selected_problems=("${problemsDCDTLZ[@]}")
elif [ "$1" = "CDTLZ" ]; then
    selected_problems=("${problemsCDTLZ[@]}")
elif [ "$1" = "RW" ]; then
    selected_problems=("${problemsRW[@]}")
else
    echo "Invalid argument. Please specify 'MW', 'CTP', 'DASCMOP', 'DCDTLZ', 'CDTLZ', or 'RW'."
    exit 1
fi

# Run runner.py for each dimension, then for each problem within that dimension
for dim in "${dimensions[@]}"; do
    for problem in "${selected_problems[@]}"; do
      problem=$(echo "$problem" | sed 's/,$//')  # Remove trailing comma if it exists
      echo -e "\nRunning problem: $problem, dimension: $dim" | tee -a "$log_file"  # Print message to the terminal and log file
      # Run runner.py
      "$PYTHON_SCRIPT" -u "$run_dir" "$problem" "$dim" "$num_samples" "$mode" "$save_feature_arrays" "$results_dir" "$num_cores" 2>&1 | tee -a "$log_file"
    done
done

# Clean up temp dir
rm -rf "$temp_dir"

# Exit
exit 0
