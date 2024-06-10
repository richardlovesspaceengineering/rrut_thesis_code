# Richard's Landscape Analysis Code
Hopefully this README is detailed enough to get everything running without me! 

- [Richard's Landscape Analysis Code](#richards-landscape-analysis-code)
  - [Installation](#installation)
    - [Git repositories to clone](#git-repositories-to-clone)
  - [Basic Workflow](#basic-workflow)
    - [Features Evaluation](#features-evaluation)
      - [Setting up update emails](#setting-up-update-emails)
    - [Post-Processing](#post-processing)
  - [Adding new problems](#adding-new-problems)
    - [Feature Computation Modifications:](#feature-computation-modifications)
    - [Post-Processing Modifications](#post-processing-modifications)
  - [Adding new features](#adding-new-features)
  - [Questions](#questions)


## Installation

### Git repositories to clone

Git setup

Required sibling directories: MODAct (PFs folder), AirfoilBenchmarkSuite, rrut_thesis_report

Conda environment: for landscape analysis and post-processing.

## Basic Workflow

The workflow consists of two major steps: features evaluation and post-processing. The overall idea is to begin by setting what problems we want to evaluate features for, run a shell script to evaluate features for these problems, then access the stored results for post-processing.

### Features Evaluation
This section details how to execute the landscape features evaluation code. These instructions assume that everything will be run from megatron2, or any other machine running Linux where everything (``git`` synchronisation, XFOIL/Python wrapper) has been set up already.

Key files/directories:
* ``problems_to_run.json`` is where we configure what problems we want to run.
* ``run_problems.sh`` shell script that allows the Python code to be run in a one-liner from the terminal.
* ``\instance_results``: directory where feature evaluation + normalisation results are stored as csv files. The results are stored in subdirectories named according to the start time e.g. ``"Jan01_1200"``

Steps to run:
1. Set whatever problems you'd like to run as ``"true"`` in ``problems_to_run.json``.
```json
{
  "debug": {
    // These problems will run in debug mode.
    "MW1_d5": "true", // this will run
    "LIRCMOP_d10": "false", // this will not run
    // and more...
  },
  "eval": {
    // These problems will run in eval mode.
    "MW1_d5": "true", // this will run
    "LIRCMOP_d10": "false", // this will not run
    // and more...
  }
}
```
2. Run the shell-script using ``.\run_problems.sh`` in a terminal. 
   1. On a fresh install of this repository, you might need to make this script executable with a ``chmod +x run_problems.sh`` command. The only setting you should change in this is ``mode``. If ``mode = "debug"``, the features eval code runs on much smaller sample sizes, while ``mode = "eval"`` runs the full evaluation.
   2. A note on sibling directories: the feature evaluation saves pregenerated random walks/global samples in a sibling directory called ``pregen_samples``, and it also saves evaluated populations for problems that take a while to run (PLATEMO, XA, MODAct) in a sibling directory called ``temp_pops``. These directories can contain very large files, so if you're running out of storage, feel free to start deleting specific problem files from ``temp_pops``.
   3. If you're having issues with memory usage (e.g. the machine starts swapping), you can reduce the number of cores used for evaluation in the ``initialize_number_of_cores`` method in ``ProblemEvaluator``.
3. Note that a run of a problem will set its value to ``"false"`` in ``problems_to_run.json`` - this behaviour occurs even if the Python script returned an error. Always double check ``problems_to_run.json`` before a run.

#### Setting up update emails
You can modify the following details, located in the ``__init__`` method in ``ProblemEvaluator``, for email updates about the progress of the landscapes run.

```python
# Gmail account credentials for email updates.
self.gmail_user = "username@gmail.com"  # Your full Gmail address
self.gmail_app_password = "password "  # Your generated App Password (set up using OAuth in gmail)
```


Mention inclusion of debug mode.

### Post-Processing
This can be run from any machine since this just accesses csv files with common Python libraries e.g. ``scikit-learn, pandas, numpy``.

Run from a *different* Python venv to the landscapes one (this is necessary as post-processing uses some features only available to newer Python versions). 

## Adding new problems

You may want to run the features evaluation for new problems/suites. To do this you need to modify two sections of the feature computation code as well as two sections of the post-processing code:

### Feature Computation Modifications:
1. We should run the feature calculation in debug mode first to ensure everything is working properly. Start by adding the new problem names to the ``debug`` and ``eval`` fields in ``problems_to_run.json``, like so
```json
{
  "debug": {
    // Existing problems.
    "MW1_d5": "false",
    "LIRCMOP_d10": "false",
    // and more...

    // New problem.
    "XA3_d10": "true"
  },
  "eval": {
    // Existing problems.
    "MW1_d5": "false",
    "LIRCMOP_d10": "false",
    // and more...

    // New problem.
    "XA3_d10": "true"
  }
}
```
2. Update the wrapper, called ``generate_instance`` inside ``runner.py``. The existing examples should make clear how to do this.
2. Run the feature calculation in ``debug`` mode. Once there are no errors, you can now run in ``eval`` mode.

Note: if the new problem is only going to be run with global features, you may need to replicate how the XA problems were treated in ``ProblemEvaluator.py``.

### Post-Processing Modifications
1. Add the new problem and its corresponding results directory name to ``results_dict`` in ``PostProcessingNotebook.ipynb``:
```python
results_dict = {
    # Existing problems.
    "MW1_d5": ["Feb12_1444"],
    "LIRCMOP_d10": ["Feb12_1444"],
    # and more...

    # New problem.
    "XA3_d10": ["Jun06_2020"]
}
```
2. Modify the ``define_problem_suites`` method to include the new problem. The existing examples should make clear how to do this. Note that if the problem is from a new suite, you will also need to add an associated plotting colour for this suite to the ``"Paired"`` palette in ``apply_custom_colours``. 
2. Rerun the notebook as necessary.

## Adding new features
Add a new method to ``GlobalAnalysis`` or ``RandomWalkAnalysis``, depending on the sampling type.

## Questions
Email me at richardrutherford@outlook.com if anything is catastrophically broken. I am in Sydney and available to help with any code issues from July 7 - July 23 2024.