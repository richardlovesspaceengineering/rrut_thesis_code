# Richard's Landscape Analysis Code Guide
Hopefully this guide is detailed enough to get everything running without me! I have tested everything is working on megatron2, but in case you need to move this code over to a new machine, the steps for a new installation have also been included.

In this guide, I will try and cover how the code works, as well as how to add new problems/suites and features.

- [Richard's Landscape Analysis Code Guide](#richards-landscape-analysis-code-guide)
  - [Basic Workflow](#basic-workflow)
    - [Features Evaluation](#features-evaluation)
      - [Setting up update emails](#setting-up-update-emails)
    - [Post-Processing](#post-processing)
  - [Adding new problems](#adding-new-problems)
    - [Feature Computation Modifications:](#feature-computation-modifications)
    - [Post-Processing Modifications](#post-processing-modifications)
  - [Adding new features](#adding-new-features)
  - [Fresh Installation](#fresh-installation)
    - [Git repositories to clone](#git-repositories-to-clone)
    - [Python virtual environment](#python-virtual-environment)
    - [Final file structure](#final-file-structure)
  - [Questions](#questions)

## Basic Workflow

The workflow consists of two major steps: features evaluation and post-processing. The overall idea is to begin by setting what problems we want to evaluate features for, run a shell script to evaluate features for these problems, then access the stored results for post-processing.

### Features Evaluation
This section details how to execute the landscape features evaluation code. These instructions assume that everything will be run from megatron2, or any other machine running Linux where everything (``git`` synchronisation, XFOIL/Python wrapper) has been set up already.

Key files/directories within ``\LandscapeAnalysisPython``:
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
2. Run the shell-script using ``.\run_problems.sh`` in a terminal. Take note of what the datetime code is e.g. ``Jan1_1200`` as you will access results from a subdirectory with this name later.
   1. On a fresh install of this repository, you might need to make this script executable with a ``chmod +x run_problems.sh`` command. The only setting you should change within this file is ``mode``. If ``mode = "debug"``, the features evaluation code runs on much smaller sample sizes, while ``mode = "eval"`` runs the full evaluation.
   2. A note on sibling directories: the feature evaluation saves pregenerated random walks/global samples in a sibling directory called ``pregen_samples``, located in ``rrut_thesis_code\pregen_samples``, and it also saves evaluated populations for problems that take a while to run (PLATEMO, XA, MODAct) in a sibling directory called ``temp_pops``. These directories can contain very large files, so if you're running out of storage, feel free to start deleting specific problem files from ``temp_pops``.
   3. If you're having issues with memory usage (e.g. the machine starts swapping), you can reduce the number of cores used for evaluation in the ``initialize_number_of_cores`` method in ``ProblemEvaluator``. Due to my time constraints, I instead just reduced the number of samples for the high-dimensional ($\geq 20$) PLATEMO, MODAct and XA instances.
   4. Note that a run of a problem will set its value to ``"false"`` in ``problems_to_run.json`` - this behaviour occurs even if the Python script returned an error. Always double check ``problems_to_run.json`` before a new run.
3. Once the run is complete, check the results have been saved into ``instance_results\<datetimecode>``.

#### Setting up update emails
You can modify the following details, located in the ``__init__`` method in ``ProblemEvaluator``, for email updates about the progress of the landscapes run.

```python
# Gmail account credentials for email updates.
self.gmail_user = "username@gmail.com"  # Your full Gmail address
self.gmail_app_password = "password "  # Your generated App Password (set up using OAuth in gmail)
```

### Post-Processing
This can be run from any machine since this just accesses csv files with common Python libraries e.g. ``scikit-learn, pandas, numpy``. To reduce having to use git so much, I would recommend running this notebook from the same machine e.g. megatron2. 

Steps to run:
1. Open the Jupyter notebook ``PostProcessingNotebook.ipynb``. In the second cell, after the package imports, there is a dictionary called ``results_dict`` which specifies the directories where the results for each problem are located.
2. Update ``results_dict`` with the appropriate ``<datetime>`` code from earlier to ensure your new results are imported for analysis:
```python
results_dict = {
    # Put appropriate datetime code here.
    "MW1_d5": ["Feb12_1444"],
    "LIRCMOP_d10": ["Feb12_1444"],
    # and more...
}
```
3. Rerun each cell as required. Much more detailed instructions are provided within the notebook.

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
3. Run the next cell, where the ``FeaturesDashboard`` class is initialised. This essentially concatenates all results into a single ``pandas`` dataframe, and all data post-processing is done on this dataframe. There is a huge amount of methods in ``FeaturesDashboard``, so if you want to navigate around the methods without scrolling, I'd recommend using the "Outline" tab in the sidebar in VSCode (or whatever equivalent in your IDE of choice). Note: if you were planning on releasing this code, I would make this class more modular.
4. Rerun the notebook cells as necessary. There are clear instructions in the notebook about the purpose of each cell, and I've done my best to include comments to show at what point certain figures/tables in the report are generated with code. I would definitely have my report open while you navigate this notebook at first.

## Adding new features
Add a new method to ``GlobalAnalysis`` or ``RandomWalkAnalysis``, depending on the sampling type. It should become obvious what to do upon inspection of these classes.

## Fresh Installation

This covers everything Richard had to set up to get things working on megatron2. It does not cover XFOIL or MATLAB for Linux installation. You will only need to consult this if you wish to work on machines other than megatron2.

The installation involves setting up our file structure (comprised of 4 git repositories) and Python environments.

We will put everything in its own self-contained folder. So create a directory in which we can store everything (we will assume it is called ``Landscapes`` for now).

### Git repositories to clone

The landscape analysis code accesses multiple repositories. To access these, clone the following repositories into ``Landscapes``:
* ``rrut_thesis_code``: landscape analysis code.
* ``rrut_thesis_report``: thesis report code.
* ``AirfoilBenchmarkSuite``: code defining the XA suite and XFOIL wrapper.
* ``MODAct``: code defining the MODAct suite.

### Python virtual environment

We use the same virtual environment (venv) for features evaluation and post-processing. These installation steps will install all packages other than the MATLAB Engine for Python. The install instructions for this specific package can be found here: https://au.mathworks.com/help/matlab/matlab_external/python-setup-script-to-install-matlab-engine-api.html

1. In the same directory as the git repositories (``Landscapes``), create a new environment using the following terminal command: ``python3.8.10 -m venv venv``. 
2. Then activate with ``source venv/bin/activate``. 
3. Install the required packages with ``pip install -r rrut_thesis_code/venv_requirements.txt``.


<!-- We use separate environments for features evaluation and post-processing. These have both been set up on 

Both of these environments should be created in the same directory as the git repositories.

**Feature Evaluation environment**: 

**Post-Processing environment**: create a new environment using the following command: ``python3.8.10 -m venv postprocessingenv``. Then activate with ``source postprocessingenv/bin/activate`` and install the required packages with ``pip install -r rrut_thesis_code/postprocessingenv.txt``. -->

### Final file structure

Your final directory should have the following subdirectories (with these exact names):
* ``rrut_thesis_code``
* ``rrut_thesis_report``
* ``AirfoilBenchmarkSuite``
* ``MODAct``
* ``venv``

All code is run from ``rrut_thesis_code``.


## Questions
Email me at richardrutherford@outlook.com if anything is catastrophically broken. I am in Sydney and available to help with any code issues from July 7 - July 23 2024.