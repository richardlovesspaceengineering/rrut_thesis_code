# Richard's Landscape Analysis Code
Hopefully this README is detailed enough to get everything running without me! 

- [Richard's Landscape Analysis Code](#richards-landscape-analysis-code)
  - [Installation](#installation)
    - [Git repositories to clone](#git-repositories-to-clone)
  - [Basic Workflow](#basic-workflow)
    - [Features Evaluation](#features-evaluation)
    - [Post-Processing](#post-processing)
  - [Adding new problems](#adding-new-problems)
    - [Feature Computation Modifications:](#feature-computation-modifications)
    - [Post-Processing Modifications](#post-processing-modifications)
  - [Adding new features](#adding-new-features)
  - [Questions](#questions)


## Installation

### Git repositories to clone

Git setup

Conda environment: for landscape analysis and post-processing.

## Basic Workflow

The workflow consists of two major steps: features evaluation and post-processing. The overall idea is to begin by setting what problems we want to evaluate features for, run a shell script to evaluate features for these problems, then access the stored results for post-processing.

### Features Evaluation
These instructions assume that everything will be run from megatron2, or any other machine running Linux where everything (``git`` synchronisation, XFOIL/Python wrapper) has been set up already.

This section details how to execute the landscape features evaluation code.

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