# Landscape Analysis

## Installation

## Basic workflow

Mention inclusion of debug mode.

## Adding new problems

To do this you need to modify two sections of the feature computation code as well as two sections of the post-processing code:

### Feature Computation Code:
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

### Post-Processing
1. Add the new problem and its corresponding results directory name to ``results_dict`` in thesis_notebook_overall.ipynb

## Adding new features

## Post-Processing