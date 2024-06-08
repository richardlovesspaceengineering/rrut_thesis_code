import numpy as np
import pandas as pd
import pickle

from optimisation.output.util import extract_data
from optimisation.output import plot


def process(test_case):

    """
    Method to extract optimisation results from serialised optimisation history
    :param test_case: String corresponding to optimisation problem history file
    """

    # Load history
    with open('../../results/optimisation_history_test_problem_' + test_case + '.pkl', 'rb') as f:
        history = pickle.load(f)
        f.close()

    # Number of generations
    n_gen = len(history)

    # Pre-allocate pop_data
    n_var = len(history[0][0].var)
    n_obj = len(history[0][0].obj)
    if history[0][0].cons:
        # One additional column for cons_sum
        n_cons = len(history[0][0].cons) + 1
    else:
        # One column for constraint = None and one for cons_sum
        n_cons = 2
    pop_data = np.zeros((0, n_var + n_obj + n_cons + 3))

    # Extract data
    names, var_dict = [], []
    for gen_idx in range(n_gen):
        gen_data, names, var_dict = extract_data(history[gen_idx], gen_idx)
        pop_data = np.concatenate((pop_data, gen_data), axis=0)

    # Forming dataframe
    df = pd.DataFrame(data=pop_data, columns=names)
    df['generation'] = df['generation'].astype(np.int)
    df['rank'] = df['rank'].astype(np.int)

    plot_pareto = True
    if plot_pareto:
        plot.pareto_front(df, test_case, colour_by_generation=True)


if __name__ == '__main__':

    test_case = 'zdt1'
    # test_case = 'zdt2'
    # test_case = 'binh_korn'
    # test_case = 'ctp1'
    # test_case = 'chankong_haimes'
    # test_case = 'CRE22'
    # test_case = 'CRE22'
    # test_case = 'CRE23'
    # test_case = 'CRE31'

    process(test_case)

