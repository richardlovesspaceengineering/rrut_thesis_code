import numpy as np
import string

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm


def pareto_front(df, test_case, colour_by_generation=True, plot_final_generation_only=False):

    """
    Method to plot Pareto front of optimisation history
    :param df: Pandas dataframe of population data
    :param test_case: Case identifying string
    :param colour_by_generation:
    :param plot_final_generation_only:
    :return:
    """

    # Plotting only feasible solutions
    df = df[df['cons_sum'] <= 0.0]

    # Plot settings
    # mpl.rc('lines', markersize=6)
    mpl.rc('axes', labelsize=12)
    mpl.rc('xtick', labelsize=12)
    mpl.rc('ytick', labelsize=12)
    mpl.rc('legend', fontsize=11)
    mpl.rc('figure', figsize=[6.4, 4.8])
    mpl.rc('savefig', dpi=200, format='pdf', bbox='tight')
    labels = list(string.ascii_uppercase)

    # Generations
    generations = df['generation'].unique()

    # Plot pareto front
    if plot_final_generation_only:
        fig, ax = plt.subplots()

        gen_mask = np.in1d(df['generation'], generations[-1])
        gen_data = df[gen_mask]

        ax.scatter(gen_data['f_0'], gen_data['f_1'], facecolor='C0', alpha=0.5, label='Final population')
        ax.set_xlabel('f_0')
        ax.set_ylabel('f_1')
        ax.grid(True)
        ax.legend()
    else:
        if colour_by_generation:
            # Colourmap
            n_gen = generations[-1]
            cmap = cm.get_cmap('viridis', n_gen)

            gen_mask = np.in1d(df['generation'], n_gen)
            non_dominated_mask = (df['rank'] == 0.0)
            mask = gen_mask & non_dominated_mask
            dominated_data = df[~mask]

            # Redoing things so that flipping the final generation dataframe generates the correct results
            gen_mask = np.in1d(df['generation'], n_gen)
            gen_data = df[gen_mask]
            gen_data = gen_data.iloc[::-1]
            non_dominated_mask = (gen_data['rank'] == 0.0)
            non_dominated_data = gen_data[non_dominated_mask]

            fig, ax = plt.subplots()
            plt.scatter(dominated_data['f_0'].values, dominated_data['f_1'].values, c=dominated_data['generation'],
                        cmap=cmap, alpha=0.3, vmin=0)
            cbar = plt.colorbar(label='Generation', ticks=np.arange(0, n_gen+10, 10, dtype=np.int).tolist())
            cbar.ax.set_yticklabels(np.arange(0, n_gen+10, 10, dtype=np.int).tolist())
            cbar.set_alpha(1.0)
            cbar.draw_all()
            plt.scatter(non_dominated_data['f_0'].values, non_dominated_data['f_1'].values, c='k', alpha=0.6,
                        label='Non-dominated solutions')

            ax.set_xlabel('f_0')
            ax.set_ylabel('f_1')
            ax.grid(True)

        else:
            fig, ax = plt.subplots()

            for gen_idx in generations:
                gen_mask = np.in1d(df['generation'], gen_idx)
                gen_data = df[gen_mask]

                # Flipping dataframe so the worst ranked individuals are plotted first
                gen_data = gen_data.iloc[::-1]

                if gen_idx == generations[-1]:
                    non_dominated_mask = (gen_data['rank'] == 0.0)
                    non_dominated_data = gen_data[non_dominated_mask]
                    dominated_data = gen_data[~non_dominated_mask]
                    ax.scatter(dominated_data['f_0'], dominated_data['f_1'], facecolor='C0', alpha=0.3,
                               label='Dominated solutions')
                    ax.scatter(non_dominated_data['f_0'], non_dominated_data['f_1'], facecolor='C1', alpha=0.6,
                               label='Non-dominated solutions')
                else:
                    ax.scatter(gen_data['f_0'], gen_data['f_1'], facecolor='C0', alpha=0.3)

            ax.set_xlabel('f_0')
            ax.set_ylabel('f_1')
            ax.grid(True)

    # Saving plot
    plt.savefig('../../figures/' + test_case + '_pareto_front' + '.pdf')

    plt.show()
    plt.close()

