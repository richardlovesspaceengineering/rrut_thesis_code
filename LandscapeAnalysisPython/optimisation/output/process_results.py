import numpy as np
import pandas as pd
import copy
import glob
import os
import math

from collections import OrderedDict
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib import rcParams

from matplotlib import font_manager
font_dirs = ['/Users/3s/Library/Fonts' ]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Gulliver-Regular'


mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=14)
mpl.rc('ytick', labelsize=14)
mpl.rc('legend', fontsize=11)


myblack = [0, 0, 0]
myblue = '#0F95D7'
myred = [220 / 255, 50 / 255, 32/ 255]
myyellow = [255 / 255, 194 / 255, 10 / 255]
# mygreen = [64/255, 176 / 255, 166/255]
# mygreen = [65/255, 141 / 255, 43/255]
mygreen = [139/255, 195 / 255, 74/255]
mybrown = [153/255,79/255,0/255]
# mydarkblue = [60/255, 105/255, 225/255]
mydarkblue = '#0069C9'
# mypurple = [0.4940, 0.1840, 0.5560]
mypurple = '#8F169B'
# myorange = [230/255, 97/255, 0/255]
myorange = '#FF8B00'
mygray = [89 / 255, 89 / 255, 89 / 255]
myviolet = '#A800FF'


colors = [ myblue, myred, myyellow,
             'forestgreen', myblack,  myviolet, myorange, mypurple, mydarkblue, mygray]

fill_colors = [ 'turquoise', 'firebrick', 'goldenrod',
               'forestgreen',  'grey',  'magenta', 'orange', 'purple',  'royalblue', 'grey']


def process(problem_name,dimensionality, use_log, sub_folders, nr_values=360,
            remove_worst=False, remove_best= False, plot_variation=True ):
    # create file path
    filepath = '../../results/' + str(dimensionality) + 'D/'


    # sub_folders = ['linear_tail','switch_constant']


    # Initialise Figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
    plt.subplots_adjust(wspace=0.35, hspace=0.3)
    fig.supxlabel('Real Function Evaluations', fontsize=14)
    fig.supylabel('Mean Objective Value', fontsize=14)

    # get all the files from each subfolder

    for cntr in range(len(sub_folders)):
        full_path = filepath + sub_folders[cntr] + '/'+ problem_name + '*.csv'
        filelist = glob.glob(full_path)

        for file_counter in range(len(filelist)):
            filename = filelist[file_counter]
            data = np.empty(nr_values)
            data[:] = np.NaN
            data_temp = np.genfromtxt(filename, delimiter=',')

            for it_counter in range(len(data_temp)):
                if it_counter < nr_values:
                    data[it_counter] = data_temp[it_counter][-1]
            data = np.nan_to_num(data,nan=min(data))

            data = np.minimum.accumulate(data, axis=0)
            # n_evals = np.arange(start=1, stop=len(data) + 1, step=1)
            # ax.plot(n_evals, data, '-', color=colors[cntr + 1], linewidth=2)


            # remove the first element

            if file_counter == 0:
                obj_vals = data
            else:
                obj_vals = np.vstack((obj_vals, data))

            # extract best value
            ind = np.nanargmin(data)
            if file_counter == 0:
                opt_obj = data[ind]
            else:
                opt_obj = np.vstack((opt_obj,data[ind]))

        # obj_vals = np.maximum.accumulate(obj_vals, axis=0)
        # obj_vals = np.minimum.accumulate(obj_vals, axis=0)
        # Extract min/ max/ mean
        obj_min = np.nanmin(obj_vals, axis=0)
        obj_min = np.minimum.accumulate(obj_min, axis=0)

        # find the worst run and remove it
        # TODO needs to be debugged
        if remove_best:
            ind = np.where(obj_vals[:, -1] == obj_min[-1])[0][0]
            obj_vals = np.delete(obj_vals, ind, 0)
            obj_min = np.nanmin(obj_vals, axis=0)
            obj_min = np.minimum.accumulate(obj_min, axis=0)

        obj_max = np.nanmax(obj_vals, axis=0)
        obj_max = np.minimum.accumulate(obj_max, axis=0)

        if remove_worst:
            ind = np.where(obj_vals[:, -1] == obj_max[-1])[0][0]
            obj_vals = np.delete(obj_vals, ind, 0)
            obj_max = np.nanmax(obj_vals, axis=0)
            obj_max = np.minimum.accumulate(obj_min, axis=0)

        obj_mean = np.nanmean(obj_vals, axis=0)
        obj_mean = np.minimum.accumulate(obj_mean, axis=0)
        obj_std = np.nanstd(obj_vals, axis=0)
        # obj_std = np.maximum.accumulate(obj_std, axis=1)

        # obj_min = obj_mean - obj_std
        # obj_max = obj_mean + obj_std

        n_evals = np.arange(start=1, stop=len(obj_mean) + 1, step=1)

        ax.plot(n_evals, obj_mean, '-', color=colors[cntr], linewidth=2, label=sub_folders[cntr])
        if plot_variation:
            ax.fill_between(n_evals, obj_max, obj_min, facecolor=fill_colors[cntr], alpha=0.2)
            ax.plot(n_evals, obj_max, '--', color=colors[cntr], linewidth=0.75)
            ax.plot(n_evals, obj_min, '--', color=colors[cntr], linewidth=0.75)

        # ax.set_title("Stock Williams Aerofoil")
    ax.set_xlim((1, max(n_evals)))
    # ax.set_ylim((np.nanmin(obj_min), np.nanmax(obj_max)))
        # ax.plot([10, 10], [f_opt[pos], np.max(obj_mean)], '--', color='k', linewidth=1)
        # ax.set_yscale('log')

        # FINAL FORMATTING PLOT
        # ax.plot(n_evals, f_opt[pos] * np.ones(len(n_evals)), color='k', linewidth=2, label='Optimum')
        # Line, Label = ax.get_legend_handles_labels()
        # lgd = plt.legend(Line, Label, loc='upper right', ncol=7, bbox_to_anchor=(0.85, 1.17), frameon=False)
        # for line in lgd.get_lines():
        #     line.set_linewidth(3.0)
    # ax.plot([20, 20], [0, 375], '--', color='k', linewidth=1)
    # y_min = math.floor(0.9 * np.nanmin(obj_min) / 10) * 10
    # ax.set_ylim((y_min, math.ceil(1.1 * np.nanmax(obj_max) / 10) * 10))
    # ax.set_xlim((1, 355))

    Line, Label = ax.get_legend_handles_labels()
    # lgd = plt.legend(Line, Label, loc='lower center', ncol=7, bbox_to_anchor=(0.5, 1.1), frameon=False)
    # lgd = plt.legend(Line, Label, loc='lower center', ncol=7, frameon=False)
    lgd = plt.legend(Line, Label, ncol=1, frameon=False)
    for line in lgd.get_lines():
        line.set_linewidth(3.0)
    if use_log:
        plt.yscale('log')
        # ax.set_ylim(10**-3,10**4)
    # ax.set_ylim(-300,800)
    plt.grid()
    plt.grid(b=True, which='major')
    plt.grid(b=True, which='minor')
    plt.minorticks_on()
    plt.title(problem_name)

    plt.show()




if __name__ == '__main__':

    # test_case = 'airfoil_testing_xfoil'

    problem_name = 'Ackley'
    # problem_name = 'CEC05_F10'
    # problem_name = 'Ellipsoid'
    problem_name = 'Griewank'
    # problem_name = 'Rastrigin'
    # problem_name = 'Rosenbrock'

    dimensionality = 30
    total_calls = 360

    use_log = True
    # use_log = False

    sub_folders = ['3_30gen','3_2expl_10pct','3_1expl_10pct','3_5expl_10pct','3_2expl_5pct',
                   '3_2expl_late_varrange','3_2expl_late_varrange_LHS']
    sub_folders = ['3_30gen','3_2expl_late_varrange','3_5expl_late_varrange_LHS',
                   '3_5expl_late_varrange_LHS2','3_5expl_late_varrange_LHS3',
                   '2_5expl_late_varrange_LHS3']

    sub_folders = ['3_30gen','3_5expl_late_varrange_LHS3','4_5expl_late_varrange_LHS3',
                   '2_5expl_late_varrange_LHS3','3_5expl_late_varrange_LHS4','3_10expl_late_varrange_LHS3',
                   '3_2expl_late_varrange_LHS3','3_20expl_late_varrange_LHS3']

    sub_folders = ['3_30gen', '3_2expl_late_varrange_LHS3','3_1expl_late_varrange_LHS3',
                   '2_1expl_late_varrange_LHS3','4_1expl_late_varrange_LHS3',
                   '2_2expl_late_varrange_LHS3','4_2expl_late_varrange_LHS3']

    sub_folders = ['3k_2it_start25','3k_2it_start0','3k_2it_start15','3k_2it_start35','3k_2it_start50']

    sub_folders = ['3k_2it_start25', '3k_2it_start25_25pct','3k_2it_start25_375pct','3k_2it_start25_75pct']

    sub_folders = ['3k_2it_start25', '3k_2it_DoE_50_025pct','3k_2it_DoE_50_05pct',
                   '3k_2it_DoE_25_025pct','3k_2it_DoE_25_05pct','3k_2it_DoE_375_025pct']

    sub_folders = ['3k_2it_start25', '3k_2it_lcb','3k_2it_ei','3k_2it_ei_lcb']

    sub_folders = ['3k_2it_start25', 'pick_best_1k','pick_best_2k', 'pick_best_3k',
                   'pick_best_2k_all', 'pick_best_3k_all']

    sub_folders = ['3k_2it_start25', '3k_best', '3k_best_explore','3k_best_explore_LHS']

    sub_folders = ['3k_2it_start25', '3k_2it_4n_cutoff10','3k_2it_4n_cutoff25',
                   '3k_2it_3n_cutoff10','3k_2it_3n_cutoff25']

    sub_folders = ['3k_2it_start25', '3k_2it_3n_cutoff50','3k_2it_4n_cutoff50',
                   '3k_2it_5n_cutoff10', '3k_2it_5n_cutoff25', '3k_2it_5n_cutoff50',
                   '3k_2it_4n_cutoff10', '3k_2it_4n_cutoff25', '3k_2it_3n_cutoff25']

    sub_folders = ['3k_2it_start25', '3k_2it_3n_cutoff50',
                   '3k_2it_3n_cutoff25','3k_2it_3n_cutoff10','3k_2it_3n_cutoff0']

    # sub_folders = ['3k_2it_start25', '3k_2it_4n_cutoff50',
    #                '3k_2it_4n_cutoff25','3k_2it_4n_cutoff10','3k_2it_4n_cutoff0']
    #
    # sub_folders = ['3k_2it_start25', '3k_2it_5n_cutoff50',
    #                '3k_2it_5n_cutoff25','3k_2it_5n_cutoff10','3k_2it_5n_cutoff0']


    #
    sub_folders = ['3k_2it_start25', '3k_2it_3n_cutoff10',
                   '3k_2it_4n_cutoff10','3k_2it_5n_cutoff10']
    #
    sub_folders = ['3k_2it_start25', '3k_2it_3n_cutoff50',
                   '3k_2it_3n_cutoff25','3k_2it_3n_cutoff10']

    sub_folders = ['3k_2it_start25',
                   '3k_2it_infill05','3k_2it_infill10','3k_2it_infill15',
                   '3k_2it_infill10_red_range','3k_2it_infill10_range01','3k_2it_infill10_range025']

    sub_folders = ['3k_2it_start25',
                   '3k_2it_infill10',
                   '3k_2it_infill10_red_range',
                   '3k_2it_infill10_range1','3k_2it_infill10_range2']
    remove_worst = False
    remove_best = remove_worst
    plot_variation = True

    process(problem_name,dimensionality,use_log,sub_folders,nr_values=total_calls,
            remove_worst=remove_worst, remove_best=remove_best,
            plot_variation=plot_variation)

