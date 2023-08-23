import numpy as np
import matplotlib.pyplot as plt
# import os

def benchmark_plot_function(self):
    pop = self.population
    pop_obj_arr = pop.extract_obj()

    hist_vals = np.zeros((len(self.history[-1][:]), 2))
    for i in range(len(self.history[-1][:])):
        hist_vals[i, :] = self.history[-1][i].var

    # Optimum Point
    x_opt = self.opt.var
    f_opt = self.opt.obj

    # Infilled Points
    infill = self.surrogate.infill_x
    gen_colors = np.linspace(0.2, 1, int(len(infill[:, 0]) / self.surrogate.n_infill))

    # Sampled Points
    x_train = self.surrogate.surrogate_sampling

    # os.environ["PATH"] += os.pathsep + '/usr/local/textlive/2015'
    # plt.rcParams['text.usetex'] = True

    fig = plt.figure()  # figsize=(7,7))
    ax = plt.subplot(111)

    # True function values
    n_exact = 300
    x1_exact = np.linspace(self.problem.x_lower[0], self.problem.x_upper[0], n_exact)
    x2_exact = np.linspace(self.problem.x_lower[1], self.problem.x_upper[1], n_exact)
    x11, x22 = np.meshgrid(x1_exact, x2_exact)
    x_mesh = np.array((x11, x22)).T
    zz = np.zeros(np.shape(x_mesh[:, :, 0]))
    gg = np.zeros(np.shape(x_mesh[:, :, 0]))

    from collections import OrderedDict
    var_dict = OrderedDict()
    for i in range(len(x11[:, 0])):
        for j in range(len(x22[:, 0])):
            var_dict["x_vars"] = x_mesh[i, j, :]
            zz[i, j], gg[i, j], _ = self.problem.obj_func(var_dict)

    idy, idx = np.unravel_index(zz.argmin(), zz.shape)
    true_opt = x_mesh[idy, idx]
    # print(true_opt)
    # print(zz.min())

    ax_surro = ax.contourf(x11, x22, zz.T, 30, label="Function Contour")
    if gg.any() is not None:
        ax_cons = ax.contour(x11, x22, gg, [0], colors='green', label="Constraint Boundaries")
    ax_sample = ax.scatter(x_train[:, 0], x_train[:, 1], color='r', marker='s', label='Training Points')
    ax_infill = ax.scatter(infill[:, 0], infill[:, 1], color='orange', alpha=gen_colors, edgecolors='orange',
                           label='Infill Points')
    ax_opt = ax.scatter(x_opt[0], x_opt[1], s=110, color='fuchsia', marker='*', label='Optimum Point')
    ax_trueopt = ax.scatter(true_opt[0], true_opt[1], s=110, color='forestgreen', marker='*', label='True Optimum')
    # ax_hist = ax.scatter(hist_vals[0], hist_vals[1], color='b', label='Final Population')

    cbar = fig.colorbar(ax_surro)
    ax.set_title(self.surrogate.sampling_strategy.upper())
    ax.set_xlabel(r"$x_{1}$")
    ax.set_ylabel(r"$x_{2}$")
    lgd = ax.legend(loc="upper right", frameon=False, bbox_to_anchor=[0.35, 1.25])  # 1.25
    for lh in lgd.legendHandles:
        lh.set_alpha(1)
    file_path = 'D:/THESIS B/BENCHNMARKING/'
    space = '_'
    file_name = file_path + self.problem.name + space + self.surrogate.sampling_strategy + space + 'max_gen' + space + str(
        self.max_gen) + space + 'sampling' + space + str(len(x_train))
    plt.show()
    # fig.savefig(file_name, bbox_extra_artists=(lgd,), bbox_inches='tight')

    ### Export Objective history data
    # Save CSV file of LHS
    file_path = 'D:/THESIS B/BENCHNMARKING/RESULTS/'
    file_name = file_path + self.problem.name + space + self.surrogate.sampling_strategy + space + 'max_gen' + space + str(
        self.max_gen) + space + 'sampling' + space + str(len(x_train)) + space + str(
        self.surrogate.sampling_seed) + space + 'objective_history' + '.csv'
    # results = np.hstack((self.surrogate.opt_count[0:-1].reshape((self.surrogate.opt_count[0:-1].size, 1)), self.opt_hist))
    # np.savetxt(file_name, results, delimiter=",")
