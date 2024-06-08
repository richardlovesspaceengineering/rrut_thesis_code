import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rcParams
from optimisation.surrogate.models.rbf import RadialBasisFunctions, RBF
from optimisation.surrogate.models.ensemble import EnsembleSurrogate

# figure size in inches
rcParams['figure.figsize'] = 6.4, 4.8
plt.style.use('seaborn-talk')
# import os

def benchmark_plot_function(self):
    surro_pop = self.surrogate.population
    obj_opt = surro_pop.extract_obj()

    # Filter out infeasible points
    if False:
        cons_viol = surro_pop.extract_cons()
        cons_pop = surro_pop.extract_cons_sum()
        cons_mask = cons_pop <= 0
        opt_ind = np.argmin(obj_opt[cons_mask])
        surro_opt_var = surro_pop.extract_var()[cons_mask][opt_ind]
        surro_opt_val = obj_opt[cons_mask][opt_ind]

        print(f"Min pos: {surro_opt_var}")
        print(f"Min val: {surro_opt_val}")

    # hist_vals = np.zeros((len(self.history[-1][:]), 2))
    # for i in range(len(self.history[-1][:])):
    #     hist_vals[i, :] = self.history[-1][i].var

    # Optimum Point
    x_opt = self.opt.var
    f_opt = self.opt.obj

    # Sampled Points
    x_train = self.surrogate.surrogate_sampling

    # Infilled Points
    infill = surro_pop[len(x_train):].extract_var()
    print(len(infill))

    # 2D Plot if desired
    if len(infill[0,:]) == 2:
        plot_2d(self, infill, x_opt, f_opt, x_train, surro_pop, surro_opt_var, surro_opt_val)

    # extract results from surrogate population
    objectives = self.surrogate.population.extract_obj()
    variables = self.surrogate.population.extract_var()
    ## SAVING FILES
    # file_path = 'D:/THESIS B/BENCHNMARKING/'
    space = '_'
    #file_name = file_path + self.problem.name + space + self.surrogate.sampling_strategy + space + self.surrogate.constraint_strategy + space + 'max_gen' + space + str(
    #    self.max_gen) + space + 'sampling' + space + str(len(x_train))
    # fig.savefig(file_name, bbox_extra_artists=(lgd,), bbox_inches='tight')

    ### Export Objective history data
    # Save CSV file of LHS
    # file_path = 'D:/THESIS B/BENCHNMARKING/2D/RESULTS/'
    # file_path = 'D:/THESIS B/BENCHNMARKING/2D/BILOG_RESULTS/'
    # file_path = 'D:/THESIS B/BENCHNMARKING/5D/BILOG_RESULTS/'
    # file_path = 'D:/THESIS B/BENCHNMARKING/5D/ALPHA_RESULTS/'
    # file_path = 'D:/THESIS B/BENCHNMARKING/5D/DOE_RESULTS/'
    # file_path = 'D:/THESIS B/BENCHNMARKING/10D/RESULTS/'
    # file_path = 'D:/THESIS B/BENCHNMARKING/10D/COMPARISON_RESULTS/'
    # file_path = 'D:/THESIS B/BENCHNMARKING/CONSTRAINED/REAL_PROBLEMS/'
    # file_path = 'D:/THESIS B/BENCHNMARKING/CONSTRAINED/10D/SCALABLE/'
    # file_path = 'C:/Users/jaras/Desktop/THESIS/venv/optimisation_framework/results/'
    # file_path = './venv/optimisation_framework/results/'
    file_path = './results/'
    file_name = file_path + self.problem.name + space + self.surrogate.sampling_strategy + space + self.surrogate.constraint_strategy + space + 'max_gen' + space + str(
        round(self.max_gen/8)) + space + 'sampling' + space + str(len(x_train)) + space + '_seed_' + space + str(
        self.surrogate.sampling_seed) + space + 'objective_history' + '.csv'  # round(self.max_gen/8) --> 125
    # counter = np.atleast_2d(np.arange(len(self.opt_hist)))
    counter = np.atleast_2d(np.arange(len(objectives)))
    counter +=1
    results = np.hstack((counter.T, objectives))
    np.savetxt(file_name, results, delimiter=",")

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    import matplotlib.colors as colors
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def plot_2d(self, infill, x_opt, f_opt, x_train, surro_pop, surro_opt_var, surro_opt_val):
    # surrogate = RadialBasisFunctions(n_dim=len(infill[0,:]), l_b=0, u_b=1, c=2.0, p_type='linear', kernel_type='gaussian_basis')
    surrogate = EnsembleSurrogate(n_dim=2, l_b=self.problem.x_lower, u_b=self.problem.x_upper)
    # surrogate.add_points(surro_pop.extract_var(), surro_pop.extract_obj().flatten())
    if self.problem.n_con > 1:
        g_values = surro_pop.extract_cons()
        x_values = surro_pop.extract_var()
        ks_values = _predict_ks(x_values, g_values)
        surrogate.add_points(x_values, ks_values)
    else:
        surrogate.add_points(surro_pop.extract_var(), surro_pop.extract_cons().flatten())
    surrogate.train()
    gen_colors = np.linspace(0.25, 1, int(len(infill[:, 0]) / self.surrogate.n_infill))

    # CREATE FIGURE
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(25, 5.5))

    # True function values
    n_exact = 100
    x1_exact = np.linspace(self.problem.x_lower[0], self.problem.x_upper[0], n_exact)
    x2_exact = np.linspace(self.problem.x_lower[1], self.problem.x_upper[1], n_exact)
    x11, x22 = np.meshgrid(x1_exact, x2_exact)
    x_mesh = np.array((x11, x22)).T
    zz = np.zeros(np.shape(x_mesh[:, :, 0]))
    gg = np.zeros((n_exact, n_exact, self.problem.n_con)) ## Change for single constraint
    f_surrogate = np.zeros(np.shape(x_mesh[:, :, 0]))

    from collections import OrderedDict
    var_dict = OrderedDict()
    for i in range(len(x11[:, 0])):
        for j in range(len(x22[:, 0])):
            var_dict["x_vars"] = x_mesh[i, j, :]
            zz[i, j], gg[i, j, :], _ = self.problem.obj_func(var_dict) ## Change for single constraint
            f_surrogate[i, j] = surrogate.predict(x_mesh[i, j, :])

    idy, idx = np.unravel_index(zz.argmin(), zz.shape)
    true_opt = x_mesh[idy, idx]
    print(true_opt)
    print(zz.min())

    N = 40
    cmap = cm.get_cmap('Blues_r', N)  # 'Blues_r'  'GnBu_r'  'PuBu_r'
    cmap = truncate_colormap(cmap, 0.0, 0.9)
    ax = fig.axes[0]

    var_opt = self.problem.variables['x_vars'][0].var_opt
    f_opt = self.problem.variables['x_vars'][0].f_opt
    ax_surro = ax.contourf(x11, x22, zz.T, N,  cmap=cmap, label="Function Contour")
    if gg.any() is not None:
        for i in range(self.problem.n_con):
            ax_cons = ax.contour(x11, x22, gg[:,:,i].T, [0], colors='lime', label="Constraint Boundaries") ## Change for single constraint

    ax_sample = ax.scatter(x_train[:, 0], x_train[:, 1], color='r', marker='s', label='Training Points')
    ax_infill = ax.scatter(infill[:, 0], infill[:, 1], color='darkorange', alpha=gen_colors, edgecolors='darkorange',
                           label='Infill Points')
    ax_opt = ax.scatter(surro_opt_var[0], surro_opt_var[1], s=200, color='fuchsia', marker='*', label='Predicted Optimum')
    ax_trueopt = ax.scatter(var_opt[:, 0], var_opt[:, 1], s=150, color='forestgreen', marker='*', label='True Optimum')
    # ax_hist = ax.scatter(hist_vals[0], hist_vals[1], color='b', label='Final Population')

    # cbar = fig.colorbar(ax_surro)
    ax.set_title(self.surrogate.sampling_strategy.upper())
    ax.set_xlabel(r"$x_{1}$")
    ax.set_ylabel(r"$x_{2}$")
    lgd = ax.legend(loc="upper right", frameon=False, bbox_to_anchor=[0.35, 1.25])  # 1.25
    for lh in lgd.legendHandles:
        lh.set_alpha(1)

    # VISUALISE SURROGATES
    ax = fig.axes[1]
    ax_surro = ax.contourf(x11, x22, f_surrogate.T, N, cmap=cmap, label="Surrogate Contour") # cmap=cmap
    ax_cons = ax.contour(x11, x22, f_surrogate.T, [0], colors='lime', label="Constraint Boundaries")
    ax_cons = ax.contourf(x11, x22, f_surrogate.T, np.linspace(np.min(f_surrogate), 0, N), color='lime', label="Constraint Boundaries")
    # im = plt.imshow( f_surrogate <= 0, extent=(x1_exact.min(), x1_exact.max(), x2_exact.min(), x2_exact.max()), origin='lower', cmap='viridis')
    cbar = fig.colorbar(ax_surro)
    ax.set_title("Constraint Surrogate")
    ax.set_xlabel(r"$x_{1}$")

    plt.show()

def _predict_ks(x, y, rho=50):

    # Take the bilog of constraints
    y = bilog_transform(y)

    if np.ndim(x) == 1:
        term = 0
        for n_c in range(len(y)):
            term += np.exp(rho * y[n_c])
        ks_sample = np.array((1 / rho) * np.log(term))
    else:
        ks_sample = np.zeros(len(x))
        for i in range(len(x)):
            term = 0
            for n_c in range(len(y[0, :])):
                term += np.exp(rho * y[i, n_c])
            ks_sample[i] = (1 / rho) * np.log(term)

    return ks_sample

def bilog_transform(obj, beta=1):
    bilog_obj = np.zeros(np.shape(obj))

    for i in range(len(obj)):
        bilog_obj[i] = np.sign(obj[i])*np.log(beta + np.abs(obj[i]))

    return bilog_obj
