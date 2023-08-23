import numpy as np

from optimisation.surrogate.adaptive_sampling import AdaptiveSampling

from optimisation.model.evaluator import Evaluator
from optimisation.model.population import Population
from optimisation.model.individual import Individual


class SurrogateStrategy(object):
    def __init__(self, problem, obj_surrogates=None, cons_surrogates=None,
                 n_training_pts=1, n_infill=5, max_real_f_evals=1000,
                 opt_npop=100, opt_ngen=25,
                 plot=False, print=False, sampling_strategy='idw', use_adaptive_sampling=True,
                 **kwargs):
        
        # Store Infill points and iteration count
        self.infill_x = np.zeros((0, problem.n_var))
        self.opt_count = np.zeros((0))

        # Real objective & constraint functions
        self.real_func = problem.obj_func
        self.real_obj_func = problem.obj_func_specific
        self.real_cons_func = problem.cons_func_specific

        # Function evaluations
        self.real_f_evals = 0
        self.max_real_f_evals = max_real_f_evals

        # Surrogate input checks
        if obj_surrogates is None:
            obj_surrogates = []
        elif len(obj_surrogates) != problem.n_obj:
            raise Exception('Number of objective surrogates must match the number of problem objectives')
        if cons_surrogates is None:
            cons_surrogates = []
        elif len(cons_surrogates) != problem.n_con:
            raise Exception('Number of constraint surrogates must match the number of problem constraints')

        # Surrogate models
        self.obj_surrogates = obj_surrogates
        self.cons_surrogates = cons_surrogates
        self.surrogates = self.obj_surrogates + self.cons_surrogates
        self.sampling_strategy = sampling_strategy

        # Population (training data)
        self.n_training_pts = n_training_pts
        self.population = Population(problem, self.n_training_pts)

        # Evaluator
        self.evaluator = Evaluator()

        # Adaptive sampling strategy
        self.adaptive_sampling = AdaptiveSampling(n_population=opt_npop, n_gen=opt_ngen,
                                                  acquisition_criteria=self.sampling_strategy,
                                                  **kwargs)

        self.n_refinement_iter = 1
        self.n_infill = n_infill

        self.plot = plot
        self.print = print
        self.use_adaptive_sampling = use_adaptive_sampling
        self.ctr = 0

    def initialise(self, problem, sampling):

        # Surrogate sampling (training data)
        sampling.do(self.n_training_pts, problem.x_lower, problem.x_upper)
        surrogate_sampling_x = np.copy(sampling.x)

        # Assign sampled design variables to surrogate population
        self.population.assign_var(problem, surrogate_sampling_x)

        # Evaluate surrogate population (training data)
        self.population = self.evaluator.do(self.real_func, problem, self.population)

        # Run model refinement
        if len(self.cons_surrogates) == 0:
            training_data = (self.population.extract_var(), self.population.extract_obj())
        elif len(self.cons_surrogates) == 1:
            # TODO - add the extraction of the cons_sum or cons_viol surrogate here
            pass
        else:
            training_data = (self.population.extract_var(), np.hstack((self.population.extract_obj(), self.population.extract_cons())))
        self.run(problem, training_data=training_data)

    def run(self, problem, training_data=None):

        if training_data is not None:

            # Add training data & re-train each model
            for i, model in enumerate(self.surrogates):
                model.add_points(training_data[0], training_data[1][:, i])
                model.train()

            # Update number of real function evaluations
            self.real_f_evals += self.n_training_pts

        # Calculate p-norm of cv-RMSE across all models
        cv_rmse = np.array([model.cv_rmse for model in self.surrogates])
        if self.print:
            print('cv_rmse:', cv_rmse)

        # Run model refinement
        if self.use_adaptive_sampling:
            # if np.linalg.norm(cv_rmse) > 0.01:
            self.model_refinement(problem)
            self.ctr += 1

        # Plot surrogate and function
        if self.plot and problem.n_var == 2:
            for i, model in enumerate(self.surrogates):
                plot_surrogate(model, self.real_func, idx=i, ctr=self.ctr)

    def model_refinement(self, problem):

        for i in range(self.n_refinement_iter):

            # Calculate adaptation parameter
            alpha = -0.5*np.cos((self.real_f_evals/self.max_real_f_evals)*np.pi) + 0.5

            # Determine infill points (x locations)
            # NOTE: This currently only uses objective function surrogates - will need an optimiser capable of handling
            # much higher dimensions if we want to incorporate constraint surrogates into the infill strategy
            eval_x = self.adaptive_sampling.generate_evals(models=self.obj_surrogates,
                                                           n_pts=self.n_infill,
                                                           alpha=alpha,
                                                           parent_prob=problem,
                                                           # cons_models=self.cons_surrogates,
                                                           # use_constraints=True,
                                                           n_processors=problem.n_processors)
            # Store infill points
            self.infill_x = np.vstack((self.infill_x, eval_x))

            # Evaluate infill points
            eval_z_obj = np.zeros(0)
            eval_z_cons = np.zeros(0)
            for j in range(len(eval_x)):

                # Form xdict
                _individual = Individual(problem)
                _individual.set_var(problem, eval_x[j, :])

                # Call real objective function to evaluate infill points
                temp_z_obj, temp_z_cons, _ = self.real_func(_individual.var_dict)

                # Concatenate z values
                if j == 0:
                    eval_z_obj = np.atleast_2d(temp_z_obj)
                    eval_z_cons = np.atleast_2d(temp_z_cons)
                else:
                    eval_z_obj = np.vstack((eval_z_obj, temp_z_obj))
                    eval_z_cons = np.vstack((eval_z_cons, temp_z_cons))

            # Add infill points to each model
            for j, model in enumerate(self.obj_surrogates):
                model.add_points(eval_x, eval_z_obj[:, j])
                model.train()
                model.update_cv()
            for j, model in enumerate(self.cons_surrogates):
                model.add_points(eval_x, eval_z_cons[:, j])
                model.train()
                model.update_cv()

            # Update number of real function evaluations
            self.real_f_evals += self.n_infill
            
            # Store number of infill points
            self.opt_count = np.hstack((self.opt_count, self.real_f_evals))

    def obj_func(self, x_dict):

        # Form design vector from x_dict
        x = x_dict['x_vars']

        # Todo: If not all objective functions need to be surrogate modelled, then update here

        # Calculating objective function values using surrogate models
        if len(self.obj_surrogates) > 0:
            obj = np.zeros(len(self.obj_surrogates))
            for i, model in enumerate(self.obj_surrogates):
                obj[i] = model.predict(x)
        else:
            obj = self.real_obj_func(x)

        # Calculate constraint function values using surrogate models
        if len(self.cons_surrogates) > 0:
            cons = np.zeros(len(self.cons_surrogates))
            for i, model in enumerate(self.cons_surrogates):
                cons[i] = model.predict(x)
        else:
            cons = self.real_cons_func(x)

        return obj, cons, None


from collections import OrderedDict
import matplotlib
import matplotlib.pyplot as plt
# Plot settings
matplotlib.rc('savefig', dpi=300, format='pdf', bbox='tight')


def plot_surrogate(model, real_func, idx=0, ctr=0):

    fig = plt.figure(idx)
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Fine grid data
    x_0_fine = np.linspace(model.l_b[0], model.u_b[0], 50)
    x_1_fine = np.linspace(model.l_b[1], model.u_b[1], 50)
    _x_0_fine, _x_1_fine = np.meshgrid(x_0_fine, x_1_fine)
    x_fine_mesh = np.array((_x_0_fine, _x_1_fine)).T
    z_fine_mesh = np.zeros(np.shape(x_fine_mesh[:, :, 0]))
    z_sm_predicted = np.zeros(np.shape(x_fine_mesh[:, :, 0]))
    var_dict = OrderedDict()
    for i in range(np.shape(x_fine_mesh)[0]):
        for j in range(np.shape(x_fine_mesh)[1]):
            var_dict['x_vars'] = x_fine_mesh[i, j, :]
            temp, _, _ = real_func(var_dict)
            z_fine_mesh[i, j] = temp[idx]
            z_sm_predicted[i, j] = model.predict(x_fine_mesh[i, j, :])

    h_training_vals = ax.scatter(model.x[:, 0], model.x[:, 1], model.y, color='red', marker='o', label='Training data')
    h_func_vals = ax.plot_wireframe(x_fine_mesh[:, :, 0], x_fine_mesh[:, :, 1], z_fine_mesh, alpha=0.7, color='black',
                                    linewidth=0.25, label='Real function values')
    h_surrogate_vals = ax.plot_surface(x_fine_mesh[:, :, 0], x_fine_mesh[:, :, 1], z_sm_predicted, alpha=0.6,
                                       cmap='viridis', label='Surrogate predicted values')

    cmap = matplotlib.cm.get_cmap('viridis')
    surface_proxy = matplotlib.patches.Patch(color=cmap(0.0), alpha=0.5, label='Surrogate predicted values')
    ax.legend(handles=[h_training_vals, h_func_vals, surface_proxy], loc='upper right')
    plt.savefig('./figures/surrogate_modelling_testing/model_' + str(idx) + '_gen_' + str(ctr) + '.pdf')

    plt.show()
    plt.close()

