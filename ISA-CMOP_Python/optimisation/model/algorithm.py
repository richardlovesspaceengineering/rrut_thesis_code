import numpy as np
import pickle
import copy

import os
from optimisation.model.evaluator import Evaluator


class Algorithm:

    def __init__(self, **kwargs):

        """
        This class is the abstract parent class of all algorithm classes.
        The class implements a number of class methods, namely:
        setup: Method which initialises a number of the important class member variables (usually called from inside
        optimise.minimise)
        do: Method which specifies whether the solve or wait methods are called depending on parallel distribution
        solve: Method which implements a wrapper around specific algorithm _initialise and _next methods
        wait: Method which implements the parallel distribution of information to workers and back again
        each_iteration: Method called at the end of every generation to update the algorithm and serialise history
        """

        # Termination criterion
        self.termination = kwargs.get("termination")

        # Optimisation problem
        self.problem = None

        # Evaluator
        self.evaluator = None

        # Initialisation parameters
        self.hot_start = False
        self.hot_start_file = None
        self.x_init = False
        self.x_init_additional = False
        self.seed = None

        # Generation parameters
        self.n_gen = 0
        if 'max_gen' in kwargs:
            max_gen = kwargs['max_gen']
            del kwargs['max_gen']
        else:
            max_gen = 200
        self.max_gen = max_gen
        if 'max_f_eval' in kwargs:
            max_f_eval = kwargs['max_f_eval']
            del kwargs['max_f_eval']
        else:
            max_f_eval = None
        self.max_f_eval = max_f_eval

        # Population parameters
        self.n_population = None
        self.population = None

        # Surrogate
        self.surrogate = None

        # Optimum position
        self.opt = None

        # Termination parameters
        self.finished = False

        # History
        self.save_history = True
        self.history = []

        # Plotting
        if 'plot' in kwargs:
            _plot = kwargs['plot']
            del kwargs['plot']
        else:
            _plot = True
        self.plot = _plot

        # Printing
        if 'print' in kwargs:
            _print = kwargs['print']
            del kwargs['print']
        else:
            _print = True
        self.print = _print

    def setup(self,
              problem,
              seed=None,
              hot_start=False,
              x_init=False,
              x_init_additional=False,
              save_history=True,
              evaluator=None):

        # Problem
        self.problem = problem

        # Evaluation
        if evaluator is None:
            evaluator = Evaluator()
        self.evaluator = evaluator

        # Random number seed
        self.seed = seed
        if self.seed is None:
            self.seed = np.random.randint(0, 10000000)
        np.random.seed(self.seed)

        # Initialisation parameters
        self.hot_start = hot_start
        self.x_init = x_init
        self.x_init_additional = x_init_additional

        # Save history
        self.save_history = save_history

    def do(self):

        if self.problem.comm.rank == 0:
            self.solve()
        else:
            self.wait()

    def solve(self):

        # Initialise the initial population and evaluate it
        self._initialise()
        self.each_iteration()

        # While termination criterion is not fulfilled
        while not self.finished:
            # Complete the next iteration
            self._next()
            self.each_iteration()

        # Finalise & post-process
        self.finalise()

        # Broadcast -1 to indicate solution has finished
        self.problem.comm.bcast(-1, root=0)

    def wait(self):

        mode = None
        obj_fun = None
        pop_obj = None
        idx_arr = None
        while True:
            # Receive mode and quit if mode is -1:
            mode = self.problem.comm.bcast(mode, root=0)
            if mode == -1:
                break
            else:  # Otherwise receive info from root function
                # Broadcast obj_func
                obj_fun = self.problem.comm.bcast(obj_fun, root=0)

                # Broadcast pop object
                pop_obj = self.problem.comm.bcast(pop_obj, root=0)

                # Scatter index array to workers on comm
                idx_arr = self.problem.comm.scatter(idx_arr, root=0)

                # Call the evaluator function
                out = self.evaluator.eval(obj_fun, pop_obj, idx_arr)

                # Gather output from all process on comm
                out = self.problem.comm.gather(out, root=0)

    def each_iteration(self):

        # Display algorithm parameters
        if self.print:
            if self.n_gen == 0:
                print('====================')
                print('n_gen\t|\t n_eval')
                print('====================')
            else:
                print('%d\t\t|\t %d' % (self.n_gen, self.evaluator.n_eval))

        if self.save_history:

            # Append current population to history
            self.history.append(copy.deepcopy(self.population))

            # Set write format
            if self.n_gen == 0:
                write_format = 'wb'
            else:
                write_format = 'ab'

            # Write history
            self.write_history(write_format)

        # Update number of generations
        self.n_gen += 1

        # Check termination criterion
        self.check_termination()

    def check_termination(self):
        if self.n_gen > self.max_gen:
            self.finished = True

        if self.evaluator.n_eval > self.max_f_eval:
            self.finished = True

    def finalise(self):

        if self.plot and self.problem.n_obj > 1:
            self.plot_results()
        if self.problem.n_obj == 1:
            self.print_results()

        if self.save_history:
            # Write history
            self.write_history(write_format='ab')

    def hot_start_initialisation(self):

        # Load population history for hot-start
        directory = os.path.dirname('./results/')
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open('./results/optimisation_history_' + self.problem.name + '.pkl', 'rb') as f:
            data = pickle.load(f)
            f.close()

        # Assigning history from hot-start history
        self.history = copy.deepcopy(data[:-1])

        # Assigning population from latest generation of hot-start history
        self.population = copy.deepcopy(data[-1])

        # Scaling variables for population
        for i in range(len(self.population)):
            self.population[i].scale_var(self.problem)

        # Set number of generations from hot-start history
        self.n_gen = len(self.history)

        # Set number of evaluations from hot-start history
        self.evaluator.n_eval = len(self.history) * self.n_population

    def _initialise(self):
        pass

    def _next(self):
        pass

    def write_history(self, write_format):

        # Serialise population history

        directory = os.path.dirname('./results/')
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open('./results/optimisation_history_' + self.problem.name + '.pkl', 'wb') as f:
            pickle.dump(self.history, f, pickle.HIGHEST_PROTOCOL)

        # Extract variable, objective & constraint arrays from population
        var_array = self.population.extract_var()
        obj_array = self.population.extract_obj()
        cons_array = self.population.extract_cons()

        # Save variable history
        with open('./results/var_history_' + self.problem.name + '.txt', write_format) as f:
            temp = np.hstack((var_array, self.n_gen * np.ones((self.n_population, 1))))
            np.savetxt(f, temp, fmt=['%.16f'] * self.problem.n_var + ['%d'])

        # Save objective history
        with open('./results/obj_history_' + self.problem.name + '.txt', write_format) as f:
            temp = np.hstack((obj_array, self.n_gen * np.ones((self.n_population, 1))))
            np.savetxt(f, temp, fmt=['%.16f'] * self.problem.n_obj + ['%d'])

        # Save constraint history
        if self.problem.n_con > 0:
            with open('./results/cons_history_' + self.problem.name + '.txt', write_format) as f:
                temp = np.hstack((cons_array, self.n_gen * np.ones((self.n_population, 1))))
                np.savetxt(f, temp, fmt=['%.16f'] * self.problem.n_con + ['%d'])

    def plot_results(self):

        pareto_front = None
        if self.problem.name[:17] == 'test_problem_' + 'zdt2':
            x = np.linspace(0, 1, 100)
            pareto_front = np.array([x, 1.0 - x ** 2.0]).T
        elif self.problem.name[:17] == 'test_problem_' + 'zdt1':
            x = np.linspace(0, 1, 100)
            pareto_front = np.array([x, 1.0 - x ** 0.5]).T

        ## Juan add ons
        if 'ZDT' in self.problem.name:
            pareto_front = self.problem.variables['x_vars'][0].f_opt
        elif 'DTLZ' in self.problem.name:
            pareto_front = self.problem.variables['x_vars'][0].f_opt
        elif 'MW' in self.problem.name:
            pareto_front = self.problem.variables['x_vars'][0].f_opt
        elif 'WFG' in self.problem.name:
            pareto_front = self.problem.variables['x_vars'][0].f_opt
        elif 'DASCMOP' in self.problem.name:
            pareto_front = self.problem.variables['x_vars'][0].f_opt
        elif 'LSMOP' in self.problem.name:
            pareto_front = self.problem.variables['x_vars'][0].f_opt
        elif 'LIRCMOP' in self.problem.name:
            pareto_front = self.problem.variables['x_vars'][0].f_opt
        elif 'LYO' in self.problem.name:
            pareto_front = self.problem.variables['x_vars'][0].f_opt

        pop = self.population
        pop_obj_arr = pop.extract_obj()

        import matplotlib
        import matplotlib.pyplot as plt

        if self.problem.n_obj == 2:
            fig, ax = plt.subplots()
            ax.scatter(pop_obj_arr[:, 0], pop_obj_arr[:, 1], facecolor='C0', alpha=0.5, label='Final population')
            if pareto_front is not None:
                ax.scatter(pareto_front[:, 0], pareto_front[:, 1], color='C1', alpha=0.8, label='Pareto front')
            ax.set_xlabel('f_0')
            ax.set_ylabel('f_1')
            ax.grid(True)
            ax.legend()
        elif self.problem.n_obj == 3:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.scatter3D(pop_obj_arr[:, 0], pop_obj_arr[:, 1], pop_obj_arr[:, 2], facecolor='C0', alpha=0.5, label='Final population')
            if pareto_front is not None:
                ax.scatter3D(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], alpha=0.8, color='C1', label='Pareto front')
            ax.set_xlabel('f_0')
            ax.set_ylabel('f_1')
            ax.set_zlabel('f_2')
            ax.grid(True)
            ax.legend()
            # ax.set_xlim((0, 1))
            # ax.set_ylim((0, 1))
            # ax.set_zlim((0, 1))

        matplotlib.rc('savefig', dpi=300, format='pdf', bbox='tight')
        directory = os.path.dirname('./figures/')
        if not os.path.exists(directory):
            os.makedirs(directory)

        if self.surrogate is not None:
            plt.savefig('./figures/pareto_front_' + self.problem.name + '_surrogate'
                        + '_n_pop_' + str(self.n_population) + '_max_gen_' + str(self.max_gen) + '.pdf')
        else:
            plt.savefig('./figures/pareto_front_' + self.problem.name + '_n_pop_' + str(self.n_population)
                        + '_max_gen_' + str(self.max_gen) + '.pdf')

        plt.show()
        plt.close()


    def print_results(self):

        if self.opt is not None:
            print('Optimum position:', self.opt.var)
            print('Optimum objective value:', self.opt.obj)
            if self.problem.n_con > 0:
                print('Optimum constraint value:', self.opt.cons)
