import numpy as np
import multiprocess


class Evaluator(object):

    def __init__(self):

        self.n_eval = 0

    def do(self, obj_fun, problem, pop, max_abs_con_vals=None, **kwargs):

        # Scale variable dicts for each individual in pop
        for i in range(len(pop)):
            pop[i].descale_var(problem)

        # obj_fun = problem.obj_fun
        pop_obj = pop
        idx_arr = np.array_split(np.arange(np.shape(pop_obj)[0]), problem.comm.size)

        if problem.map_internally:

            # Open pool
            problem.pool = multiprocess.Pool(problem.n_processors)

            # Evaluate objective function
            out = self.eval_async(problem, obj_fun, pop_obj, idx_arr[0])

            # Clean up pool
            problem.pool.close()
            problem.pool.join()

        else:
            # Broadcast the type of call (0 means regular call)
            problem.comm.bcast(0, root=0)

            # Broadcast obj_func
            obj_fun = problem.comm.bcast(obj_fun, root=0)

            # Broadcast pop object
            pop_obj = problem.comm.bcast(pop_obj, root=0)

            # Scatter index array to workers on comm
            idx_arr = problem.comm.scatter(idx_arr, root=0)

            # Evaluate objective function (all processes on comm)
            out = self.eval(obj_fun, pop_obj, idx_arr)

            # Gather output from all process on comm
            out = problem.comm.gather(out, root=0)

        # Process output
        data = [(0, 0, 0)]*len(pop)
        for output in out:
            for obj_data in output:
                data[obj_data[0]] = (obj_data[1], obj_data[2], obj_data[3])

        # Assign output to individuals in population
        for idx in range(len(pop)):
            pop[idx].obj = data[idx][0]
            pop[idx].cons = data[idx][1]
            pop[idx].performance = data[idx][2]

            # Updating evaluation counter
            self.n_eval += 1

        # Scaling constraint values
        if problem.n_con > 0:
            pop = self.sum_normalised_cons(pop, problem, max_abs_con_vals=max_abs_con_vals)

        return pop

    @staticmethod
    def sum_normalised_cons(pop, problem, max_abs_con_vals=None):

        if max_abs_con_vals is None:
            max_abs_con_vals = Evaluator.calc_max_abs_cons(pop, problem)

        # Scaling constraint values & summing them for each individual
        for idx in range(len(pop)):
            cons_temp = pop[idx].cons
            cons_temp[cons_temp <= 0.0] = 0.0
            pop[idx].cons_sum = np.sum(cons_temp/max_abs_con_vals)

        return pop

    @staticmethod
    def calc_max_abs_cons(pop, problem):

        # Extracting constraint values from pop
        cons_array = np.zeros((len(pop), problem.n_con))
        for idx in range(len(pop)):
            if pop[idx].cons is not None:
                cons_array[idx, :] = pop[idx].cons

        # Calculating maximum value of each constraint across population
        max_abs_con_vals = np.amax(np.abs(cons_array), axis=0)

        # Avoid dividing by zero
        max_abs_con_vals[max_abs_con_vals == 0.0] = 1.0

        return max_abs_con_vals

    @staticmethod
    def eval_async(problem, obj_fun, pop, idx_arr):

        # Extract var_dict from pop & form iterable
        pop_var_dict = []
        for idx in range(len(pop)):
            pop_var_dict.append(pop[idx].var_dict)

        # Map population inputs across objective function
        # result = problem.pool.map_async(obj_fun, pop_var_dict)
        result = problem.pool.starmap_async(obj_fun, zip(pop_var_dict, [i for i in range(len(pop))]))
        temp = result.get()

        # Extract results
        out = [(0, 0, 0, 0)] * np.shape(idx_arr)[0]
        for i, idx in enumerate(idx_arr):
            out[i] = [(idx, temp[i][0], temp[i][1], temp[i][2])]

        return out

    @staticmethod
    def eval(obj_fun, pop, idx_arr, **kwargs):

        # Evaluate objective function
        out = [(0, 0, 0, 0)] * np.shape(idx_arr)[0]
        for i, idx in enumerate(idx_arr):
            # obj, cons, performance = obj_fun(pop[idx].var_dict, **kwargs)
            obj, cons, performance = obj_fun(pop[idx].var_dict, idx=i, **kwargs)
            out[i] = (idx, obj, cons, performance)

        return out

