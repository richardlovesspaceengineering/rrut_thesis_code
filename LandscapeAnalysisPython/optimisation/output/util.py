import numpy as np


def extract_data(pop, gen_id):

    """
    Method to extract variables, constraints, objectives and population data from optimisation history
    :param pop: Population instance
    :param gen_id: Generation id
    :return:
    """

    # Extract design variable data
    pop_var, var_names, var_dict = extract_var(pop)

    # Extract objective function values
    pop_obj, obj_names = extract_obj(pop)

    # Extract constraint values
    pop_cons, cons_names = extract_cons(pop)

    # Extract rank & crowding distance
    pop_rank = pop.extract_rank()
    pop_crowding = pop.extract_crowding()

    # Generation number
    pop_gen = gen_id*np.ones(len(pop_rank), dtype=np.int)

    # Concatenating arrays
    data = np.concatenate(
        (pop_var, pop_obj, pop_cons, pop_rank[:, np.newaxis], pop_crowding[:, np.newaxis], pop_gen[:, np.newaxis]),
        axis=1)
    names = var_names + obj_names + cons_names + ['rank', 'crowding_distance', 'generation']

    # Output
    return data, names, var_dict


def extract_var(pop):

    """
    Method to extract variables from population
    :param pop:
    :return:
    """

    # Extract var_dict & variable names
    var_dict = pop[0].var_dict
    var_names = []
    for key in pop[0].var_dict.keys():
        for idx in range(len(pop[0].var_dict[key])):
            var_names.append(key + '_' + str(idx))

    # Extract variable values
    var_array = pop.extract_var()

    return var_array, var_names, var_dict


def extract_obj(pop):

    """
    Method to extract objectives from population
    :param pop:
    :return:
    """

    # Objective names
    obj_names = ['f_' + str(idx) for idx in range(len(pop[0].obj))]

    # Extract objective values
    obj_array = pop.extract_obj()

    return obj_array, obj_names


def extract_cons(pop):

    """
    Method to extract constraints from population
    :param pop:
    :return:
    """

    # Constraint names
    if pop[0].cons:
        cons_names = ['constraint_' + str(idx) for idx in range(len(pop[0].cons))]
    else:
        cons_names = ['constraint']
    cons_names.append('cons_sum')

    # Extract constraint values
    cons_array = pop.extract_cons()
    cons_sum = pop.extract_cons_sum()
    cons_array = np.concatenate((cons_array, cons_sum[:, np.newaxis]), axis=1)

    return cons_array, cons_names

