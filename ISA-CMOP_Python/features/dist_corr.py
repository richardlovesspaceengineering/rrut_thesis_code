from feature_helpers import remove_imag_rows


def dist_corr(pop, NonDominated):
    """
    Distance correlation.

    Distance for each solution to nearest global solution in decision space. Correlation of distance and constraints norm.

    Fairly certain that NonDominated is an instance of the Population class containing the non-dominated solutions
    """

    objvar = pop.extract_obj()
    decvar = pop.extract_var()
    consvar = pop.extract_cons()

    # Remove imaginary rows.
    objvar = remove_imag_rows(objvar)

    return [dist_c_corr]
