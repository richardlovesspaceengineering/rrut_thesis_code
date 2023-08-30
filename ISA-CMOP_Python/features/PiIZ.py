from feature_helpers import remove_imag_rows, corr_coef
from optimisation.util.non_dominated_sorting import NonDominatedSorting


def PiIZ(pop):
    """
    Compute proportion of solutions in the ideal zone (the lower left quadrant of the fitness-violation scatterplot) for each objective and for unconstrained fronts-violation scatterplot.
    """

    # Extracting matrices.
    objvar = pop.extract_obj()
    decvar = pop.extract_var()
    consvar = pop.extract_cons()

    # Get CV, a row vector containing the norm of the constraint violations. Assuming this can be standardised for any given problem setup.
    consvar[consvar <= 0] = 0
    cv = np.sum(consvar, axis=1)

    # Remove imaginary rows. Deep copies are created here.
    objvar = remove_imag_rows(objvar)

    # Defining the ideal zone.
    minobjs = min(objvar)

    return [piz_ob, piz_f]
