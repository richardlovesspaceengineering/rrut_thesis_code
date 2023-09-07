from features.feature_helpers import corr_coef


def f_corr(objvar):
    """
    Significant correlation between objective values.

    Verel2013 provides a detailed derivation, but it seems as if we assume the objective correlation is the same metween all objectives. That is, for a correlation matrix C, C_np = rho for n != p, C_np = 1 for n = p. We want the value of rho.

    Since correlations are assumed to be equal across all objectives, we can just compute one pairwise correlation coefficient. Alsouly finds the same but computes the symmetric 2x2 correlation coefficient and pvalue matrices before extracting the upper-right value.
    """
    corr_obj = corr_coef(objvar[:, 0], objvar[:, 1])

    return corr_obj
