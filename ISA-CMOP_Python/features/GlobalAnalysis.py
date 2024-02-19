import numpy as np

from scipy.stats import yeojohnson
from features.Analysis import Analysis, MultipleAnalysis

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew, iqr
from features.feature_helpers import *
from scipy.spatial.distance import pdist, cdist
from scipy.stats import iqr
from optimisation.model.population import Population
from optimisation.util.calculate_hypervolume import calculate_hypervolume_pygmo
from pymoo.indicators.gd import GD
from pymoo.indicators.igd import IGD
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors


class GlobalAnalysis(Analysis):
    """
    Calculate all features generated from a random sample.
    """

    def create_empty_analysis_obj(self):
        return GlobalAnalysis(None, self.normalisation_values, self.results_dir)

    def cv_distr(self, norm_method):
        """
        Distribution of constraints violations.
        """

        # Extract normalisation values.
        var_lb, var_ub, obj_lb, obj_ub, cv_lb, cv_ub = super().extract_norm_values(
            norm_method
        )

        # Remove any rows with imaginary values.
        cv = Analysis.apply_normalisation(self.pop.extract_cv(), cv_lb, cv_ub)

        # Setting axis = 0 ensures computation columnwise. Output is a row vector.
        mean_cv = np.mean(cv)
        std_cv = np.std(cv)
        min_cv = np.min(cv)
        max_cv = np.max(cv)

        # Check if the vector is constant
        if std_cv == 0:
            kurt_cv = skew_cv = 0
            print("CVs are all equal - setting skew_cv = kurtosis_cv to 0.")
        else:
            kurt_cv = kurtosis(cv, axis=None)
            skew_cv = skew(cv, axis=None)

        return mean_cv, std_cv, min_cv, max_cv, skew_cv, kurt_cv

    def uc_rk_distr(self):
        """
        Distribution of unconstrained ranks.
        """

        # Remove any rows with imaginary values.
        # TODO: need to check this is working properly
        rank_uncons = self.pop.extract_uncons_rank()

        # Setting axis = 0 ensures computation columnwise. Output is a row vector.
        mean_uc_rk = np.mean(rank_uncons)
        std_uc_rk = np.std(rank_uncons)
        min_uc_rk = np.min(rank_uncons)
        max_uc_rk = np.max(rank_uncons)

        # Check if the vector is constant
        if std_uc_rk == 0:
            kurt_uc_rk = skew_uc_rk = 0
            print(
                "Unconstrained ranks are all equal - setting skew_uc_rk = kurtosis_uc_rk to 0."
            )
        else:
            kurt_uc_rk = kurtosis(rank_uncons)
            skew_uc_rk = skew(rank_uncons)

        return mean_uc_rk, std_uc_rk, min_uc_rk, max_uc_rk, skew_uc_rk, kurt_uc_rk

    def cv_var_mdl(self):
        """
        Fit a linear model to decision variables-constraint violation, then take the R2 and difference between the max and min of the absolute values of the linear model coefficients.
        """
        var = self.pop.extract_var()
        cv = self.pop.extract_cv()

        cv_mdl_r2, cv_range_coeff = Analysis.fit_linear_mdl(var, cv)

        return cv_mdl_r2, cv_range_coeff

    def dist_corr(self):
        """
        Distance correlation.

        Distance for each solution to nearest global minimum in decision space. Correlation of distance and constraints norm.

        """

        var = self.pop.extract_var()
        cv = self.pop.extract_cv()

        # Reshape nondominated variables array to be 2D if needed.
        nondominated_var = self.pop.extract_nondominated().extract_var()
        if nondominated_var.ndim == 1:
            nondominated_var = np.reshape(nondominated_var, (1, -1))

        # For each ND decision variable, find the smallest distance to the nearest population decision variable.
        dist_matrix = cdist(nondominated_var, var, "euclidean")
        min_dist = np.min(dist_matrix, axis=0)

        # Then compute correlation coefficient between CV and these minimum distances.
        dist_c_corr = Analysis.corr_coef(cv, min_dist)

        return dist_c_corr

    def corr_obj(self):
        """
        Significant correlation between objective values.

        Verel2013 provides a detailed derivation, but it seems as if we assume the objective correlation is the same metween all objectives. That is, for a correlation matrix C, C_np = rho for n != p, C_np = 1 for n = p. We want the value of rho.

        Since correlations are assumed to be equal across all objectives, we can just compute one pairwise correlation coefficient. Alsouly finds the same but computes the symmetric 2x2 correlation coefficient and pvalue matrices before extracting the upper-right value.
        """

        # Determine the symmetric correlation matrix.
        obj = self.pop.extract_obj()
        corr_obj, _ = Analysis.compute_correlation_matrix(
            obj, correlation_type="spearman", alpha=0.05
        )

        # So that we do not consider entries on the main diagonal = corr(x,x), fill these with nans.
        # Taken from: https://stackoverflow.com/questions/29394377/minimum-of-numpy-array-ignoring-diagonal
        mask = np.ones((corr_obj.shape[0], corr_obj.shape[0]))
        mask = (mask - np.diag(np.ones(corr_obj.shape[0]))).astype(bool)

        # Extract min, max, range
        corr_obj_min = np.amin(corr_obj[mask])
        corr_obj_max = np.amax(corr_obj[mask])
        corr_obj_rnge = corr_obj_min - corr_obj_max

        return corr_obj_min, corr_obj_max, corr_obj_rnge

    def compute_ps_pf_distances(self, norm_method):
        """
        Properties of the estimated (and normalised) Pareto-Set (PS) and Pareto-Front (PF). Includes global maximum, global mean and mean IQR of distances across the PS/PF.
        """
        # Extract normalisation values.
        var_lb, var_ub, obj_lb, obj_ub, cv_lb, cv_ub = super().extract_norm_values(
            norm_method
        )
        obj = Analysis.apply_normalisation(self.pop.extract_obj(), obj_lb, obj_ub)
        var = Analysis.apply_normalisation(self.pop.extract_var(), var_lb, var_ub)
        cv = Analysis.apply_normalisation(self.pop.extract_cv(), cv_lb, cv_ub)

        # Compute IGD between normalised PF and cloud of points formed by this sample.
        IGDind = IGD(
            Analysis.apply_normalisation(self.pop.extract_pf(), obj_lb, obj_ub)
        )
        PFd = IGDind(obj)

        # Initialise binary tree for nearest neighbour lookup on normalised PF.
        tree = cKDTree(obj)

        # Query the tree to find the nearest neighbours in obj for each point on the PF.
        distances, indices = tree.query(
            Analysis.apply_normalisation(self.pop.extract_pf(), obj_lb, obj_ub),
            k=20,
            workers=-1,
        )  # use parallel processing

        # For each point in the Pareto front, average the CV of the nearest neighbours to the PF in the sample.
        avg_cv_neighbours = []
        for i in range(indices.shape[0]):
            avg_cv_neighbours.append(np.mean(cv[indices[i, :]]))

        # Compute global average
        PFCV = np.mean(avg_cv_neighbours)

        # Initialize metrics.
        PS_dist_max = 0
        PS_dist_mean = 0
        PS_dist_iqr = 0
        PF_dist_max = 0
        PF_dist_mean = 0
        PF_dist_iqr = 0

        if obj.size > 1:
            # Constrained ranks.
            ranks = self.pop.extract_rank()

            # Distance across and between n1 and n2 rank fronts in decision space. Each argument of cdist should be arrays corresponding to the DVs on front n1 and front n2.
            var_dist_matrix = pdist(var[ranks == 1, :], "euclidean")
            obj_dist_matrix = pdist(obj[ranks == 1, :], "euclidean")

            # Compute statistics on this var_dist_matrix, presuming there is more than one point on the front.
            if var_dist_matrix.size != 0:
                PS_dist_max = np.max(var_dist_matrix)
                PS_dist_mean = np.mean(var_dist_matrix)
                PF_dist_max = np.max(obj_dist_matrix)
                PF_dist_mean = np.mean(obj_dist_matrix)

                # Find average of 25th and 75th percentile values.
                PS_dist_iqr = iqr(var_dist_matrix)
                PF_dist_iqr = iqr(obj_dist_matrix)

        return (
            PFd,
            PFCV,
            PS_dist_max,
            PS_dist_mean,
            PS_dist_iqr,
            PF_dist_max,
            PF_dist_mean,
            PF_dist_iqr,
        )

    def obj_skew(self):
        """
        Checks the skewness of the objective values for this population - only univariate avg/max/min/range.
        """
        obj = self.pop.extract_obj()
        skew_avg = np.mean(skew(obj, axis=0))
        skew_max = np.max(skew(obj, axis=0))
        skew_min = np.min(skew(obj, axis=0))
        skew_rnge = skew_max - skew_min
        return skew_avg, skew_min, skew_max, skew_rnge

    def obj_kurt(self):
        """
        Checks the kurtosis of the objective values for this population - only univariate avg/max/min/range.
        """
        obj = self.pop.extract_obj()
        kurt_avg = np.mean(kurtosis(obj, axis=0))
        kurt_max = np.max(kurtosis(obj, axis=0))
        kurt_min = np.min(kurtosis(obj, axis=0))
        kurt_rnge = kurt_max - kurt_min
        return kurt_avg, kurt_min, kurt_max, kurt_rnge

    def compute_ranks_cv_corr(self):
        """
        Compute correlation between objective values and norm violation values using Spearman's rank correlation coefficient.
        """

        obj = self.pop.extract_obj()
        cv = self.pop.extract_cv()
        ranks_cons = self.pop.extract_rank()
        ranks_uncons = self.pop.extract_uncons_rank()

        # Initialise correlation between objectives.
        corr_obj_cv = np.zeros(obj.shape[0])
        corr_obj_uc_rk = np.zeros(obj.shape[0])

        # Find correlations of each objective function with the CVs.
        for i in range(obj.shape[1]):
            objx = obj[:, i]

            # Compute correlation.
            corr_obj_cv[i] = Analysis.corr_coef(cv, objx, spearman=True)
            corr_obj_uc_rk[i] = Analysis.corr_coef(ranks_uncons, objx, spearman=True)

        # Compute max and min.
        corr_obj_cv_min = np.min(corr_obj_cv)
        corr_obj_cv_max = np.max(corr_obj_cv)
        corr_obj_uc_rk_min = np.min(corr_obj_uc_rk)
        corr_obj_uc_rk_max = np.max(corr_obj_uc_rk)

        # Find Spearman's correlation between CV and ranks of solutions.
        corr_cv_ranks = Analysis.corr_coef(cv, ranks_cons, spearman=True)

        return (
            corr_obj_cv_min,
            corr_obj_cv_max,
            corr_obj_uc_rk_min,
            corr_obj_uc_rk_max,
            corr_cv_ranks,
        )

    def PiIZ(self):
        """
        Compute proportion of solutions in the ideal zone (the lower left quadrant of the fitness-violation scatterplot) for each objective and for unconstrained fronts-violation scatterplot.

        May need to play around with the axis definitions while debugging.
        """

        # Extracting matrices.
        obj = self.pop.extract_obj()
        var = self.pop.extract_var()
        cons = self.pop.extract_cons()
        cv = self.pop.extract_cv()

        # Remove imaginary rows. Deep copies are created here.
        obj = Analysis.remove_imag_rows(obj)
        var = Analysis.remove_imag_rows(var)

        # Defining the ideal zone.
        minobjs = np.min(obj, axis=0)
        maxobjs = np.max(obj, axis=0)
        mincv = np.min(cv, axis=0)
        maxcv = np.max(cv, axis=0)
        mconsIdealPoint = mincv + (0.25 * (maxcv - mincv))
        conZone = np.all(cv <= mconsIdealPoint, axis=1)

        # Find PiZ for each objXcon
        piz_ob = np.zeros(obj.shape[1])
        for i in range(obj.shape[1]):
            objIdealPoint = minobjs[i] + (0.25 * (maxobjs[i] - minobjs[i]))
            objx = obj[:, i]
            iz = np.asarray(objx[conZone] <= objIdealPoint)

            piz_ob[i] = np.count_nonzero(iz) / self.pop.extract_obj().shape[0]

        # Find PiZ for each frontsXcon

        # May need to transpose.
        uncons_ranks = self.pop.extract_uncons_rank()

        # Axes may need to change depending on the structure of ranks. Right now we are taking the min of a column vector.
        minrank = np.min(uncons_ranks)
        maxrank = np.max(uncons_ranks)
        rankIdealPoint = minrank + (0.25 * (maxrank - minrank))
        iz = np.asarray(uncons_ranks[conZone] <= rankIdealPoint)
        piz_f = np.count_nonzero(iz) / self.pop.extract_obj().size

        # Return summary statistics.
        piz_ob_min = np.min(piz_ob)
        piz_ob_max = np.max(piz_ob)

        return piz_ob_min, piz_ob_max, piz_f

    def rk_uc_var_mdl(self):
        """
        Fit a linear model to decision variables-front location, then take the R2 and difference between the max and min of the absolute values of the linear model coefficients.

        Very similar to the cv_mdl function, except the model is fit to different parameters.
        """

        var = self.pop.extract_var()
        uncons_ranks = self.pop.extract_uncons_rank()

        # Reshape data for compatibility. Assumes that y = mx + b where x is a matrix, y is a column vector
        uncons_ranks = uncons_ranks.reshape((-1, 1))

        # Fit linear model and compute adjusted R2 and difference between variable coefficients.
        rk_uc_mdl_r2, rk_uc_range_coeff = Analysis.fit_linear_mdl(var, uncons_ranks)

        return rk_uc_mdl_r2, rk_uc_range_coeff

    def compute_fsr(self):
        feasible = self.pop.extract_feasible()
        return len(feasible) / len(self.pop)

    def compute_PF_UPF_features(self, norm_method):
        # Extract normalisation values.
        var_lb, var_ub, obj_lb, obj_ub, cv_lb, cv_ub = super().extract_norm_values(
            norm_method
        )

        # Define the nadir for HV calculations.
        nadir = 1.1 * np.ones(obj_lb.size)

        # Extract constrained and unconstrained non-dominated individuals.
        nondominated_cons = self.pop.extract_nondominated(constrained=True)
        nondominated_uncons = self.pop.extract_nondominated(constrained=False)

        # Normalise.
        obj_cons, _ = Analysis.trim_obj_using_nadir(
            Analysis.apply_normalisation(
                nondominated_cons.extract_obj(), obj_lb, obj_ub
            ),
            nadir,
        )
        obj_uncons, _ = Analysis.trim_obj_using_nadir(
            Analysis.apply_normalisation(
                nondominated_uncons.extract_obj(), obj_lb, obj_ub
            ),
            nadir,
        )

        # Hypervolume of estimated PF (constrained, unconstrained).
        try:
            hv_est = calculate_hypervolume_pygmo(obj_cons, nadir)
        except:
            hv_est = np.nan

        try:
            uhv_est = calculate_hypervolume_pygmo(obj_uncons, nadir)
        except:
            uhv_est = np.nan

        hv_uhv_n = hv_est / uhv_est

        # Compute generational distance between constrained and unconstrained PF. First need to create indicator object from pymoo.
        GDind = GD(obj_uncons)
        GD_cpo_upo = GDind(obj_cons)

        # Proportion of sizes of PF and UPFs.
        cpo_upo_n = len(nondominated_cons) / len(nondominated_uncons)

        # Proportion of PO solutions.
        upo_n = len(nondominated_uncons) / len(self.pop)
        po_n = len(nondominated_cons) / len(self.pop)

        # Proportion of UPF covered by PF.

        # Make a merged population and evaluate.
        # TODO: time consuming step - could just use parallel NDSort here.
        merged_dec = np.vstack(
            (nondominated_cons.extract_var(), nondominated_uncons.extract_var())
        )
        new_pop = Population(self.pop[0].problem, n_individuals=merged_dec.shape[0])
        new_pop.evaluate(merged_dec, eval_fronts=True)

        cons_combined_ranks = new_pop.extract_rank()[: len(nondominated_cons)]
        uncons_combined_ranks = new_pop.extract_rank()[len(nondominated_cons) :]

        # Count elements in 'a' that are less than or equal to at least one value in 'y'
        count_upf_dominates_pf = np.sum(
            np.any(
                uncons_combined_ranks <= cons_combined_ranks[:, np.newaxis],
                axis=0,
            )
        )

        cover_cpo_upo_n = count_upf_dominates_pf / len(uncons_combined_ranks)

        return (
            hv_est,
            uhv_est,
            hv_uhv_n,
            GD_cpo_upo,
            upo_n,
            po_n,
            cpo_upo_n,
            cover_cpo_upo_n,
        )

    @staticmethod
    def compute_ic_features(pop, sample_type="global"):
        # Can be reused for RW and global samples.

        # Setup
        if sample_type == "global":
            ic_sorting = (
                "nn"  # nearest neighbours sorting. Can change to random if needed.
            )
        elif sample_type == "rw":
            ic_sorting = None  # no sorting needed for RWs.
        ic_nn_neighborhood = 20
        ic_epsilon = np.insert(10 ** np.linspace(start=-5, stop=15, num=1000), 0, 0)
        ic_settling_sensitivity = 0.05
        ic_info_sensitivity = 0.5

        # Taken directly from https://github.com/Reiyan/pflacco/blob/master/pflacco/classical_ela_features.py
        X = pd.DataFrame(pop.extract_var())
        y = pd.Series(pop.extract_cv().ravel(), name="y")
        epsilon = ic_epsilon  # can probably remove later.

        n = X.shape[1]

        # dist based on ic_sorting
        if ic_sorting == "random":
            permutation = np.random.choice(
                range(X.shape[0]), size=X.shape[0], replace=False
            )
            X = X.iloc[permutation].reset_index(drop=True)
            d = [
                np.sqrt((X.iloc[idx] - X.iloc[idx + 1]).pow(2).sum())
                for idx in range(X.shape[0] - 1)
            ]
        elif ic_sorting is None:
            # Keep the points in the same order.
            permutation = range(0, X.shape[0])

            # Calculate the consecutive differences
            diff_x = np.diff(X, axis=0)
            # Calculate the norm of each difference vector
            d = np.linalg.norm(diff_x, axis=1)

        elif ic_sorting == "nn":  # nearest neighbours
            # Randomly choose start point.
            ic_nn_start = np.random.choice(range(X.shape[0]), size=1)[0]
            nbrs = NearestNeighbors(
                n_neighbors=min(ic_nn_neighborhood, X.shape[0]), algorithm="kd_tree"
            ).fit(X)
            distances, indices = nbrs.kneighbors(X)

            current = ic_nn_start
            candidates = np.delete(np.array([x for x in range(X.shape[0])]), current)
            permutation = [current]
            permutation.extend([None] * (X.shape[0] - 1))
            dists = [None] * (X.shape[0])

            for i in range(1, X.shape[0]):
                currents = indices[permutation[i - 1]]
                current = np.array([x for x in currents if x in candidates])
                if len(current) > 0:
                    current = current[0]
                    permutation[i] = current
                    candidates = candidates[candidates != current]
                    dists[i] = distances[permutation[i - 1], currents == current][0]
                else:
                    nbrs2 = NearestNeighbors(n_neighbors=min(1, X.shape[0])).fit(
                        X.iloc[candidates].to_numpy()
                    )
                    (
                        distances2,
                        indices2,
                    ) = nbrs2.kneighbors(
                        X.iloc[permutation[i - 1]].to_numpy().reshape(1, X.shape[1])
                    )
                    current = candidates[np.ravel(indices2)[0]]
                    permutation[i] = current
                    candidates = candidates[candidates != current]
                    dists[i] = np.ravel(distances2)[0]

            d = dists[1:]

        # Calculate phi eps
        phi_eps = []
        y_perm = y[permutation]
        diff_y = np.ediff1d(y_perm)
        ratio = diff_y / d
        for eps in ic_epsilon:
            phi_eps.append([0 if abs(x) < eps else np.sign(x) for x in ratio])

        phi_eps = np.array(phi_eps)
        H = []
        M = []
        for row in phi_eps:
            # Calculate H values
            a = row[:-1]
            b = row[1:]
            probs = []
            probs.append(np.bitwise_and(a == -1, b == 0).mean())
            probs.append(np.bitwise_and(a == -1, b == 1).mean())
            probs.append(np.bitwise_and(a == 0, b == -1).mean())
            probs.append(np.bitwise_and(a == 0, b == 1).mean())
            probs.append(np.bitwise_and(a == 1, b == -1).mean())
            probs.append(np.bitwise_and(a == 1, b == 0).mean())
            H.append(-sum([0 if x == 0 else x * np.log(x) / np.log(6) for x in probs]))

            # Calculate M values
            n = len(row)
            row = row[row != 0]
            len_row = (
                len(row[np.insert(np.ediff1d(row) != 0, 0, False)])
                if len(row) > 0
                else 0
            )
            M.append(len_row / (n - 1))

        H = np.array(H)
        M = np.array(M)
        eps_s = epsilon[H < ic_settling_sensitivity]
        eps_s = (
            np.log(eps_s.min()) / np.log(10)
            if len(eps_s) > 0 and eps_s.min() != 0
            else np.nan
        )

        m0 = M[epsilon == 0][0]
        eps05 = np.where(M > ic_info_sensitivity * m0)[0]
        eps05 = np.log(epsilon[eps05].max()) / np.log(10) if len(eps05) > 0 else np.nan
        return (H.max(), eps_s, m0, eps05)

    def eval_features(self):
        # Remove any samples if they contain infs or nans.
        new_pop, _ = self.pop.remove_nan_inf_rows("global", re_evaluate=True)

        # Global scr. "glob" will be appended to the name in the results file.
        self.features["scr"] = Analysis.compute_solver_crash_ratio(self.pop, new_pop)

        # Now work with the trimmed population from now on.
        self.pop = new_pop

        # Feasibility
        self.features["fsr"] = self.compute_fsr()

        # Correlation of objectives.
        (
            self.features["corr_obj_min"],
            self.features["corr_obj_max"],
            self.features["corr_obj_range"],
        ) = self.corr_obj()

        # Skewness of objective values.
        (
            self.features["skew_avg"],
            self.features["skew_min"],
            self.features["skew_max"],
            self.features["skew_rnge"],
        ) = self.obj_skew()

        # Kurtosis of objective values.
        (
            self.features["kurt_avg"],
            self.features["kurt_min"],
            self.features["kurt_max"],
            self.features["kurt_rnge"],
        ) = self.obj_kurt()

        # Distribution of unconstrained ranks.
        (
            self.features["mean_uc_rk"],
            self.features["std_uc_rk"],
            self.features["min_uc_rk"],
            self.features["max_uc_rk"],
            self.features["skew_uc_rk"],
            self.features["kurt_uc_rk"],
        ) = self.uc_rk_distr()

        # Distribution of CV (normalised).
        (
            self.features["mean_cv"],
            self.features["std_cv"],
            self.features["min_cv"],
            self.features["max_cv"],
            self.features["skew_cv"],
            self.features["kurt_cv"],
        ) = self.cv_distr(norm_method="95th")

        # Proportion of solutions in ideal zone per objectives and overall proportion of solutions in ideal zone.
        (
            self.features["piz_ob_min"],
            self.features["piz_ob_max"],
            self.features["piz_ob_f"],
        ) = self.PiIZ()

        # Pareto set and front properties (normalised).
        (
            self.features["PFd"],
            self.features["PFCV"],
            self.features["PS_dist_max"],
            self.features["PS_dist_mean"],
            self.features["PS_dist_iqr"],
            self.features["PF_dist_max"],
            self.features["PF_dist_mean"],
            self.features["PF_dist_iqr"],
        ) = self.compute_ps_pf_distances(norm_method="95th")

        # Get PF-UPF relationship features.
        (
            self.features["hv_est"],
            self.features["uhv_est"],
            self.features["hv_uhv_n"],
            self.features["GD_cpo_upo"],
            self.features["upo_n"],
            self.features["po_n"],
            self.features["cpo_upo_n"],
            self.features["cover_cpo_upo_n"],
        ) = self.compute_PF_UPF_features(norm_method="95th")

        # Extract violation-distance correlation.
        self.features["dist_c_corr"] = self.dist_corr()

        # Correlations of objectives with cv, unconstrained ranks and then cv with ranks.
        (
            self.features["corr_obj_cv_min"],
            self.features["corr_obj_cv_max"],
            self.features["corr_obj_uc_rk_min"],
            self.features["corr_obj_uc_rk_max"],
            self.features["corr_cv_ranks"],
        ) = self.compute_ranks_cv_corr()

        # Decision variables-unconstrained ranks model properties.
        (
            self.features["rk_uc_mdl_r2"],
            self.features["rk_uc_range_coeff"],
        ) = self.rk_uc_var_mdl()

        # Decision variables-CV model properties.
        (
            self.features["cv_mdl_r2"],
            self.features["cv_range_coeff"],
        ) = self.rk_uc_var_mdl()

        # Information content features.
        (
            self.features["H_max"],
            self.features["eps_s"],
            self.features["m0"],
            self.features["eps05"],
        ) = GlobalAnalysis.compute_ic_features(self.pop, sample_type="global")
