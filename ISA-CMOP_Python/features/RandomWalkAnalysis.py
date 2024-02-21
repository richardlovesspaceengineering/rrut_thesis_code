import numpy as np
from scipy.stats import yeojohnson
from features.Analysis import Analysis
from features.GlobalAnalysis import GlobalAnalysis
import numpy as np
from scipy.spatial.distance import cdist
from features.ancillary_functions import *
from optimisation.util.calculate_hypervolume import calculate_hypervolume_pygmo
from optimisation.model.population import Population
from features.ancillary_functions import *
import copy
import time


class RandomWalkAnalysis(Analysis):
    """
    Calculate all features generated from random walk samples.

    Populations is a list of populations that represents a walk, each entry is a solution and its neighbours.
    """

    def __init__(
        self, pop_walk, pop_neighbours_list, normalisation_values, results_dir
    ):
        """
        Populations must already be evaluated.
        """

        self.pop_walk = pop_walk
        self.pop_neighbours_list = pop_neighbours_list
        self.normalisation_values = normalisation_values
        self.features = {}
        self.results_dir = results_dir

    def clear_for_storage(self):
        self.pop_walk = None
        self.pop_neighbours_list = None

    def create_empty_analysis_obj(self):
        return RandomWalkAnalysis(
            None, None, self.normalisation_values, self.results_dir
        )

    def preprocess_nans_on_walks(self):
        # Remove any steps and corresponding neighbours if they contain infs or nans.
        pop_walk_new, num_rows_removed = self.pop_walk.remove_nan_inf_rows(
            "walk", re_evaluate=True
        )
        removal_idx = self.pop_walk.get_nan_inf_idx()
        pop_neighbours_new = [
            n for i, n in enumerate(self.pop_neighbours_list) if i not in removal_idx
        ]

        # Remove any neighbours if they contain infs or nans.
        var = pop_walk_new.extract_var()

        # Make new list of populations for neighbours.
        pop_neighbours_checked = []

        # Loop over each solution in the walk.
        for i in range(var.shape[0]):
            # Extract neighbours for this point and append.
            pop_neighbourhood = copy.deepcopy(pop_neighbours_new[i])
            pop_neighbourhood, num_rows_removed = pop_neighbourhood.remove_nan_inf_rows(
                "neig", re_evaluate=True
            )  # Don't think we need to revaluate fronts.

            # Save to list.
            pop_neighbours_checked.append(pop_neighbourhood)

        return pop_walk_new, pop_neighbours_new, pop_neighbours_checked

    def extract_feasible_steps_neighbours(self):
        # Remove any steps and corresponding neighbours if they are infeasible
        pop_walk_feas = self.pop_walk.extract_feasible()
        removal_idx = self.pop_walk.get_infeas_idx()
        pop_neighbours_new = [
            n for i, n in enumerate(self.pop_neighbours_list) if i not in removal_idx
        ]

        # Make new list of populations for neighbours.
        pop_neighbours_checked = []

        # Loop over each solution in the walk.
        for i in range(len(pop_walk_feas)):
            # Extract neighbours for this point and append.
            pop_neighbourhood = copy.deepcopy(pop_neighbours_new[i])
            pop_neighbourhood = pop_neighbourhood.extract_feasible()

            # Save to list.
            pop_neighbours_checked.append(pop_neighbourhood)

        return pop_walk_feas, pop_neighbours_new, pop_neighbours_checked

    @staticmethod
    def trim_neig_using_mask(lst, mask):
        # Use list comprehension and zip to create a new list with elements
        # corresponding to False in the mask
        filtered_list = [value for value, keep in zip(lst, mask) if keep]

        return filtered_list

    @staticmethod
    def compute_neighbourhood_crash_ratio(
        full_pop_neighbours_list, trimmed_pop_neighbours_list
    ):
        """
        Proportion of neighbourhood solutions that crash the solver.
        """

        ncr_array = np.zeros(len(full_pop_neighbours_list))

        for i in range(len(full_pop_neighbours_list)):
            full_neig = full_pop_neighbours_list[i]
            trimmed_neig = trimmed_pop_neighbours_list[i]

            # Compute ratio.
            ncr_array[i] = 1 - len(trimmed_neig) / len(full_neig)

        ncr_avg = np.mean(ncr_array)
        ncr_r1 = Analysis.autocorr(ncr_array, lag=1)

        return ncr_avg, ncr_r1

    def compute_neighbourhood_distance_features(self, norm_method):
        """
        Calculate neighbourhood_features.

        pop is a matrix representing a CMOP evaluated over a random walk, each row is a solution (step) and its neighbours.

        Instances is the problem name, PF is the known Pareto Front.

        Currently only returns [dist_f_dist_x_avg_rws, dist_c_dist_x_avg_rws, bhv_avg_rws] since these are the only features required in Eq.(13) of Alsouly.
        """

        # Extract normalisation values.
        var_lb, var_ub, obj_lb, obj_ub, cv_lb, cv_ub = super().extract_norm_values(
            norm_method
        )

        # Extract walk arrays.
        walk_var = Analysis.apply_normalisation(
            self.pop_walk.extract_var(), var_lb, var_ub
        )
        walk_obj = Analysis.apply_normalisation(
            self.pop_walk.extract_obj(), obj_lb, obj_ub
        )
        walk_cv = Analysis.apply_normalisation(self.pop_walk.extract_cv(), cv_lb, cv_ub)

        # Initialise arrays.
        dist_x_array = np.zeros(len(self.pop_walk))
        dist_f_array = np.zeros(len(self.pop_walk))
        dist_c_array = np.zeros(len(self.pop_walk))
        dist_f_c_array = np.zeros(len(self.pop_walk))
        dist_f_dist_x_array = np.zeros(len(self.pop_walk))
        dist_c_dist_x_array = np.zeros(len(self.pop_walk))
        dist_f_c_dist_x_array = np.zeros(len(self.pop_walk))

        # Loop over each solution in the walk.
        for i in range(len(self.pop_walk)):
            # Extract step values.
            step_var = np.atleast_2d(walk_var[i, :])
            step_obj = np.atleast_2d(walk_obj[i, :])
            step_cv = np.atleast_2d(walk_cv[i])

            # Extract neighbours for this point and append.
            pop_neighbourhood = copy.deepcopy(self.pop_neighbours_list[i])

            # Extract evaluated values for this neighbourhood and apply normalisation.
            neig_var = Analysis.apply_normalisation(
                pop_neighbourhood.extract_var(), var_lb, var_ub
            )
            neig_obj = Analysis.apply_normalisation(
                pop_neighbourhood.extract_obj(), obj_lb, obj_ub
            )
            neig_cv = Analysis.apply_normalisation(
                pop_neighbourhood.extract_cv(), cv_lb, cv_ub
            )

            # Distance from solution to neighbours in variable space.
            distdec = cdist(step_var, neig_var, "euclidean")
            dist_x_array[i] = np.mean(distdec)

            # Distance from solution to neighbours in objective space.
            distobj = cdist(step_obj, neig_obj, "euclidean")
            dist_f_array[i] = np.mean(distobj)

            # Distance from solution to neighbours in constraint-norm space.
            distcons = cdist(step_cv, neig_cv, "euclidean")
            dist_c_array[i] = np.mean(distcons)

            # Distance between neighbours in objective-violation space.

            # Construct objective-violation space by horizontally joining objectives and CV so that each solution forms a ((objectives), cv) tuple.
            step_obj_violation = np.concatenate((step_obj, step_cv), axis=1)
            neig_obj_violation = np.concatenate((neig_obj, neig_cv), axis=1)
            dist_obj_violation = cdist(
                step_obj_violation, neig_obj_violation, "euclidean"
            )
            dist_f_c_array[i] = np.mean(dist_obj_violation)

            # Take ratios.
            dist_f_dist_x_array[i] = dist_f_array[i] / dist_x_array[i]
            dist_c_dist_x_array[i] = dist_c_array[i] / dist_x_array[i]
            dist_f_c_dist_x_array[i] = dist_f_c_array[i] / dist_x_array[i]

        # Aggregate for this walk.
        dist_x_avg = np.mean(dist_x_array)
        dist_f_avg = np.mean(dist_f_array)
        dist_c_avg = np.mean(dist_c_array)
        dist_f_c_avg = np.mean(dist_f_c_array)
        dist_f_dist_x_avg = np.mean(dist_f_dist_x_array)
        dist_c_dist_x_avg = np.mean(dist_c_dist_x_array)
        dist_f_c_dist_x_avg = np.mean(dist_f_c_dist_x_array)

        # Compute Analysis.autocorrelations
        dist_x_r1 = Analysis.autocorr(dist_x_array, lag=1)
        dist_f_r1 = Analysis.autocorr(dist_f_array, lag=1)
        dist_c_r1 = Analysis.autocorr(dist_c_array, lag=1)
        dist_f_c_r1 = Analysis.autocorr(dist_f_c_array, lag=1)
        dist_f_dist_x_r1 = Analysis.autocorr(dist_f_dist_x_array, lag=1)
        dist_c_dist_x_r1 = Analysis.autocorr(dist_c_dist_x_array, lag=1)
        dist_f_c_dist_x_r1 = Analysis.autocorr(dist_f_c_dist_x_array, lag=1)

        return (
            dist_x_avg,
            dist_x_r1,
            dist_f_avg,
            dist_f_r1,
            dist_c_avg,
            dist_c_r1,
            dist_f_c_avg,
            dist_f_c_r1,
            dist_f_dist_x_avg,
            dist_f_dist_x_r1,
            dist_c_dist_x_avg,
            dist_c_dist_x_r1,
            dist_f_c_dist_x_avg,
            dist_f_c_dist_x_r1,
        )

    def compute_cons_neighbourhood_hv_features(self, norm_method):
        (
            pop_walk_feas,
            _,
            pop_neighbours_feas,
        ) = self.extract_feasible_steps_neighbours()

        return self.compute_neighbourhood_hv_features(
            pop_walk_feas, pop_neighbours_feas, norm_method
        )

    def compute_uncons_neighbourhood_hv_features(self, norm_method):
        return self.compute_neighbourhood_hv_features(
            self.pop_walk, self.pop_neighbours_list, norm_method
        )

    def compute_neighbourhood_hv_features(
        self, pop_walk, pop_neighbours_list, norm_method
    ):
        """
        This function is used for feasible and unconstrained spaces.
        """

        # Extract normalisation values.
        var_lb, var_ub, obj_lb, obj_ub, cv_lb, cv_ub = super().extract_norm_values(
            norm_method
        )

        # Define the nadir
        nadir = 1.1 * np.ones(obj_lb.size)

        if len(pop_walk) == 0:
            # If we are here, the population is empty. This might occur when using this function to determine constrained HV values and there are no feasible solutions.
            return (0,) * 8  # this function returns 8 values

        # Extract evaluated population values, normalise and trim any points larger than the nadir.
        obj = pop_walk.extract_obj()

        obj, mask = Analysis.trim_obj_using_nadir(
            Analysis.apply_normalisation(pop_walk.extract_obj(), obj_lb, obj_ub),
            nadir,
        )

        num_rows_trimmed = len(pop_walk) - np.count_nonzero(mask)
        if num_rows_trimmed > 0:
            print(
                "Had to remove {} rows that were further than the nadir from the origin.".format(
                    num_rows_trimmed
                )
            )

        # Also remove corresponding neighbours for steps on the walk larger than the nadir.
        pop_neighbours_list = RandomWalkAnalysis.trim_neig_using_mask(
            pop_neighbours_list, mask
        )

        # Initialise arrays
        hv_ss_array = np.zeros(obj.shape[0])
        nhv_array = np.zeros(obj.shape[0])
        hvd_array = np.zeros(obj.shape[0])
        bhv_array = np.zeros(obj.shape[0])

        # Set nonsense value if obj.size == 0
        if obj.size == 0:
            hv_ss_array.fill(0)
            nhv_array.fill(0)
            hvd_array.fill(0)
            bhv_array.fill(0)
            print(
                "There are no individuals closer to the origin than the nadir. Setting all step HV metrics for this sample to 0."
            )
        else:
            # Loop over each solution in the walk.
            for i in range(obj.shape[0]):
                # Extract neighbours for this point and normalise.
                pop_neighbourhood = pop_neighbours_list[i]

                # Quick check to see if this is empty.
                if len(pop_neighbourhood) == 0:
                    hv_ss_array[i] = 0
                    nhv_array[i] = 0
                    hvd_array[i] = 0
                    bhv_array[i] = 0
                    print(
                        "There are no neighbours for step {} of {} - this may occur when computing HVs of feasible solutions. Setting all neighbourhood HV metrics for this step to 0.".format(
                            i + 1, obj.shape[0]
                        )
                    )
                    continue

                neig_obj, _ = Analysis.trim_obj_using_nadir(
                    Analysis.apply_normalisation(
                        pop_neighbourhood.extract_obj(), obj_lb, obj_ub
                    ),
                    nadir,
                )
                if neig_obj.size == 0:
                    hv_ss_array[i] = 0
                    nhv_array[i] = 0
                    hvd_array[i] = 0
                    bhv_array[i] = 0
                    print(
                        "There are no neighbours closer to the origin than the nadir for step {} of {}. Setting all neighbourhood HV metrics for this step to 0.".format(
                            i + 1, obj.shape[0]
                        )
                    )
                    continue
                else:
                    # Compute HV of single solution at this step.
                    hv_ss_array[i] = calculate_hypervolume_pygmo(
                        np.atleast_2d(obj[i, :]), nadir
                    )

                    # Compute HV of neighbourhood
                    nhv_array[i] = calculate_hypervolume_pygmo(neig_obj, nadir)

                    # Compute HV difference between neighbours and that covered by the current solution
                    hvd_array[i] = nhv_array[i] - hv_ss_array[i]

                    # Compute HV of non-dominated neighbours (trimmed).
                    # print(f"Neig rank: {pop_neighbourhood.extract_rank()}")
                    # print(len(pop_neighbourhood.extract_nondominated()))
                    bestrankobjs, _ = Analysis.trim_obj_using_nadir(
                        Analysis.apply_normalisation(
                            pop_neighbourhood.extract_nondominated().extract_obj(),
                            obj_lb,
                            obj_ub,
                        ),
                        nadir,
                    )
                    try:
                        bhv_array[i] = calculate_hypervolume_pygmo(bestrankobjs, nadir)
                    except:
                        # In case the NDFront is further from the origin than the nadir.
                        bhv_array[i] = 0
                        print(
                            "There are no non-dominated neighbours closer to the origin than the nadir for step {} of {}. Setting HV metric bhv_avg to 0.".format(
                                i + 1, obj.shape[0]
                            )
                        )

        # Compute means (allowing for nans if need be)
        hv_ss_avg = np.mean(hv_ss_array)
        nhv_avg = np.mean(nhv_array)
        hvd_avg = np.mean(hvd_array)
        bhv_avg = np.mean(bhv_array)

        # Compute Analysis.autocorrelations
        hv_ss_r1 = Analysis.autocorr(hv_ss_array, 1)
        nhv_r1 = Analysis.autocorr(nhv_array, 1)
        hvd_r1 = Analysis.autocorr(hvd_array, 1)
        bhv_r1 = Analysis.autocorr(bhv_array, 1)

        return (
            hv_ss_avg,
            hv_ss_r1,
            nhv_avg,
            nhv_r1,
            hvd_avg,
            hvd_r1,
            bhv_avg,
            bhv_r1,
        )

    def compute_neighbourhood_violation_features(self, norm_method):
        # Extract normalisation values.
        var_lb, var_ub, obj_lb, obj_ub, cv_lb, cv_ub = super().extract_norm_values(
            norm_method
        )

        # Extract evaluated population values.
        var = self.pop_walk.extract_var()
        obj = self.pop_walk.extract_obj()

        # Should only need to normalsie CV here.
        cv = Analysis.apply_normalisation(self.pop_walk.extract_cv(), cv_lb, cv_ub)

        # Initialise arrays.
        cross_array = np.zeros(var.shape[0] - 1)
        nncv_array = np.zeros(var.shape[0])
        ncv_array = np.zeros(var.shape[0])
        bncv_array = np.zeros(var.shape[0])

        # Maximum ratio of feasible boundary crossing.
        num_steps = obj.shape[0]
        num_feasible_steps = self.pop_walk.extract_feasible().extract_obj().shape[0]
        rfb_max = (
            2
            / (num_steps - 1)
            * np.minimum(
                np.minimum(num_feasible_steps, num_steps - num_feasible_steps),
                (num_steps - 1) / 2,
            )
        )

        if rfb_max == 0:
            compute_rfb = False
        else:
            compute_rfb = True

        # Initialise boundary crossings index.
        cross_idx = -1

        # Loop over each solution in the walk.
        for i in range(var.shape[0]):
            # Extract neighbours for this point and normalise.
            pop_neighbourhood = self.pop_neighbours_list[i]
            neig_cv = Analysis.apply_normalisation(
                pop_neighbourhood.extract_cv(), cv_lb, cv_ub
            )

            # Average neighbourhood violation value.
            nncv_array[i] = np.mean(neig_cv)

            # Single solution violation value.
            ncv_array[i] = cv[i]

            # Average violation value of neighbourhood's non-dominated solutions.
            bncv_array[i] = np.mean(
                Analysis.apply_normalisation(
                    pop_neighbourhood.extract_nondominated(
                        constrained=True
                    ).extract_cv(),
                    cv_lb,
                    cv_ub,
                )
            )

            # Feasible boundary crossing
            if compute_rfb:
                if i > 0:
                    cross_idx += 1
                    if cv[i] > 0 and cv[i - 1] > 0:
                        # Stayed in infeasible region.
                        cross_array[cross_idx] = 0
                    else:
                        if cv[i] <= 0 and cv[i - 1] <= 0:
                            # Stayed in feasible region.
                            cross_array[cross_idx] = 0
                        else:
                            # Crossed between regions.
                            cross_array[cross_idx] = 1

        # Aggregate total number of feasible boundary crossings.
        if compute_rfb:
            nrfbx = np.sum(cross_array) / (num_steps - 1) / rfb_max
        else:
            nrfbx = 0

        # Calculate means
        nncv_avg = np.mean(nncv_array)
        ncv_avg = np.mean(ncv_array)
        bncv_avg = np.mean(bncv_array)

        # Calculate Analysis.autocorrelations
        nncv_r1 = Analysis.autocorr(nncv_array, lag=1)
        ncv_r1 = Analysis.autocorr(ncv_array, lag=1)
        bncv_r1 = Analysis.autocorr(bncv_array, lag=1)

        return nrfbx, nncv_avg, nncv_r1, ncv_avg, ncv_r1, bncv_avg, bncv_r1

    def compute_neighbourhood_dominance_features(self):
        # Extract evaluated population values.
        var = self.pop_walk.extract_var()

        # Initialise arrays.
        sup_array = np.zeros(var.shape[0])
        inf_array = np.zeros(var.shape[0])
        inc_array = np.zeros(var.shape[0])
        lnd_array = np.zeros(var.shape[0])
        nfronts_array = np.zeros(var.shape[0])

        for i in range(var.shape[0]):
            # Extract neighbours and step populations.
            pop_neighbourhood = self.pop_neighbours_list[i]
            pop_step = self.pop_walk.slice_population(i, i + 1)  # get the current step.

            # Compute proportion of locally non-dominated solutions.
            lnd_array[i] = np.atleast_2d(
                pop_neighbourhood.extract_nondominated()
            ).shape[0] / len(pop_neighbourhood)

            # Create merged population and re-evaluate ranks.
            merged_pop = Population.merge(pop_step, pop_neighbourhood)
            merged_pop.evaluate_fronts()
            # print(pop_step.extract_var())
            # print(pop_neighbourhood.extract_var())
            # print(merged_pop.extract_var())

            ranks = merged_pop.extract_rank()
            step_rank = ranks[0]

            # Compute number of fronts in this neighbourhood relative to neighbourhood size.
            nfronts_array[i] = np.unique(ranks).size / len(pop_neighbourhood)

            # Compute proportion of neighbours dominated by current solution.
            dominated_neighbours = ranks[ranks > step_rank]
            sup_array[i] = dominated_neighbours.size / len(pop_neighbourhood)

            # Compute proportion of neighbours dominating the current solution.
            dominating_neighbours = ranks[ranks < step_rank]
            inf_array[i] = dominating_neighbours.size / len(pop_neighbourhood)

            # Compute proportion of neighbours incomparable to the current solution.
            incomparable_neighbours = ranks[ranks == step_rank] - 1
            inc_array[i] = incomparable_neighbours.size / len(pop_neighbourhood)

        # Calculate means
        sup_avg = np.mean(sup_array)
        inf_avg = np.mean(inf_array)
        inc_avg = np.mean(inc_array)
        lnd_avg = np.mean(lnd_array)
        nfronts_avg = np.mean(nfronts_array)

        # Calculate Analysis.autocorrelations
        sup_r1 = Analysis.autocorr(sup_array, lag=1)
        inf_r1 = Analysis.autocorr(inf_array, lag=1)
        inc_r1 = Analysis.autocorr(inc_array, lag=1)
        lnd_r1 = Analysis.autocorr(lnd_array, lag=1)
        nfronts_r1 = Analysis.autocorr(nfronts_array, lag=1)

        return (
            sup_avg,
            sup_r1,
            inf_avg,
            inf_r1,
            inc_avg,
            inc_r1,
            lnd_avg,
            lnd_r1,
            nfronts_avg,
            nfronts_r1,
        )

    def compute_solver_related_features(self):
        # Preprocess nans and infinities, compute solver crash ratio and update attributes.
        pop_new, pop_neighbours_new, pop_neighbours_checked = (
            self.preprocess_nans_on_walks()
        )
        self.features["scr"] = Analysis.compute_solver_crash_ratio(
            self.pop_walk, pop_new
        )
        (
            self.features["ncr_avg"],
            self.features["ncr_r1"],
        ) = RandomWalkAnalysis.compute_neighbourhood_crash_ratio(
            pop_neighbours_new, pop_neighbours_checked
        )

        # Update populations
        self.pop_walk = pop_new
        self.pop_neighbours_list = pop_neighbours_checked

    def eval_features(self):
        """
        Evaluate features along the random walk and save to class.
        """

        self.compute_solver_related_features()

        # Evaluate neighbourhood distance features
        (
            self.features["dist_x_avg"],
            self.features["dist_x_r1"],
            self.features["dist_f_avg"],
            self.features["dist_f_r1"],
            self.features["dist_c_avg"],
            self.features["dist_c_r1"],
            self.features["dist_f_c_avg"],
            self.features["dist_f_c_r1"],
            self.features["dist_f_dist_x_avg"],
            self.features["dist_f_dist_x_r1"],
            self.features["dist_c_dist_x_avg"],
            self.features["dist_c_dist_x_r1"],
            self.features["dist_f_c_dist_x_avg"],
            self.features["dist_f_c_dist_x_r1"],
        ) = self.compute_neighbourhood_distance_features(norm_method="95th")

        # Evaluate unconstrained neighbourhood HV features
        (
            self.features["uhv_ss_avg"],
            self.features["uhv_ss_r1"],
            self.features["nuhv_avg"],
            self.features["nuhv_r1"],
            self.features["uhvd_avg"],
            self.features["uhvd_r1"],
            _,
            _,
        ) = self.compute_uncons_neighbourhood_hv_features(norm_method="95th")

        (
            self.features["hv_ss_avg"],
            self.features["hv_ss_r1"],
            self.features["nhv_avg"],
            self.features["nhv_r1"],
            self.features["hvd_avg"],
            self.features["hvd_r1"],
            self.features["bhv_avg"],
            self.features["bhv_r1"],
        ) = self.compute_cons_neighbourhood_hv_features(norm_method="95th")

        # Evaluate neighbourhood violation features
        (
            self.features["nrfbx"],
            self.features["nncv_avg"],
            self.features["nncv_r1"],
            self.features["ncv_avg"],
            self.features["ncv_r1"],
            self.features["bncv_avg"],
            self.features["bncv_r1"],
        ) = self.compute_neighbourhood_violation_features(norm_method="95th")

        # Evaluate neighbourhood domination features. Note that no normalisation is needed for these.
        (
            self.features["sup_avg"],
            self.features["sup_r1"],
            self.features["inf_avg"],
            self.features["inf_r1"],
            self.features["inc_avg"],
            self.features["inc_r1"],
            self.features["lnd_avg"],
            self.features["lnd_r1"],
            self.features["nfronts_avg"],
            self.features["nfronts_r1"],
        ) = self.compute_neighbourhood_dominance_features()

        # Information content features.
        (
            self.features["H_max"],
            self.features["eps_s"],
            self.features["m0"],
            self.features["eps05"],
        ) = GlobalAnalysis.compute_ic_features(self.pop_walk, sample_type="rw")
