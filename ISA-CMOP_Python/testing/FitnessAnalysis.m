function dat_out = FitnessAnalysis(Population, Instances, PF)
%FITNESSANALYSIS main function to calculate a problem features from a random sample
%(population), Instances is the problem name, PF is the known Pareto Front


	sample_size = size(Population.decs,1);
	%decsize = size(Population.decs,2);
	%probsize = size(Population.objs,2);

	Feasible = Population(find(all(Population.normcvs<=0,2)));
	NonDominated = Population(NDSort(Population.objs, Population.cons,1) == 1); %best front based on Deb's ranking, could be infeasible
	NonDominatedUncons = Population(NDSort(Population.objs,1) == 1);

	% propotion of solutions
	feature_fsr = FsR(Population, Feasible); %propotion of f_solutions
	feature_po_n = numel(NonDominated)/numel(Population);  %propotion of PF to pop
	feature_upo_n = numel(NonDominatedUncons)/numel(Population); %propotion of unconstrained PF to pop -new
	feature_cpo_upo_n = numel(NonDominated)/numel(NonDominatedUncons); %propotion of constrained PF to unconstrained PF --new
	feature_GD_cpo_upo = GD(NonDominated.objs,NonDominatedUncons.objs); % distance between PF and unconstrained PF --new
	feature_cover_cpo_upo = Coverage(NonDominatedUncons.objs,NonDominated.objs); %propotion of unconstrained PF coverd by PF --new


	[feature_uhv,temp] = HV(NonDominatedUncons.objs,PF); %HV of unconstrained PF -new
	[feature_hv,temp] = HV(NonDominated.objs,PF); %HV of constrained PF -new
	feature_hv_uhv_n = feature_hv/feature_uhv; %--new

	%Objectives correlation
	feature_corr_obj = f_corr(Population.objs);

	%constraint violation x fitness features
	[feature_corr_cobj,feature_corr_cf] = fvc(Population); %fitness and cons correlation
	feature_corr_cobj_min = min(feature_corr_cobj); %min of fitness and cons correlation
	feature_corr_cobj_max = max(feature_corr_cobj); %max of fitness and cons correlation
	[feature_piz_ob,feature_piz_f] = PiIZ(Population); %propotion of points in Ideal zone
	feature_piz_ob_min = min(feature_piz_ob); %min of propotion of points in Ideal zone
	feature_piz_ob_max = max(feature_piz_ob); %max of propotion of points in Ideal zone

	%distance features
	[feature_ps_dist_max, feature_ps_dist_mean, feature_ps_dist_iqr_mean] = f_decdist(NonDominated, 1, 1); % distance across PS
	[feature_pf_dist_max, feature_pf_dist_mean, feature_pf_dist_iqr_mean] = obdisc(NonDominated,1); %discontinuities in PF, distance across PF

	%distance-fitness correlation
	[feature_dist_c_corr] = dist_corr(Population, NonDominated); %--new

	%y-distrbution features: unconstrained Fitness density
	[feature_mean_f,feature_std_f,feature_min_f,feature_max_f,feature_skew_f,feature_kurt_f] = y_distFronts(Population.objs); %density of fronts
	[feature_kurt_avg, feature_kurt_min, feature_kurt_max, feature_kurt_rnge] = f_kurt(Population.objs); %density of objs
	[feature_skew_avg, feature_skew_min, feature_skew_max, feature_skew_rnge] = f_skew(Population.objs); %density of objs
	%y-distrbution features: constraint violation density
	[feature_mean_cv,feature_std_cv,feature_min_cv,feature_max_cv,feature_skew_cv,feature_kurt_cv] = cv_distr(Population.normcvs); %density of cons violation

	[feature_f_mdl_r2, feature_f_range_coeff] = rank_mdl(Population.objs, Population.decs); %meta-model features: reflect variable scaling
	[feature_pop_cv_mdl_r2, feature_pop_cv_range_coeff] = cv_mdl(Population.normcvs, Population.decs); %meta-model features


	%%%%%%%%%%%%%%

	dat_out = table(sample_size, ...
	    feature_fsr, feature_po_n, feature_upo_n, feature_cpo_upo_n, ...
	    feature_GD_cpo_upo, feature_cover_cpo_upo, feature_uhv, feature_hv, feature_hv_uhv_n, ...
	    feature_corr_obj, feature_corr_cobj_min,feature_corr_cobj_max, feature_corr_cf, ...
	    feature_piz_ob_min, feature_piz_ob_max, feature_piz_f, feature_ps_dist_max, feature_ps_dist_mean, feature_ps_dist_iqr_mean, ...
	    feature_pf_dist_max, feature_pf_dist_mean, feature_pf_dist_iqr_mean, ...
	    feature_dist_c_corr, ...
	    feature_mean_f,feature_std_f,feature_max_f,feature_skew_f,feature_kurt_f, ...
	    feature_kurt_avg, feature_kurt_min, feature_kurt_max, feature_kurt_rnge, ...
	    feature_skew_avg, feature_skew_min, feature_skew_max, feature_skew_rnge, ...
	    feature_mean_cv, feature_std_cv, feature_min_cv, feature_max_cv, feature_skew_cv, feature_kurt_cv, ...
	    feature_f_mdl_r2, feature_f_range_coeff, feature_pop_cv_mdl_r2, feature_pop_cv_range_coeff);
	    
	%disp(dat_out)
        
end

