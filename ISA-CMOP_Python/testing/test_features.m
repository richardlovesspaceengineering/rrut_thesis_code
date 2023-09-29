% Testing the NDSort Function

%% Preamble
clc;
clear;

%% Creating data structures.

% Global
pop_global.decs = readmatrix("global/dec.csv");
pop_global.objs = readmatrix("global/obj.csv");
pop_global.cons = readmatrix("global/cons.csv");
pop_global.normcvs = readmatrix("global/cv.csv");
pop_global.rank_cons_rrut = readmatrix("global/rank_cons.csv");
pop_global.rank_uncons_rrut = readmatrix("global/rank_uncons.csv");

% Get Pareto front.
PF = readmatrix("global/pf.csv");

%% Checking NDSort - can use normcvs or cons for the same result.

rank_cons_alsouly = NDSort(pop_global.objs, pop_global.normcvs, length(pop_global.objs(:,1)))';
rank_uncons_alsouly = NDSort(pop_global.objs, length(pop_global.objs(:,1)))';

% Check equal
cons_equal = all(rank_cons_alsouly == pop_global.rank_cons_rrut);
uncons_equal = all(rank_uncons_alsouly==pop_global.rank_uncons_rrut);

% Given these are equal, set the rank

%% Checking best rank
sort = rank_cons_alsouly;
bestrankobjs = readmatrix("global/bestobjs.csv");

bestrankobs_aslouly = pop_global.objs(sort == 1, :);

bestrank_equal = all(all(bestrankobs_aslouly == bestrankobjs));

%% Checking HV
% Value in Python is 0.27008098253228957
bhv_rrut = 0.27008098253228957;
bhv_alsouly = HV(bestrankobs_aslouly, PF);

bhv_equal = bhv_alsouly == bhv_rrut;

%% Checking f_mdl_r2
% Unadjusted R2 value in Python
f_mdl_r2_unadj_rrut = 0.10302631588601052;

% Adjusted R2 Value in Python
f_mdl_r2_adj_rrut = 0.0930154488758097;

[f_mdl_r2_alsouly, f_range_coeff_alsouly] = ...
    rank_mdl(pop_global.objs, pop_global.decs); %meta-model features: reflect variable scaling

f_mdl_r2_equal = round(f_mdl_r2_adj_rrut,6) == round(f_mdl_r2_alsouly,6);

%% Checking corr_obj
% Value in Python
corr_ob_rrut = -0.6434191514698974;

corr_obj_alsouly = f_corr(pop_global.objs);

corr_obj_equal = corr_ob_rrut == corr_obj_alsouly;

%% Checking corr_cf
corr_f_rrut = 0;

[corr_ob_alsouly, corr_f_alsouly] = fvc(pop_global);

corr_f_equal = corr_f_rrut == corr_f_alsouly;