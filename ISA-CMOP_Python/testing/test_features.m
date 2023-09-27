% Testing the NDSort Function

%% Preamble
clc;
clear;

%% Creating data structures.

% Global
pop_global.decs = readmatrix("global/dec.csv");
pop_global.objs = readmatrix("global/obj.csv");
pop_global.normcvs = readmatrix("global/cv.csv");
pop_global.rank_cons_rrut = readmatrix("global/rank_cons.csv");
pop_global.rank_uncons_rrut = readmatrix("global/rank_uncons.csv");

% Get Pareto front.
PF = readmatrix("global/pf.csv");

%% Checking NDSort

rank_cons_alsouly = NDSort(pop_global.objs, pop_global.normcvs, length(pop_global.objs(:,1)))';
rank_uncons_alsouly = NDSort(pop_global.objs, length(pop_global.objs(:,1)))';

% Check equal
cons_equal = all(rank_cons_alsouly == pop_global.rank_cons_rrut);
uncons_equal = all(rank_uncons_alsouly==pop_global.rank_uncons_rrut);

% Given these are equal, set the rank

%% Pass into FitnessAnalysis
dat_out = FitnessAnalysis(pop_global, PF);
