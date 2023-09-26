% Testing the NDSort Function
clc;
clear;
obj = readmatrix("test_obj.csv");
cv = readmatrix("test_cv.csv");
rank_cons_rrut = readmatrix("test_rank.csv");
rank_uncons_rrut = readmatrix("test_rank_uncons.csv");

rank_cons_alsouly = NDSort(obj, cv, length(obj(:,1)))';
rank_uncons_alsouly = NDSort(obj, length(obj(:,1)))';

% Check equal
cons_equal = all(rank_cons_alsouly == rank_cons_rrut);
uncons_equal = all(rank_uncons_alsouly==rank_uncons_rrut);
