function x = FsR(pop, fs)
%The feasibility ratio (FsR) simply approximates the size of the feasible
% space in relation to the overall search space.

 x = numel(fs.objs)/numel(pop.objs);
end

