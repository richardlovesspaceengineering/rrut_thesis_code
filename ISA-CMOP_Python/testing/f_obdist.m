function x = f_obdist(objvar, n1, n2)
	%objvar = pop.objs; decvar = pop.decs; consvar = pop.cons;
	rmimg = find(sum(imag(objvar)~= 0,2));
	objvar(rmimg,:) = [];
	%decvar(rmimg,:) = [];
    %consvar(rmimg,:) = [];
    ranksort = NDSort(objvar, length(objvar));
    % Distance across and between n1-th and n2-th best rank front front in objective space
    x = mean(pdist2(objvar(ranksort == n1,:), objvar(ranksort == n2,:)), 'all');
end
