function x = f_rankprop(pop,n)
	objvar = pop.objs; consvar = pop.cons;
	rmimg = find(sum(imag(objvar)~= 0,2));
	objvar(rmimg,:) = [];
    %consvar(rmimg,:) = [];
    %------------------Evolvability------------------
    % Proportion of rank n
    rankn = objvar(NDSort(objvar, consvar, length(objvar))==n,:);
    x = numel(rankn)/numel(objvar);
    if numel(objvar) == 0
        x=0;
    end
end
