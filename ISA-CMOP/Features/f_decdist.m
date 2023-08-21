function [PSdecdist_max, PSdecdist_mean, PSdecdist_iqr_mean] = f_decdist(pop, n1, n2)
    objvar = pop.objs; decvar = pop.decs; %consvar = pop.cons;
	rmimg = find(sum(imag(objvar)~= 0,2));
	objvar(rmimg,:) = [];
	%decvar(rmimg,:) = [];
    %consvar(rmimg,:) = [];
    [PSdecdist_max, PSdecdist_mean, PSdecdist_iqr_mean] = deal(0);
    if numel(objvar) > 1
        ranksort = NDSort(objvar, length(objvar));
        % Distance across and between n1 and n2 rank fronts in decision space
        dist = pdist2(decvar(ranksort == n1,:),decvar(ranksort == n2,:));
        PSdecdist_max = max(max(dist));
        PSdecdist_mean = mean(dist,'all');
        PSdecdist_iqr_mean = mean(iqr(dist)); % difference between the 75th and the 25th percentiles of distances
    end
end
