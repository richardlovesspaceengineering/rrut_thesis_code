function [obdisc_max, obdisc_mean, obdisc_iqr_mean] = obdisc(pop,n)
%------------------Discontinuous front------------------
	objvar = pop.objs; %decvar = pop.decs; consvar = pop.cons;
	rmimg = find(sum(imag(objvar)~= 0,2));
	objvar(rmimg,:) = [];
	%decvar(rmimg,:) = [];
    %consvar(rmimg,:) = [];
% Large deviances between the rank n front might suggest discontinuities in
% PF
    [obdisc_max, obdisc_mean, obdisc_iqr_mean] = deal(NaN);
    if numel(objvar) > 0
        rankn = objvar(NDSort(objvar, length(objvar))==n,:);
        dist = pdist2(rankn, rankn);
        obdisc_max = max(max(dist));
        obdisc_mean = mean(dist,'all');
        obdisc_iqr_mean = mean(iqr(dist));
    end
end