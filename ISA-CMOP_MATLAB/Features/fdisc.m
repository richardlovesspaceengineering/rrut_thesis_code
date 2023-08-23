function [fdisc_max, fdisc_mean, fdisc_iqr_mean] = fdisc(pop)
%------------------Discontinuous feasibile area------------------
	objvar = pop.objs; decvar = pop.decs; consvar = pop.cons;
	%rmimg = find(sum(imag(objvar)~= 0,2));
	%objvar(rmimg,:) = [];
	%decvar(rmimg,:) = [];
    %consvar(rmimg,:) = [];
    [fdisc_max, fdisc_mean, fdisc_iqr_mean] =  deal(NaN);
    if numel(objvar) > 0
        dist = pdist2(decvar, decvar);
        fdisc_max = max(max(dist));
        fdisc_mean = mean(dist,'all');
        fdisc_iqr_mean = mean(iqr(dist)); % difference between the 75th and the 25th percentiles of distances
    end
    
end

