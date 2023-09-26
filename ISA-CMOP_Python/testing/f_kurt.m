function [kurt_avg, kurt_min, kurt_max, kurt_rnge] = f_kurt(objvar) 
%% Checks the kurtosis of the population of objective values - both the rank
    % and univariate avg/max/min/range
	%objvar = pop.objs; %decvar = pop.decs; consvar = pop.cons;
	rmimg = find(sum(imag(objvar)~= 0,2));
	objvar(rmimg,:) = [];
	%decvar(rmimg,:) = [];
    %consvar(rmimg,:) = [];
    [kurt_avg, kurt_min, kurt_max, kurt_rnge] = deal(NaN);
    if numel(objvar) > 0
        %ranksort = NDSort(objvar, length(objvar));
        %size(ranksort)
        %size(objvar)
        %kurt_rank = kurtosis(ranksort);
        kurt_avg = mean(kurtosis(objvar));
        kurt_max = max(kurtosis(objvar));
        kurt_min = min(kurtosis(objvar));
        kurt_rnge = kurt_max - kurt_min;
    end
end