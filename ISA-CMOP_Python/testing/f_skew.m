function [skew_avg, skew_min, skew_max, skew_rnge] = f_skew(objvar)
%% Checks the skewness of the population of objective values - both the rank
    % and univariate avg/max/min/range
	rmimg = find(sum(imag(objvar)~= 0,2));
	objvar(rmimg,:) = [];

    %ranksort = NDSort(objvar, length(objvar));
    [skew_avg, skew_min, skew_max, skew_rnge] = deal(NaN);
    if numel(objvar) > 0
        %skew_rank = skewness(ranksort);
        skew_avg = mean(skewness(objvar));
        skew_max = max(skewness(objvar));
        skew_min = min(skewness(objvar));
        skew_rnge = skew_max - skew_min;
    end
end

