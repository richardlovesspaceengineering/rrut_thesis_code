function [mean_f,std_f,min_f,max_f,skew_f,kurt_f] = y_distFronts(objvar)
%Y-distribution of fronts
    rmimg = find(sum(imag(objvar)~= 0,2));
	objvar(rmimg,:) = [];
    [mean_f,std_f,min_f,max_f,skew_f,kurt_f] = deal(NaN);
    if numel(objvar) > 0
        ranksort = NDSort(objvar, length(objvar));
        mean_f = mean(ranksort,'all');
        std_f = std(ranksort);
        min_f = min(ranksort);
        max_f = max(ranksort);
        %rank_dist = zeros(1,max_f);
        %for i=1:max_f
        %    rank_dist(i)= numel(objvar(ranksort==i,:));
        %end
        %[pks,locs] = findpeaks(rank_dist);
        %n_peaks_f = numel(pks);
        kurt_f = kurtosis(ranksort);
        skew_f = skewness(ranksort);
    end
end

