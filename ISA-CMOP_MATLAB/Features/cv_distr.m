function [mean_f,std_f,min_f,max_f,skew_f,kurt_f] = cv_distr(objvar)
%Y-distribution of constaints violations
    rmimg = find(sum(imag(objvar)~= 0,2));
	objvar(rmimg,:) = [];
    
    mean_f = mean(objvar,'all');
    std_f = std(objvar);
    min_f = min(objvar);
    max_f = max(objvar);
    kurt_f = kurtosis(objvar);
    skew_f = skewness(objvar);
end

