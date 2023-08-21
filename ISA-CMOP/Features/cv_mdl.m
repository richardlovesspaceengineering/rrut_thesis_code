function [mdl_r2, range_coeff] = cv_mdl(objvar, decvar)

	rmimg = find(sum(imag(objvar)~= 0,2));
	objvar(rmimg,:) = [];
	%decvar(rmimg,:) = [];
    
%------------------Linearity------------------
% Fit a linear model using constraint violation
    mdl = fitlm(decvar,objvar);
% R2
    mdl_r2 = mdl.Rsquared.Adjusted;
% Intercept
%intercept = mdl.Coefficients.Estimate(1);
% Difference between variable coefficient. Large values mean that
% there is at least one variable which is carrying most of the
% weight (given linearity)
    range_coeff = max(mdl.Coefficients.Estimate(2:end))-min(mdl.Coefficients.Estimate(2:end));
end

