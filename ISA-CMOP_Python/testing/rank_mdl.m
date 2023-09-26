function [mdl_r2, range_coeff] = rank_mdl(objvar, decvar)
	%objvar = pop.objs; decvar = pop.decs; consvar = pop.cons;
	rmimg = find(sum(imag(objvar)~= 0,2));
	objvar(rmimg,:) = [];
	%decvar(rmimg,:) = [];
    %consvar(rmimg,:) = [];
    [mdl_r2, range_coeff] = deal(NaN);
    if numel(objvar) > 2
    %------------------Linearity------------------
        ranksort = NDSort(objvar, length(objvar));
    % Fit a linear model using the normalised decision objectives
        mdl = fitlm(decvar,ranksort);
    % R2
        mdl_r2 = mdl.Rsquared.Adjusted;
    % Intercept
    %intercept = mdl.Coefficients.Estimate(1);
    % Difference between variable coefficient. Large values mean that
    % there is at least one variable which is carrying most of the
    % weight (given linearity)
        range_coeff = max(mdl.Coefficients.Estimate(2:end))-min(mdl.Coefficients.Estimate(2:end));
    end
end