function [corr_ob,corr_f] = fvc(pop)
%the correlation between the objective function values and norm violation
%values are calculated using Spearman's rank correlation coefficent.
objvar = pop.objs; mconsvar = pop.normcvs;
rmimg = find(sum(imag(objvar)~= 0,2));
objvar(rmimg,:) = [];
%mconsvar(rmimg,:) = [];

corr_ob = zeros(1,width(objvar));

for i=1:width(objvar)
    objx = objvar(:,i);
    [c,pvalue] = corrcoef(mconsvar,objx);
    corr_ob(i) = c(1,2);
    if pvalue(1,2) > 0.05 %there is no corr 
        corr_ob(i) = 0;
    end
end
corr_ob(isnan(corr_ob))=0; %NaN might appear when there is no change in one vector

%corr between cons and origional (unconstrained) fronts
ranksort = NDSort(objvar, length(objvar));
[c,pvalue] = corrcoef(mconsvar,ranksort);
corr_f = c(1,2);
if pvalue(1,2) > 0.05 %there is no corr 
    corr_f = 0;
end
corr_f(isnan(corr_f))=0;

end

