function corr_obj = f_corr(objvar)
%F_CORR Summary of this function goes here
%   Detailed explanation goes here

[c,pvalue] = corrcoef(objvar(:,1),objvar(:,2));
corr_obj = c(1,2);
if pvalue(1,2) > 0.05 %there is no corr 
    corr_obj = 0;
end
corr_obj(isnan(corr_obj))=0; %NaN might appear when there is no change in one vector

end

