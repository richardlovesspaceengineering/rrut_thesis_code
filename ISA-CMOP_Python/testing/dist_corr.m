function [dist_c_corr] = dist_corr(pop, NonDominated)
%distance correlation
%distance for each solution to nearest global solution in the decsion space
%correlation of distance and constraints norm. 

    objvar = pop.objs; decvar = pop.decs; consvar = pop.cons;
	rmimg = find(sum(imag(objvar)~= 0,2));
	objvar(rmimg,:) = [];
	%decvar(rmimg,:) = [];
    consvar(rmimg,:) = [];
    
    dist = pdist2(NonDominated.decs,decvar,'euclidean','Smallest',1);
    [c1,pvalue] = corrcoef(pop.normcvs,dist.');
    dist_c_corr = c1(1,2);
    if pvalue(1,2) > 0.05 %there is no corr 
        dist_c_corr = 0;
    end
    
end

