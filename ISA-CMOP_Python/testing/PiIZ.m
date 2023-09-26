function [piz_ob,piz_f] = PiIZ(pop)
%proportion in ideal zone (PiIZ)  quantifies the proportion of points in the
%lower left quadrant of the fitness-violation scatterplot for each obj and
%for unconstraind fronts-violation scatterplot

%calculate ideal point of cv and fitness: min+(10%or25% * (maximum-minimum))
%then count points in the range

	objvar = pop.objs; mconsvar = pop.normcvs; %consvar = pop.cons;
	rmimg = find(sum(imag(objvar)~= 0,2));
	objvar(rmimg,:) = [];
	%mconsvar(rmimg,:) = [];
	%consvar(rmimg,:) = [];

	minobjs = min(objvar); maxobjs = max(objvar);
	minmcons = min(mconsvar); maxmcons = max(mconsvar);
	mconsIdealPoint = minmcons+(0.25*(maxmcons-minmcons));
	conZone = find(all(pop.normcvs<=mconsIdealPoint,2));

	piz_ob = zeros(1,width(objvar));

	for i=1:width(objvar) %PiZ for each objXcon
	    objIdealPoint = minobjs(i)+(0.25*(maxobjs(i)-minobjs(i)));
	    objx = objvar(:,i);
	    iz = find(all(objx(conZone)<=objIdealPoint ,2));
	    piz_ob(i) = numel(iz)/numel(pop.objs);
	end

	ranksort = (NDSort(objvar, length(objvar)))';
	minrank = min(ranksort);  maxrank = max(ranksort);
	rankIdealPoint = minrank+(0.25*(maxrank-minrank));
	iz = find(all(ranksort(conZone)<=rankIdealPoint ,2));
	piz_f = numel(iz)/numel(pop.objs); %PiZ for frontsXcon


end

