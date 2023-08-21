function dat_out = randomwalkfeatures(Populations, Instances, PF)
%RANDOMWALKFEATURES calculate all the features that could be generated from
%random walk samples
% Populations is matrix that represents a walk, each row is a solution(step) and its neighbours
% Instances is the problem name, PF is the known Pareto Front
    
    walkLength = size(Populations,1);
    neighbourhoodSize = size(Populations(1,:),1);
    %decsize = size(Populations(1,:).decs,2);
    %probsize = size(Populations(1,:).objs,2);

    j=0;
    for i=1:walkLength %for each step in the walk measure its neighbourhood features
        
        pop = Populations(i,:);
        fspop = pop(find(all(pop.normcvs<=0,2)));
        objvar = pop.objs;
        decvar = pop.decs;
        consvar = pop.cons;
        normcvsvar = pop.normcvs;
        
        [sort,k] = NDSort(objvar,consvar,length(objvar));

        % Proportion first solution is dominated by others
        sup(i) = 1/(length(objvar)-1)*sum(sort(2:end) > sort(1));
        % Proportion first solution is dominating others
        infi(i) = 1/(length(objvar)-1)*sum(sort(2:end) < sort(1));
        % Proportion of incomparable neighbours
        inc(i) = 1/(length(objvar)-1)*sum(sort(2:end) == sort(1));
        % Proportion of Non-dominated solutions in neighbourhood
        lnd(i) = 1/(length(objvar)-1)*sum(sort(2:end) == 1);
        
        %average distance from neighbours in the variable space
        distdec = pdist2(decvar,decvar);
        dist_x_avg(i) = mean(distdec,'all');
        %average distance from neighbours in the objective space
        distobj = pdist2(objvar,objvar);
        dist_f_avg(i) = mean(distobj,'all');
        %average distance from neighbours in the constraints space
        distcons = pdist2(normcvsvar,normcvsvar);
        dist_c_avg(i) = mean(distcons,'all');
        %average distance from neighbours in the objective-constraints space
        distobjscons = pdist2([objvar normcvsvar],[objvar normcvsvar]);
        dist_f_c_avg(i) = mean(distobjscons,'all');
        %ratio of the average distance from neighbours in the objective and variable spaces
        dist_f_dist_x_avg(i) = dist_f_avg(i)/dist_x_avg(i);
        %ratio of the average distance from neighbours in the constraints and variable spaces
        dist_c_dist_x_avg(i) = dist_c_avg(i)/dist_x_avg(i);
        %ratio of the average distance from neighbours in the objective-constraints and variable spaces
        dist_f_c_dist_x_avg(i) = dist_f_c_avg(i)/dist_x_avg(i);

        % added features %not used
        % Number of fronts
        nfronts(i)=k;
        % Feasibility ratio
        fsr(i) = numel(fspop.objs)/numel(pop.objs);
        % Fitness cons correlation
        [cob,cf] = fvc(pop);
        corrob(i,:)=cob; corrf(i)=cf;
        [zob,zf] = PiIZ(pop);
        pizob(i,:)=zob; pizf(i)=zf;
        % Hypervolume covered by feasible part of neighbourhood
        if (fsr(i)>0)
            [nhv(i),temp] = HV(fspop.objs,PF);
        else
            nhv(i) = 0;
        end
        % Hypervolume covered by unconstraind neighbourhood
        nuhv(i) = HV(objvar,PF);
        
        %first solution norm violation
        ncv(i) = normcvsvar(1,:);
        % neighbourhood avg norm violation
        nncv(i) = mean(normcvsvar);
        % 1st rank mean violation
        %bestrank = pop(find(all(sort == 1,2)));
        bestrankobjs = objvar(sort == 1,:);
        bestrankncvs = normcvsvar(sort == 1,:);
        bncv(i) = mean(bestrankncvs);
        % 1st rank mean fitness per obj (or HV?)
        [bhv(i),temp] = HV(bestrankobjs,PF);
        bfit(i,:) = mean(bestrankobjs,1);
        
        %Feasible Boundray Crossing 
        if i>1
            j=j+1;
            if (ncv(i)>0 && ncv(i-1)>0)
                cross(j)=0;
            else
                if (ncv(i)<=0 && ncv(i-1)<=0)
                    cross(j)=0;
                else
                    cross(j)=1;
                end
            end  
        end
        %
    end
    
    %avg of features across walk
    feature_sup_avg_rws = mean(sup);
    feature_inf_avg_rws = mean(infi);
    feature_inc_avg_rws = mean(inc);
    feature_lnd_avg_rws = mean(lnd);
    %hv_avg_rws = mean(hv);
    feature_nhv_avg_rws = mean(nhv);
    feature_nuhv_avg_rws = mean(nuhv);
    %hvd_avg_rws = mean(hvd);
    
    feature_dist_x_avg_rws = mean(dist_x_avg);
    feature_dist_f_avg_rws = mean(dist_f_avg);
    feature_dist_c_avg_rws = mean(dist_c_avg);
    feature_dist_f_c_avg_rws = mean(dist_f_c_avg);
    feature_dist_f_dist_x_avg_rws = mean(dist_f_dist_x_avg);
    feature_dist_c_dist_x_avg_rws = mean(dist_c_dist_x_avg);
    feature_dist_f_c_dist_x_avg_rws = mean(dist_f_c_dist_x_avg);
    
    feature_nfronts_avg_rws = mean(nfronts);
    feature_fsr_avg_rws = mean(fsr);
    feature_corrob_avg_rws = mean(corrob);
    feature_corrf_avg_rws = mean(corrf);
    feature_pizob_avg_rws = mean(pizob);
    feature_pizf_avg_rws = mean(pizf);
    feature_ncv_avg_rws = mean(ncv);
    feature_nncv_avg_rws = mean(nncv);
    feature_bncv_avg_rws = mean(bncv);
    feature_bhv_avg_rws = mean(bhv);
    feature_bfit_avg_rws = mean(bfit);
    
    
    %first autocorr: calculation one step ahead
    sup_acf = autocorr(sup,1);
    inf_acf = autocorr(infi,1);
    inc_acf = autocorr(inc,1);
    lnd_acf = autocorr(lnd,1);
    %hv_acf = autocorr(hv,1);
    nhv_acf = autocorr(nhv,1);
    %hvd_acf = autocorr(hvd,1);
    nuhv_acf = autocorr(nuhv,1);
    
    dist_x_acf = autocorr(dist_x_avg,1);
    dist_f_acf = autocorr(dist_f_avg,1);
    dist_c_acf = autocorr(dist_c_avg,1);
    dist_f_c_acf = autocorr(dist_f_c_avg,1);
    dist_f_dist_x_acf = autocorr(dist_f_dist_x_avg,1);
    dist_c_dist_x_acf = autocorr(dist_c_dist_x_avg,1);
    dist_f_c_dist_x_acf = autocorr(dist_f_c_dist_x_avg,1);
    
    nfronts_acf = autocorr(nfronts,1);
    fsr_acf = autocorr(fsr,1);
    for i=1:size(corrob,2)
        corrob_acf(i,:) = autocorr(corrob(:,i),1);
    end
    corrf_acf = autocorr(corrf,1);
    for i=1:size(pizob,2)
        pizob_acf(i,:) = autocorr(pizob(:,i),1);
    end
    pizf_acf = autocorr(pizf,1);
    mv_acf = autocorr(ncv,1);
    nmv_acf = autocorr(nncv,1);
    bmv_acf = autocorr(bncv,1);
    bhv_acf = autocorr(bhv,1);
    for i=1:size(bfit,2)
        bfit_acf(i,:) = autocorr(bfit(:,i),1);
    end
    %first autocorr
    feature_sup_r1_rws = sup_acf(2);
    feature_inf_r1_rws = inf_acf(2);
    feature_inc_r1_rws = inc_acf(2);
    feature_lnd_r1_rws = lnd_acf(2);
    %hv_r1_rws = hv_acf(2);
    feature_nhv_r1_rws = nhv_acf(2);
    %hvd_r1_rws = hvd_acf(2);
    feature_nuhv_r1_rws = nuhv_acf(2);
    
    feature_dist_x_r1_rws = dist_x_acf(2);
    feature_dist_f_r1_rws = dist_f_acf(2);
    feature_dist_c_r1_rws = dist_c_acf(2);
    feature_dist_f_c_r1_rws = dist_f_c_acf(2);
    feature_dist_f_dist_x_r1_rws = dist_f_dist_x_acf(2);
    feature_dist_c_dist_x_r1_rws = dist_c_dist_x_acf(2);
    feature_dist_f_c_dist_x_r1_rws = dist_f_c_dist_x_acf(2);
    
    feature_nfronts_r1_rws = nfronts_acf(2);
    feature_fsr_r1_rws = fsr_acf(2);
    for i=1:size(corrob,2)
        feature_corrob_r1_rws(i) = corrob_acf(i,2);
    end
    feature_corrf_r1_rws = corrf_acf(2);
    for i=1:size(pizob,2)
        feature_pizob_r1_rws(i) = pizob_acf(i,2);
    end
    feature_pizf_r1_rws = pizf_acf(2);
    feature_ncv_r1_rws = mv_acf(2);
    feature_nncv_r1_rws = nmv_acf(2);
    feature_bncv_r1_rws = bmv_acf(2);
    feature_bhv_r1_rws = bhv_acf(2);
    for i=1:size(bfit,2)
        feature_bfit_r1_rws(i) = bfit_acf(i,2);
    end
    
    
    feature_rfbx_rws = sum(cross)/walkLength; %ratio of feasible boundray crossing
    
    dat_out = table(walkLength, neighbourhoodSize, ...
        feature_sup_avg_rws, feature_inf_avg_rws, feature_inc_avg_rws, feature_lnd_avg_rws, ...
        feature_dist_x_avg_rws, feature_dist_f_avg_rws, feature_dist_c_avg_rws, feature_dist_f_c_avg_rws, ...
        feature_dist_f_dist_x_avg_rws, feature_dist_c_dist_x_avg_rws, feature_dist_f_c_dist_x_avg_rws, ...
        feature_nhv_avg_rws, feature_nuhv_avg_rws, feature_nfronts_avg_rws, ...
        feature_fsr_avg_rws, feature_corrob_avg_rws, feature_corrf_avg_rws, feature_pizob_avg_rws, feature_pizf_avg_rws, ...
        feature_ncv_avg_rws, feature_nncv_avg_rws, feature_bncv_avg_rws, feature_bhv_avg_rws, feature_bfit_avg_rws, ...
        feature_sup_r1_rws, feature_inf_r1_rws, feature_inc_r1_rws, feature_lnd_r1_rws, ...
        feature_dist_x_r1_rws, feature_dist_f_r1_rws, feature_dist_c_r1_rws, feature_dist_f_c_r1_rws, ...
        feature_dist_f_dist_x_r1_rws, feature_dist_c_dist_x_r1_rws, feature_dist_f_c_dist_x_r1_rws, ...
        feature_nhv_r1_rws, feature_nuhv_r1_rws, feature_nfronts_r1_rws, ...
        feature_fsr_r1_rws, feature_corrob_r1_rws, feature_corrf_r1_rws, feature_pizob_r1_rws, feature_pizf_r1_rws, ...
        feature_ncv_r1_rws, feature_nncv_r1_rws, feature_bncv_r1_rws, feature_bhv_r1_rws, feature_bfit_r1_rws, ...
        feature_rfbx_rws);
    
    
end

