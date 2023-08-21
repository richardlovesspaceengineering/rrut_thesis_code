function LhcWalk(Global)
% <algorithm> <U>
% LHC-based Walk algorithm
%dimension=bounds.length. sNumbers = 1000. sSize=10%

    data_out = [];
    Instances = convertCharsToStrings(class(Global.problem));
    decsize = Global.D;
    probsize = Global.M;

    meta = [table(Instances, decsize, probsize)];

    bounds = [Global.lower; Global.upper];
    PF = Global.PF;

    % file to save features
    fileName = strcat('Data/your_path/',char(Instances),'_D',num2str(Global.D),'.csv');

    %sSize = 0.2; %step size
    n=(2*decsize)+1; % # ofneighbours 
    sNumber= floor((Global.D/n)*1000); %step numbers (total evaluations d*1000,some lit used fixed length)

    for r=1:30
        %generate walk
        %rng default % For reproducibility
        walkNorm = lhsdesign(sNumber,decsize);
        walk = bsxfun(@plus,bounds(1),bsxfun(@times,walkNorm,(bounds(2)-bounds(1))));

        %generate points for each step in the walk (neighbours)
        Population=[];
        for i=1:sNumber
            Populations(i,:)=INDIVIDUAL(GenerateNeighbours(bounds, walk(i,:), n));
            Population = [Population ; Populations(i,:)];
        end

        data_out = [data_out; meta, ...
            FitnessAnalysis(Population, Instances, PF), randomwalkfeatures(Populations, Instances, PF)];

    end

    writetable(data_out, fileName);
    Global.NotTermination(Population);

                
end
