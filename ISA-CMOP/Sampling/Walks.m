function Walks(Global)
% <algorithm> <U>
% Generate multiple walks by using one of the walk's techniques
    
data_out = [];
Instances = convertCharsToStrings(class(Global.problem));
decsize = Global.D;
probsize = Global.M;

meta = [table(Instances, decsize, probsize)];

bounds = [Global.lower; Global.upper];
PF = Global.PF;

fileName = strcat(char(Instances),'_D',num2str(Global.D),'.csv');

sSize = 0.2; %step size
n=(2*decsize)+1; % # of neighbours
sNumber= floor((Global.D/n)*1000); %step numbers (total evaluations d*1000,some lit used fixed length)
%clusters = zeros(n,length(bounds),sNumber);

% use already produced sample file (to save time)
prob_dir = fullfile(pwd,'Data/Sample10D'); 
probfiles = dir(fullfile(prob_dir,'*.mat'));
nprobfiles = length(probfiles);

for r=1:30
    %generate walk from the initial sample file
    %%walk = RandomWalk(bounds, sNumber, sSize); % this is the function that has been used to generate the sample
    fid = probfiles(r).name;
    sample_file = string(fid);
    load(sample_file);
    %size(walk)
    Population=[];
    %generate neighbourhood for each step in the walk
    for i=1:sNumber
        cluster=result{1,2}(i,:).decs;
        Populations(i,:)=INDIVIDUAL(cluster);
        Population = [Population ; Populations(i,:)];
        %size(Populations(i,:))
    end
    
    %Population = INDIVIDUAL(walk);
    
    data_out = [data_out; meta, ...
        FitnessAnalysis(Population, Instances, PF), randomwalkfeatures(Populations, Instances, PF)];
end
writetable(data_out, fileName);
Global.NotTermination(Population);

end

