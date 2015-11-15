% specify K_FS
K_FS = 1;

% number of rounds of AFS
numberOfRounds = 25;

% number of runs to average over
numberOfRuns = 100;

% number of random features added
numberOfRandomFeatureValues = 0:1:20;

timeList = [];
for index = 1:length(numberOfRandomFeatureValues)
    
    % debug
    index
        
    % number of random features
    numberOfRandomFeatures = numberOfRandomFeatureValues(index);
    
    % average over a certain number of runs
    totalTime = 0;
    for run = 1:numberOfRuns
        % test time of each run
        tic;
        
        Accuracy_AFS = AFS_run(K_FS, numberOfRounds, numberOfRandomFeatures);
        totalTime = totalTime + toc;
    end
    averageTime = totalTime / numberOfRuns;

    timeList = [timeList; averageTime];    
end

% add number of original features
plot(4+numberOfRandomFeatureValues, timeList);