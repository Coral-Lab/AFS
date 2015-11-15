% specify significant digits in result
digits(4);
%K_FS_SEEDS = 0:1:1;
K_FS_SEEDS = -5:0.5:5;
K_FS_VALUES = vpa(10.^(-K_FS_SEEDS));

% number of rounds of AFS
numberOfRounds = 50;

% number of runs to average over
numberOfRuns = 25;

% number of random features added
numberOfRandomFeatures = 0;

% dataset to be used
datasetName = 'iris';

symbol_list = ['*' '^' 's'];
beta_test_list = [0.1 0.5 0.9];

figure();
hold on;

for beta_index = 1:length(beta_test_list)

    accuracyList = [];
    beta_test = beta_test_list(beta_index);

    parfor index = 1:length(K_FS_VALUES)
    
        % debug
        display(index);
    
        K_FS = K_FS_VALUES(index);
    
            % average over a certain number of runs
    
            totalAccuracy = 0;
            for run = 1:numberOfRuns
	        Accuracy_AFS = AFS_run(K_FS, numberOfRounds, numberOfRandomFeatures, datasetName, beta_test);
                totalAccuracy = totalAccuracy + Accuracy_AFS;
            end

            averageAccuracy = totalAccuracy / numberOfRuns;
            accuracyList = [accuracyList; averageAccuracy];     
    end

    plot(K_FS_SEEDS, accuracyList, strcat('-k',symbol_list(beta_index)));
end

xlabel('-10 log_{10}U_{FS}')
ylabel('Accuracy')
legend('Location','Best');
legend('0.1','0.5','0.9');
savefig(datasetName);
