% Algorithm 2: A
function Accuracy_AFS = AFS_run(K_FS, numberOfRounds, numberOfRandomFeatures, datasetName, beta_test)

%Constants
delta = 0.01;
%K_FS = 0.0001;
%numberOfRounds = 10;

%% initialize logo
%% logo_parammeters for Logo
logo_param.plotfigure = 0;     % 1: plot of the result of each iteration; 0: do not plot
logo_param.distance = 'block'; % 'euclidean';  
logo_param.sigma= 2;           % kernel width; If the algorithm does not converge, use a larger kernel width.
logo_param.lambda = 1;         % regularization logo_parammeter
% We arbitarily set sigma= 2 and lambda = 1. The proposed algorithm is not sensitive to logo_parammeters. 
% The algorithm can used for classification. The logo_parammeters can be learning via cross-validation (see the paper).

%% load data

if(strcmp(datasetName,'banana') == 1)
    eval(['load banana_train_data.asc'])
    eval(['load banana_train_labels.asc'])
    eval(['load banana_test_data.asc'])
    eval(['load banana_test_labels.asc'])

    eval(['train_patterns = banana_train_data;']);
    train_patterns = train_patterns'; % Each column is a pattern.
    eval(['train_targets = banana_train_labels;']);
    eval(['test_patterns = banana_test_data;']);
    test_patterns = test_patterns';
    eval(['test_targets = banana_test_labels;']);
elseif(strcmp(datasetName,'data_banknote_authentication') == 1)
    eval(['load data_banknote_authentication_train_data.asc'])
    eval(['load data_banknote_authentication_train_labels.asc'])
    eval(['load data_banknote_authentication_test_data.asc'])
    eval(['load data_banknote_authentication_test_labels.asc'])

    eval(['train_patterns = data_banknote_authentication_train_data;']);
    train_patterns = train_patterns'; % Each column is a pattern.
    eval(['train_targets = data_banknote_authentication_train_labels;']);
    eval(['test_patterns = data_banknote_authentication_test_data;']);
    test_patterns = test_patterns';
    eval(['test_targets = data_banknote_authentication_test_labels;']);

    % Logo needs non zero labels
    train_targets = train_targets + 1;
    test_targets = test_targets + 1;    
elseif(strcmp(datasetName,'diabetes') == 1)
    eval(['load diabetes_train_data.asc'])
    eval(['load diabetes_train_labels.asc'])
    eval(['load diabetes_test_data.asc'])
    eval(['load diabetes_test_labels.asc'])

    eval(['train_patterns = diabetes_train_data;']);
    train_patterns = train_patterns'; % Each column is a pattern.
    eval(['train_targets = diabetes_train_labels;']);
    eval(['test_patterns = diabetes_test_data;']);
    test_patterns = test_patterns';
    eval(['test_targets = diabetes_test_labels;']);
    
    % Logo needs non zero labels
    train_targets = train_targets + 1;
    test_targets = test_targets + 1;    
elseif(strcmp(datasetName,'heart') == 1)
    eval(['load heart_train_data.asc'])
    eval(['load heart_train_labels.asc'])
    eval(['load heart_test_data.asc'])
    eval(['load heart_test_labels.asc'])

    eval(['train_patterns = heart_train_data;']);
    train_patterns = train_patterns'; % Each column is a pattern.
    eval(['train_targets = heart_train_labels;']);
    eval(['test_patterns = heart_test_data;']);
    test_patterns = test_patterns';
    eval(['test_targets = heart_test_labels;']);
elseif(strcmp(datasetName,'twonorm') == 1)
    eval(['load twonorm_train_data.asc'])
    eval(['load twonorm_train_labels.asc'])
    eval(['load twonorm_test_data.asc'])
    eval(['load twonorm_test_labels.asc'])

    eval(['train_patterns = twonorm_train_data;']);
    train_patterns = train_patterns'; % Each column is a pattern.
    eval(['train_targets = twonorm_train_labels;']);
    eval(['test_patterns = twonorm_test_data;']);
    test_patterns = test_patterns';
    eval(['test_targets = twonorm_test_labels;']);

    % Logo needs non zero labels
    train_targets = train_targets + 1;
    test_targets = test_targets + 1;    
elseif(strcmp(datasetName,'iris') == 1)
    eval(['load iris_train_data.asc'])
    eval(['load iris_train_labels.asc'])
    eval(['load iris_test_data.asc'])
    eval(['load iris_test_labels.asc'])

    eval(['train_patterns = iris_train_data;']);
    train_patterns = train_patterns'; % Each column is a pattern.
    eval(['train_targets = iris_train_labels;']);
    eval(['test_patterns = iris_test_data;']);
    test_patterns = test_patterns';
    eval(['test_targets = iris_test_labels;']);
elseif(strcmp(datasetName,'waveform') == 1)
    eval(['load waveform_train_data.asc'])
    eval(['load waveform_train_labels.asc'])
    eval(['load waveform_test_data.asc'])
    eval(['load waveform_test_labels.asc'])

    eval(['train_patterns = waveform_train_data;']);
    train_patterns = train_patterns'; % Each column is a pattern.
    eval(['train_targets = waveform_train_labels;']);
    eval(['test_patterns = waveform_test_data;']);
    test_patterns = test_patterns';
    eval(['test_targets = waveform_test_labels;']);
end


%% load complete
N = length(train_targets);                              % Number of patterns
Original_dim = size(train_patterns,1);                  % Number of original features
dim = size(train_patterns,1);                           % Data dimenionality

% add random features
train_length = length(train_targets);
train_patterns = [train_patterns; randn(numberOfRandomFeatures, train_length)];  
test_length = length(test_targets);
test_patterns = [test_patterns; randn(numberOfRandomFeatures, test_length)];  

%Preprocess the data: 'unif' tranform each feature into [0, 1]
[MIN,I] = min(train_patterns,[],2);
[MAX,I] = max(train_patterns,[],2);  
for n=1:dim
    train_patterns(n,:) = (train_patterns(n,:)-MIN(n))/(MAX(n)-MIN(n));
    test_patterns(n,:) = (test_patterns(n,:)-MIN(n))/(MAX(n)-MIN(n));
end

S = [train_patterns; train_targets'];
T = [test_patterns; test_targets'];

%% calculate associated weights

% extract data subset and calculate associated weights
featureSize = length(S(:,1)) - 1;
f_train = S(1:featureSize,:);
y_train = S(featureSize+1,:);
classes = unique(y_train);

if(length(classes) == 2)
    % two classes
    w = Logo(f_train, y_train, logo_param);          
elseif(length(classes) == 3)
    % three classes
    % we will accumulate weights generated here
    w_oneversusall = [];

    for index = 1:length(classes)        
        % current "one" in one versus all approach
        current_one = classes(index);

        % modify the targets accordingly    
        y_train_modified = y_train;    
        for index = 1:length(y_train_modified)
            if(y_train_modified(index) ~= current_one)
                y_train_modified(index) = 2;
            else
                y_train_modified(index) = 1;
            end
        end

        % now run logo on the modified set
        w_current_one = Logo(f_train, y_train_modified, logo_param);
        w_oneversusall = [w_oneversusall; w_current_one];    
    end
end

%% try to implement AFS

% initialize the game
P_FS = ones(1,size(train_patterns,1));      % uniform probabilities
P_FS = P_FS / sum(P_FS);
F_FS = [randi(featureSize)];                % starts with one randomly selected feature

if(length(classes) == 2)                    % initialize cost
    % two classes
    V_w = V(w);
elseif(length(classes) == 3)
    % three classes
    w_avg = sum(w_oneversusall) / size(w_oneversusall, 1);
    V_w = V(w_avg);
end

% modify T by correlating every non best feature with best feature

F_best_list = find(V_w == (max(V_w)));
F_best = F_best_list(1);

% correlation percentage
% modify based on beta_test

for F_index = 1:length(featureSize) 
    % modify if non best feature
    if(F_index ~= F_best)
        test_patterns(:,F_index) = test_patterns(:,F_index) + beta_test * (test_patterns(:,F_best) - test_patterns(:,F_index));
    end
end

T = [test_patterns; test_targets'];

featureInclusionCounts = zeros(1,size(train_patterns,1));   % count number of times a feature has been in F_FS
U_A_old_correlation = 1;                                    % in the first iteration, update f_bad to f_good
U_A_old_noise = 1;                                          % in the first iteration, add maximum noise

% simulate the game in turns
%numberOfRounds = 10;

for round = 1:numberOfRounds
    %
    % feature selector turn
    %
    [F_FS, P_FS] = FS(P_FS, S, T, F_FS, V_w, K_FS);
    
    %
    % correlation adversary turn    
    %
    % update feature inclusion counts and probabilities
    for index = 1:length(featureInclusionCounts)
        % if that feature is included in F_FS, increase count
        if(sum(F_FS == index) > 0)
            featureInclusionCounts(index) = featureInclusionCounts(index) + 1;
        else
            featureInclusionCounts(index) = featureInclusionCounts(index) + delta;
        end
    end
    
    P_A_correlation = featureInclusionCounts / sum(featureInclusionCounts);
    f = S(1:featureSize,:);
    F = [1:length(P_A_correlation)];
    [f_prime, Utility_A_correlation] = A_correlation(P_A_correlation, f, F, F_FS, U_A_old_correlation, V_w);
    U_A_old_correlation = Utility_A_correlation;
    
    % adversary modifies the dataset
    f = f_prime;
    
    %
    % noise adversary turn
    %
    %P_A_noise = P_A_correlation;
    %[f_prime, Utility_A_noise] = A_noise(P_A_noise, f, F, F_FS, U_A_old_noise, V_w);
    %U_A_old_noise = Utility_A_noise;

end

%
% evaluate generated feature set
%

Accuracy_AFS = U_FS(S, T, F_FS, K_FS) / K_FS;

end
% end of Algorithm
