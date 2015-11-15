% UFS
function Utility_FS = U_FS(S, T, F_FS, K_FS)

%% Constants
%K_FS = 0.0000001;                  % constant factor in multiplication

%% Function

%%%%%%%%%%%%%%%%%%%%%%% LOGO PARAMETERS CAN BE PASSED IN AS AN ARGUMET
%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%%
%% logo_parammeters for Logo
logo_param.plotfigure = 0;     % 1: plot of the result of each iteration; 0: do not plot
logo_param.distance = 'block'; % 'euclidean';  
logo_param.sigma= 2;           % kernel width; If the algorithm does not converge, use a larger kernel width.
logo_param.lambda = 1;         % regularization logo_parammeter
% We arbitarily set sigma= 2 and lambda = 1. The proposed algorithm is not sensitive to logo_parammeters. 
% The algorithm can used for classification. The logo_parammeters can be learning via cross-validation (see the paper).

% extract data subset and calculate associated weights
featureSize = length(S(:,1)) - 1;
f_train = S(1:featureSize,:);
y_train = S(featureSize+1,:);
f_train_sub = f_train(F_FS, :);
classes = unique(y_train);

if(length(classes) == 2)
    % two classes
    w_sub = Logo(f_train_sub, y_train, logo_param);          
elseif(length(classes) == 3)
    % three classes
    % we will accumulate weights generated here
    w_sub_oneversusall = [];

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
        w_current_one = Logo(f_train_sub, y_train_modified, logo_param);
        w_sub_oneversusall = [w_sub_oneversusall; w_current_one];    
    end
end

%% classification
f_test = T(1:featureSize,:);
y_test = T(featureSize+1,:);
f_test_sub = f_test(F_FS, :);

if(length(classes) == 2)
    % two classes
    weights = w_sub;
    index_1 = find(y_train==classes(1));N(1) = length(index_1);
    index_2 = find(y_train==classes(2));N(2) = length(index_2);
    patterns_1 = f_train_sub(:,index_1);
    patterns_2 = f_train_sub(:,index_2);

    % taking probability which is maximum
    for n = 1:size(f_test_sub,2)
        test = f_test_sub(:,n);
        products = [];
        
        temp = abs(patterns_1-test*ones(1,N(1)));        
        dist_1    = weights*temp;    
        prob_1 = exp(-dist_1/logo_param.sigma);prob_1 = prob_1/sum(prob_1);
        products = [products;sum(dist_1.*prob_1)];

        temp = abs(patterns_2-test*ones(1,N(2)));
        dist_2    = weights*temp;
        prob_2 = exp(-dist_2/logo_param.sigma);prob_2 = prob_2/sum(prob_2);
        products = [products;sum(dist_2.*prob_2)];

        class_indices = find(products == min(products));
        winning_index = datasample(class_indices, 1);
        Prediction(n) = classes(winning_index);
    end
elseif(length(classes) == 3)
    % three classes
    weights_oneversusall = w_sub_oneversusall;
    index_1 = find(y_train==classes(1));N(1) = length(index_1);
    index_2 = find(y_train==classes(2));N(2) = length(index_2);
    index_3 = find(y_train==classes(3));N(3) = length(index_3);
    patterns_1 = f_train_sub(:,index_1);
    patterns_2 = f_train_sub(:,index_2);
    patterns_3 = f_train_sub(:,index_3);

    % taking probability which is maximum
    for n = 1:size(f_test_sub,2)
        test = f_test_sub(:,n);
        products = [];

        temp = abs(patterns_1-test*ones(1,N(1)));        
        dist_1    = (weights_oneversusall(1,:))*temp;
        prob_1 = exp(-dist_1/logo_param.sigma);prob_1 = prob_1/sum(prob_1);
        products = [products;sum(dist_1.*prob_1)];

        temp = abs(patterns_2-test*ones(1,N(2)));
        dist_2    = (weights_oneversusall(2,:))*temp;
        prob_2 = exp(-dist_2/logo_param.sigma);prob_2 = prob_2/sum(prob_2);
        products = [products;sum(dist_2.*prob_2)];

        temp = abs(patterns_3-test*ones(1,N(3)));
        dist_3    = (weights_oneversusall(3,:))*temp;
        prob_3 = exp(-dist_3/logo_param.sigma);prob_3 = prob_3/sum(prob_3);
        products = [products;sum(dist_3.*prob_3)];

        class_indices = find(products == min(products));
        winning_index = datasample(class_indices, 1);
        Prediction(n) = classes(winning_index);
    end
end
    
% calculate utility based on accuracy
test_Error = length(find(Prediction(:)~=y_test(:)))/length(y_test);
Utility_FS = K_FS*(1 - test_Error);
end
% end of U_FS