% Algorithm 3: FS
function [F_FS, P_FS] = FS(P_FS, S, T, F_FS, V_w, K_FS)

% Constants
K_P = 1;
epsilon = 0.01;

%
% select a feature to add
%

% calculate feature selection pool
featureSize = length(S(:,1)) - 1;
F_pool = [1:featureSize];
F_pool(F_FS) = [];
P_FS_pool = P_FS(F_pool);
P_FS_pool = P_FS_pool / sum(P_FS_pool);

%
% check if pool is empty
% if so, try to add a feature
%

if(length(F_pool) > 0)
    % select from pool
    select_probability = rand;
    cumulative_probability = 0;
    F_i = -1;

    for index = 1:length(P_FS_pool)
        cumulative_probability = cumulative_probability + P_FS_pool(index);

        if(cumulative_probability > select_probability)
            F_i = F_pool(index);
            break;
        end    
    end
    
    % calculate utility 
    Utility_FS_current = U_FS(S, T, F_FS, K_FS);

    % add this feature to F_FS and sort for safety

    F_FS_new = [F_FS; F_i];
    F_FS_new  = sort(F_FS_new);

    % calculate new utility and change in utility
    Utility_FS_new = U_FS(S, T, F_FS_new, K_FS);
    delta_U_FS = Utility_FS_new - Utility_FS_current;

    % evaluate new feature and promote selection if found to be useful

    if(V_w(F_i) < delta_U_FS)
        P_FS(F_i) = P_FS(F_i) + K_P * abs(delta_U_FS);
        F_FS = F_FS_new;
    else
        P_FS(F_i) = P_FS(F_i) - K_P * abs(delta_U_FS);
    end
end

% reward features which are currently selected
P_epsilon = zeros(1, length(P_FS));
for index = 1:length(F_FS)
    P_epsilon(F_FS(index)) = epsilon;
end
P_FS = P_FS + P_epsilon;

% normalize probabilities
P_FS = P_FS / norm(P_FS, 2);

%
% select a feature to remove
%

% calculate feature reject pool
F_reject_pool = F_FS;
P_FS_pool = P_FS(F_FS);
P_FS_pool = P_FS_pool / sum(P_FS_pool);
P_FS_reject_pool = 1 - P_FS_pool;

reject_probability = rand;
reject_feature_indices = [];        % these are indices relative to F_FS and not actual feature indices in F

% add features to the pool based on rejection probability
for index = 1:P_FS_reject_pool    
    if(reject_probability < P_FS_reject_pool(index))
        reject_feature_indices = [reject_feature_indices; index];
    end    
end

% now select one of these based on their reject probabilities
if(isempty(reject_feature_indices) && isempty(F_FS))
    % remove a feature
    reject_probabilities = P_FS_reject_pool(reject_feature_indices);
    reject_probabilities = reject_probabilities / norm(reject_probabilities, 2);
    reject_probability = rand;
    cumulative_probability = 0;

    for index = 1:reject_probabilities
        cumulative_probability = cumulative_probability + reject_probabilities(index);
        
        if(cumulative_probability > reject_probability)
            F_reject_index = reject_feature_indices(index);
            break;
        end    
    end
    
    % remove this index from F_FS
    F_FS(F_reject_index) = [];
end

%DEBUG
%F_FS

end
% end of Algorithm 3
