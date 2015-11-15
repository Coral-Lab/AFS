% Algorithm 1: FindOCC
function [OCC, F_OCC, V_projected] = FindOCC_correlation(P_A, f, F, F_FS, U_A_old, V_w)

%% Constants
K_beta = 1;                  % constant factor in multiplication

% calculate feature selection pool
F_pool = F;
F_pool(F_FS) = [];

if(length(F_pool) == 1)
    % there is only one element present
    F_good = F_pool(1);
    F_bad = F_pool(1);
elseif(length(F_pool) > 1)
    % calculate associated probabilities
    P_A_pool = P_A(F_pool);
    P_A_pool = P_A_pool / sum(P_A_pool);

    % select good from pool
    P_A_good = P_A_pool;
    probability = rand;
    cumulative_probability = 0;
    F_good = -1;

    for index = 1:length(P_A_good)
        cumulative_probability = cumulative_probability + P_A_good(index);

        if(cumulative_probability > probability)
            F_good = F(index);
            break;
        end    
    end

    % select bad from pool
    P_A_bad = 1 - P_A_pool;
    P_A_bad = P_A_bad / sum(P_A_bad);
    probability = rand;
    cumulative_probability = 0;
    F_bad = -1;

    for index = 1:length(P_A_bad)
        cumulative_probability = cumulative_probability + P_A_bad(index);

        if(cumulative_probability > probability)
            F_bad = F(index);
            break;
        end    
    end
end

% we will modify the vector based on confidence
beta = K_beta * U_A_old;

% calculate the updated vector and cost
f_bad = f(F_bad, :);
f_good = f(F_good, :);
V_bad = V_w(F_bad);
V_good = V_w(F_good);
f_bad_prime = f_bad + beta*(f_good - f_bad);
V_projected = V_bad + beta*(V_good - V_bad);

% assign result values
OCC = f_bad_prime;
F_OCC = F_bad;

end
% end of Algorithm 1
