% Algorithm 2: A
function [f_prime, Utility_A] = A_noise(P_A, f, F, F_FS, U_A_old, V_w)

%% Constants
K_noise = 1;                  % constant factor in multiplication

% calculate feature selection pool
F_pool = F_FS;

% calculate associated probabilities
P_A_pool = P_A(F_pool);
P_A_pool = P_A_pool / sum(P_A_pool);

% select from pool
probability = rand;
cumulative_probability = 0;
F_noise = -1;

for index = 1:length(P_A_good)
    cumulative_probability = cumulative_probability + P_A_pool(index);

    if(cumulative_probability > probability)
        F_noise = F(index);
        break;
    end    
end

% add some noise to the selected feature vector
dataSize = length(f(1, :));
f_prime = f;

percentage_change = 

for index = 1:dataSize
    f_prime(F_noise,index) = -f_prime(F_noise,index) + (2*rand)*f_prime(F_noise,index);
end





if(isempty(reducedFeatures))
    % no action can be taken
    % at least one feature must exist in pool to take action
    Utility_A = 0;
    f_prime = f;
else
    % calculate OCC
    [OCC, F_OCC, V_projected] = FindOCC_correlation(P_A, f, F, F_FS, U_A_old, V_w);

    % calculate projected utility
    V_OCC = V_w(F_OCC);
    Utility_A = U_A_correlation(V_projected, V_OCC);

    % modify the dataset if it might be worth it
    if(Q_correlation(f(:,F_OCC), OCC) < Utility_A)
        % integrate OCC to f
        f_prime = f;
        dataSize = length(f(1, :));
        for index = 1:dataSize
            f_prime(F_OCC,index) = OCC(index);
        end
    else
        f_prime = f;
    end
end

end
% end of Algorithm 2
