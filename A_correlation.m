% Algorithm 2: A
function [f_prime, Utility_A] = A_correlation(P_A, f, F, F_FS, U_A_old, V_w)

% check if all featues have not been selected

reducedFeatures = F;
reducedFeatures(F_FS) = []; 

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
    if(Q_correlation(f(F_OCC, :), OCC) < Utility_A)
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
