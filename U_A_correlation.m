% UFS
function Utility_A = U_A_correlation(V_projected, V_OCC)

%% Constants
K_A = 1;                  % constant factor in multiplication

%% Function
% we simple use Euclidean distance for now
difference = V_projected - V_OCC;
Utility_A = K_A * difference/norm(difference,2);

end
% end of U_A