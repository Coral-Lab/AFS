% V
function V_w = V(w)

%% Constants
K_V = 1;                  % constant factor in multiplication

%% Function
% we use cost proportional to weights
V_w = K_V * w/norm(w,2);

end
% end of V