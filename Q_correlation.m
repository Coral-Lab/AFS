% Q
function cost = Q_correlation(f_i, f_i_prime)

%% Constants
K_Q = 1;                  % constant factor in multiplication

%% Function
% we simply use Euclidean distance for now
cost = K_Q * norm(f_i - f_i_prime);

end
% end of Q