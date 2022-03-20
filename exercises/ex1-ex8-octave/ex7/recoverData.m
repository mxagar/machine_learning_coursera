function X_rec = recoverData(Z, U, K)
%RECOVERDATA Recovers an approximation of the original data when using the 
%projected data
%   X_rec = RECOVERDATA(Z, U, K) recovers an approximation the 
%   original data that has been reduced to K dimensions. It returns the
%   approximate reconstruction in X_rec.
%

% You need to return the following variables correctly.
X_rec = zeros(size(Z, 1), size(U, 1));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the approximation of the data by projecting back
%               onto the original space using the top K eigenvectors in U.
%
%               For the i-th example Z(i,:), the (approximate)
%               recovered data for dimension j is given as follows:
%                    v = Z(i, :)';
%                    recovered_j = v' * U(j, 1:K)';
%
%               Notice that U(j, 1:K) is a row vector.
%               

% Since we are undoing the mapping
% I would have expected that we need to invert the rotation
% Since U*U' = I, U' = inv(U)
% Then, we would have taken again the first K columns
UT = U'; % n x n
%X_rec = UT(:, 1:K)*Z'; % n x m
% BUT: The logic above does not work
X_rec = U(:, 1:K)*Z'; % n x m
X_rec = X_rec'; % m x n


% =============================================================

end
