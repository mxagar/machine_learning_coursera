function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


% Cost (without regularization)
h = X*theta; % (m x (n+1)) x ((n+1) x 1) -> m x 1
e = (h - y); % m x 1
J = (0.5/m) * (e'*e);
% Regularization term
t = theta(2:end,:); % n x 1
J = J + ((0.5*lambda/m) * (t'*t));

% Gradient (without regularization)
grad = (1/m) * (e'*X); % (1 x m) x (m x (n+1)) -> 1 x (n+1)
% Regularization term
r = (lambda/m) * theta(2:end,1); % n x 1
grad = grad' + [0; r];
    

% =========================================================================

grad = grad(:);

end
