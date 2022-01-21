function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % Cost derivatives: dJ/dt
    p = X*theta; % m x 1
    d = (p-y);
    dJ_0 = (1.0/m)*d'*X(:,1);
    dJ_1 = (1.0/m)*d'*X(:,2);
    % Update theta
    t0 = theta(1,1) - alpha*dJ_0;
    theta(1,1) = t0;
    t1 = theta(2,1) - alpha*dJ_1;
    theta(2,1) = t1;    

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
