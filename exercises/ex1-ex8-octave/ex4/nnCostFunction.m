function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



%%% Feedforward
% h: 5000 x 10

% Initialize return variable: predictions
h = zeros(size(X, 1), 1);
% Layer 1
a1 = [ones(size(X,1),1), X]; % 5000 x 401
% Layer 1 -> Layer 2
z2 = a1*Theta1'; % (5000 x 401) x (401 x 25) -> (5000 x 25)
a2 = sigmoid(z2);
a2 = [ones(size(a2,1),1), a2]; % bias -> (5000 x 26)
% Layer 2 -> Layer 3
z3 = a2*Theta2'; % (5000 x 26) x (26 x 10) -> (5000 x 10)
h = sigmoid(z3);

%%% Part 1: Cost

% Re-arrange y: one-hot encoding
yh = zeros(size(y, 1), num_labels); % 5000 x 10
for k = 1:num_labels
    yh(:,k) = y == k;
end

% Error: for each class, then sum
e = zeros(1, num_labels);
for k = 1:num_labels
    e(k) = -yh(:,k)'*log(h(:,k)) - (1 .- yh(:,k))'*log(1 .- h(:,k));
end
E = sum(e);
J = (1.0/m)*E;

%%% Part 2: Regularization of the Cost Function
t1 = Theta1(:,2:end)(:); % remove bias weight and unroll to column vector
t2 = Theta2(:,2:end)(:); % remove bias weight and unroll to column vector
R = (0.5*lambda/m) * (t1'*t1 + t2'*t2);
J = J + R;

%%% Part 3: Backpropagation & Gradient Computation
% We need to perform backpropagation in a for loop iterating examples
% Note that I use transposed vectors:
% in the script, column vectors are used,
% but I take example rows from the X matrix,
% and operate with them without converting them in column vectors
D1 = zeros(size(Theta1));
D2 = zeros(size(Theta2));
for i = 1:m
    % Feedfoward
    a1 = [1, X(i,:)]; % 1 x 401
    z2 = a1*Theta1'; % (1 x 401) x (401 x 25) -> (1 x 25)
    a2 = sigmoid(z2);
    a2 = [ones(size(a2,1),1), a2]; % bias -> (1 x 26)
    z3 = a2*Theta2'; % (1 x 26) x (26 x 10) -> (1 x 10)
    h = sigmoid(z3); % 1 x 10
    % Error backpropagation
    d3 = h - yh(i,:); % 1 x 10
    d2 = Theta2' * d3' .* [1, sigmoidGradient(z2)]'; % (26 x 10) x (10 x 1) .x (26, 1)-> 
    % d1: there is no such thing, because we have the inputs in layer 1
    d2 = d2(2:end)'; % remove the bias component, transpose: 1 x 25
    % Gradient computation: Accumulate gradients
    D1 = D1 + d2'*a1; % (25 x 1) x (1 x 401) -> (25 x 401)
    D2 = D2 + d3'*a2; % (10 x 1) x (1 x 26) -> (10 x 26) 
end
% Normalize
Theta1_grad = (1 / m) * D1;
Theta2_grad = (1 / m) * D2;

%%% Part 4: Regularization Component of the Gradient
R1 = Theta1; % (25 x 401)
R2 = Theta2; % (10 x 26)
R1(:,1) = zeros(size(Theta1, 1),1); % remove bias component: first column
R2(:,1) = zeros(size(Theta2, 1),1); % remove bias component: first column
Theta1_grad = Theta1_grad + (lambda / m)*R1;
Theta2_grad = Theta2_grad + (lambda / m)*R2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
