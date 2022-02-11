function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Sizes
m = size(X, 1);
num_labels = size(Theta2, 1);
% Initialize return variable: predictions
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Layer 1
a1 = [ones(size(X,1),1), X]; % 5000 x 401
% Layer 1 -> Layer 2
z2 = a1*Theta1'; % (5000 x 401) x (401 x 25) -> (5000 x 25)
a2 = sigmoid(z2);
a2 = [ones(size(a2,1),1), a2]; % bias -> (5000 x 26)
% Layer 2 -> Layer 3
z3 = a2*Theta2'; % (5000 x 26) x (26 x 10) -> (5000 x 10)
a3 = sigmoid(z3);
% Select class
[v, p] = max(a3,[],2); % maximum column-value (v) and column-index (p) for each row

% =========================================================================

end
