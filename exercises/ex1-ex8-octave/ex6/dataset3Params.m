function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% Possible C & sigma values
C_pool = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_pool = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
% Accuracy for each C-sigma combination
accuracy = zeros(1,size(C_pool,2)*size(sigma_pool,2));
max_accuracy = 0;
count = 1;
for i = 1:size(C_pool,2)
    for j = 1:size(sigma_pool,2)
        % Choose params
        c = C_pool(i);
        s = sigma_pool(j);
        % Train with training set
        model = svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, s));
        % Cross-Validation: accuracy?
        pred = svmPredict(model, Xval);
        acc = mean(double(pred == yval));
        accuracy(1,count) = acc;
        % Choose if best pair (best accuracy so far)
        if acc > max_accuracy
            max_accuracy = acc;
            C = c;
            sigma = s;
        end
        % Increase C-sigma combination counter
        count = count + 1;
    end
end

% =========================================================================

end
