function cost = costFunction(X,y,theta)
% Watch out: select order depending on size/dimensions
% predictions = theta'*X;
predictions = X*theta; % column
cost = (1/(2*length(y)))*sum((predictions - y).^2);