function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

fake_theta = [0,theta(2:end)']';

for i = [1:m]
    hx = sigmoid(X(i,:)*theta);
    J = J + (-1/m) * (y(i,:)*log(hx) + (1-y(i,:))*log(1-hx));
    
    grad = grad + (1/m) * (hx - y(i,:)) * X(i,:)';
    
end

grad = grad + fake_theta*lambda/m;
J = J + lambda/(2*m) * sum(fake_theta.^2);





% =============================================================

end
