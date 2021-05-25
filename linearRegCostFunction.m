function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
l=length(theta);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
grad=grad(:);
J=(sum((X*theta-y).^2)+(lambda*sum(theta(2:l).^2)))/(2*m);
t=theta;
t(1)=0;
for j=1:length(theta)
  grad(j)=(sum((X*theta-y).*X(:,j))+lambda*t(j))/m;
endfor