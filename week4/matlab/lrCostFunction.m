function [J, grad] = lrCostFunction(theta, X, y, lambda)

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

hypothesis = sigmoid(X * theta);
reg_param = (lambda/(2*m))* sum(theta(2:end).^2);
J =  (1/m) * sum((-y'*log(hypothesis)) - (1-y')*(log(1-hypothesis))) + reg_param;

grad = (1/m) * X' * (hypothesis-y);
grad_parameter = theta(2:size(grad)) * lambda / m;
grad(2:size(grad)) = grad(2:size(grad)) +  grad_parameter;

end
