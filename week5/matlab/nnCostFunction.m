function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

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
X  = [ones(m,1),X];
a2 = sigmoid(X*Theta1');

a2 = [ones(m,1),a2];
a3 = sigmoid(a2*Theta2');

y_dummy = zeros(num_labels, m); % 10*5000
for ii=1:m
  y_temp = zeros(10,1); 
  y_temp(y(ii)) = 1;
  y_dummy(:,ii) = y_temp;
end 

theta1_grad = sum(sum(Theta1(:,2:end).^2));
theta2_grad = sum(sum(Theta2(:,2:end).^2));

reg_param = (lambda / (2*m)) * (theta1_grad + theta2_grad);

J = (1/m) * sum(sum((-y_dummy' .* log(a3))-((1-y_dummy').*log(1-a3)))) + reg_param;





















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
