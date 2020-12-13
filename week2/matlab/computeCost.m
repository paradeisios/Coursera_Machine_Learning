function J = computeCost(X, y, theta)

m = length(y);
sum_of_errors = (X*theta)-y;
J = (1/(2*m)) *sum((sum_of_errors.^2));

end
