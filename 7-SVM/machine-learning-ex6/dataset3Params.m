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

sample_val = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
% error_vec = zeros(length(sample_val)^2, 3);
error_vec = [0 0 0];
for c = 1:length(sample_val)
    for s = 1:length(sample_val)
        pred = svmPredict(svmTrain(X,y,sample_val(c),@(x1, x2) gaussianKernel(x1, x2, sample_val(s))), Xval);
        error = mean(double(pred ~= yval));
        error_vec = [error_vec; sample_val(c) sample_val(s) error];
    end
end
[val i] = min(error_vec(2:end,3));
C = error_vec(i+1,1);
sigma = error_vec(i+1,2);

% =========================================================================

end
