function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
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


a1 = [ones(m, 1), X]; % 构造矩阵（包括extra bias unit which is 1） [5000, 401]

z2 = Theta1 * a1'; % 计算第一层网络的加权求和，构造出一个 [25, 5000]的矩阵

% Hidden Layer
a2 = sigmoid(z2);  % 计算第一层网络的阀门值，构造出一个 [25, 5000]的矩阵

a2 = [ones(1, size(a2, 2)); a2]; % 加入extra bias unit，构造出[26, 5000]的矩阵

% Output layer
z3 = Theta2 * a2; % 计算第二层的加权值，构造 [10, 5000]的矩阵

a3 = sigmoid(z3); % 计算输出的阀门值，是一个[10, 5000]的矩阵

% calculating max on the transpose of a3 so the index 
% result, p, has the expected dimensions [5000, 1]
 [Val,p] = max(a3', [], 2);


% =========================================================================


end
