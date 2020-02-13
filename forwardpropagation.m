function [h1,h2] = forwardpropagation(validX,validY,nn_params,length_nn_params_1,input_layer_size,hidden_layer_size_1,hidden_layer_size_2,output_layer_size)
%FORWARDPROPAGATION Summary of this function goes here
%   to predict values on a test set
%h is the predicted value
%accuracy is defined as the fraction of values which were predicted within
%5 percent of the ground value
nn_params_1=nn_params(1:length_nn_params_1);
nn_params_2=nn_params(length_nn_params_1 +1:end);

theta1_1 = reshape(nn_params_1(1:hidden_layer_size_1 * (input_layer_size + 1)), ...
                 hidden_layer_size_1, (input_layer_size + 1));
theta2_1 = reshape(nn_params_1((1 + (hidden_layer_size_1 * (input_layer_size + 1))): (hidden_layer_size_1 * (input_layer_size + 1))+(hidden_layer_size_2 * (hidden_layer_size_1 + 1))),...
                 hidden_layer_size_2,hidden_layer_size_1+1);
theta3_1 = reshape(nn_params_1((1 + (hidden_layer_size_1 * (input_layer_size + 1))+(hidden_layer_size_2 * (hidden_layer_size_1 + 1))):end), ...
                 output_layer_size, hidden_layer_size_2 + 1);


theta1_2 = reshape(nn_params_2(1:hidden_layer_size_1 * (input_layer_size + 1)), ...
                 hidden_layer_size_1, (input_layer_size + 1));

theta2_2 = reshape(nn_params_2((1 + (hidden_layer_size_1 * (input_layer_size + 1))): (hidden_layer_size_1 * (input_layer_size + 1))+(hidden_layer_size_2 * (hidden_layer_size_1 + 1))),...
                 hidden_layer_size_2,hidden_layer_size_1+1);
theta3_2 = reshape(nn_params_2((1 + (hidden_layer_size_1 * (input_layer_size + 1))+(hidden_layer_size_2 * (hidden_layer_size_1 + 1))):end), ...
                 output_layer_size, hidden_layer_size_2 + 1);

%Implementation of a 3 layer NN cost function. Using Mean Squared Error.
m=size(validX,1);
% a1=[ones(m,1) validX];
% z2=a1*theta1';
% a2=ReLU(z2);
% a2=[ones(size(a2,1),1) a2];
% z3_1=a2*theta2_1';
% a3_1=ReLU(z3_1);
% a3_1=[ones(size(a3_1,1),1) a3_1];
% z4_1=a3_1*theta3_1';
% a4_1=ReLU(z4_1);
% h1=a4_1;
% 
% z3_2=a2*theta2_2';
% a3_2=ReLU(z3_2);
% a3_2=[ones(size(a3_2,1),1) a3_2];
% z4_2=a3_2*theta3_2';
% a4_2=ReLU(z4_2);
% h2=a4_2;

a1=[ones(m,1) validX];

m=size(validX,1);
a1=[ones(m,1) validX];
z2_1=a1*theta1_1';
%a2_1=sigmoid(z2_1);
a2_1=z2_1;
a2_1=[ones(size(a2_1,1),1) a2_1];
z3_1=a2_1*theta2_1';
%a3_1=sigmoid(z3_1);
a3_1=z3_1;
a3_1=[ones(size(a3_1,1),1) a3_1];
z4_1=a3_1*theta3_1';
%a4_1=sigmoid(z4_1);
a4_1=z4_1;
h1=a4_1;



z2_2=a1*theta1_2';
%a2_2=sigmoid(z2_2);
a2_2=z2_2;
a2_2=[ones(size(a2_2,1),1) a2_2];
z3_2=a2_2*theta2_2';
%a3_2=sigmoid(z3_2);
a3_2=z3_2;
a3_2=[ones(size(a3_2,1),1) a3_2];
z4_2=a3_2*theta3_2';
%a4_2=sigmoid(z4_2);
a4_2=z4_2;
h2=a4_2;
end

