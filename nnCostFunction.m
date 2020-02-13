function [J ,grad] = nnCostFunction(nn_params,len,input_layer_size,hidden_layer_size_1,hidden_layer_size_2,output_layer_size, X,Y , lambda)

nn_params_1=nn_params(1:len);
nn_params_2=nn_params(len+1:end);

theta1_1 = reshape(nn_params_1(1:hidden_layer_size_1 * (input_layer_size + 1)), ...
                 hidden_layer_size_1, (input_layer_size + 1));
theta2_1 = reshape(nn_params_1((1 + (hidden_layer_size_1 * (input_layer_size + 1))): (hidden_layer_size_1 * (input_layer_size + 1))+(hidden_layer_size_2 * (hidden_layer_size_1 + 1))),...
                 hidden_layer_size_2,hidden_layer_size_1+1);
theta3_1 = reshape(nn_params_1((1 + (hidden_layer_size_1 * (input_layer_size + 1))+(hidden_layer_size_2 * (hidden_layer_size_1 + 1))):end), ...
                 output_layer_size, hidden_layer_size_2 + 1);


%nn_params_2=[theta1(:); nn_params(length_nn_params_1 +1:end)];
theta1_2 = reshape(nn_params_2(1:hidden_layer_size_1 * (input_layer_size + 1)), ...
                 hidden_layer_size_1, (input_layer_size + 1));
theta2_2 = reshape(nn_params_2((1 + (hidden_layer_size_1 * (input_layer_size + 1))): (hidden_layer_size_1 * (input_layer_size + 1))+(hidden_layer_size_2 * (hidden_layer_size_1 + 1))),...
                 hidden_layer_size_2,hidden_layer_size_1+1);
theta3_2 = reshape(nn_params_2((1 + (hidden_layer_size_1 * (input_layer_size + 1))+(hidden_layer_size_2 * (hidden_layer_size_1 + 1))):end), ...
                 output_layer_size, hidden_layer_size_2 + 1);

%Implementation of a 3 layer NN cost function. Using Mean Squared Error.
m=size(X,1);
a1=[ones(m,1) X];
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

%z3_2=a2*theta2_2';
%a3_2=ReLU(z3_2);
%a3_2=[ones(size(a3_2,1),1) a3_2];
%z4_2=a3_2*theta3_2';
%a4_2=ReLU(z4_2);
%h2=a4_2;

% calculte penalty
p = sum(theta1_1(:,2:end).^2,'all')+sum(theta2_1(:,2:end).^2,'all')+sum(theta3_1(:,2:end).^2,'all')+sum(theta1_2(:,2:end).^2,'all')+sum(theta2_2(:,2:end).^2,'all')+sum(theta3_2(:,2:end).^2,'all');
%calculate J
J = (sum((h1-Y(:,1)).^2)/(2*m)) +  (sum((h2-Y(:,2)).^2)/(2*m)) + (lambda*p/(2*m));
% disp(sum(p,all));
% disp(sum(J,all));

%%%%%%%%%%%%gradient cost%%%%%%%

%function g = ReLUGradient(z)
%g=z>0;
%end
%% PATH 1: CURRENT CALCULATION
sigma4_1= a4_1 - Y(:,1);
sigma3_1 = (sigma4_1*theta3_1);%.*sigmoidGradient([ones(size(z3_1, 1), 1) z3_1]);
sigma3_1 = sigma3_1(:,2:end);
sigma2_1 = (sigma3_1*theta2_1);%.*sigmoidGradient([ones(size(z2_1, 1), 1) z2_1]);
sigma2_1 = sigma2_1(:, 2:end);

% accumulate gradients
delta_1_1 = (sigma2_1'*a1);
delta_2_1 = (sigma3_1'*a2_1);
delta_3_1 = (sigma4_1'*a3_1);

% calculate regularized gradient
p1_1 = (lambda/m)*[zeros(size(theta1_1, 1), 1) theta1_1(:, 2:end)];
p2_1 = (lambda/m)*[zeros(size(theta2_1, 1), 1) theta2_1(:, 2:end)];
p3_1 = (lambda/m)*[zeros(size(theta3_1, 1), 1) theta3_1(:, 2:end)];

theta1_1_grad = delta_1_1./m + p1_1;
theta2_1_grad = delta_2_1./m + p2_1;
theta3_1_grad=  delta_3_1./m + p3_1;

%% PATH 2: VOLTAGE CALCULATION
sigma4_2= a4_2 - Y(:,2);
sigma3_2 = (sigma4_2*theta3_2);%.*sigmoidGradient([ones(size(z3_2, 1), 1) z3_2]);
sigma3_2 = sigma3_2(:,2:end);
sigma2_2 = (sigma3_2*theta2_2);%.*sigmoidGradient([ones(size(z2_2, 1), 1) z2_2]);
sigma2_2 = sigma2_2(:, 2:end);

% accumulate gradients
delta_1_2 = (sigma2_2'*a1);
delta_2_2 = (sigma3_2'*a2_2);
delta_3_2 = (sigma4_2'*a3_2);

% calculate regularized gradient
p1_2 = (lambda/m)*[zeros(size(theta1_2, 1), 1) theta1_2(:, 2:end)];
p2_2 = (lambda/m)*[zeros(size(theta2_2, 1), 1) theta2_2(:, 2:end)];
p3_2 = (lambda/m)*[zeros(size(theta3_2, 1), 1) theta3_2(:, 2:end)];

theta1_2_grad = delta_1_2./m + p1_2;
theta2_2_grad = delta_2_2./m + p2_2;
theta3_2_grad=  delta_3_2./m + p3_2;

%================unroll gradients
grad = [theta1_1_grad(:) ; theta2_1_grad(:) ; theta3_1_grad(:) ; theta1_2_grad(:); theta2_2_grad(:) ; theta3_2_grad(:) ];
end