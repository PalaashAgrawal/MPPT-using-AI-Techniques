clear;
clc;
%data values :
%1. Time stamp	2.Weather file ambient temperature | (C)	3.Weather file wind speed | (m/s)	4.Irradiance DHI from weather file | (W/m2)
%5. Irradiance DNI from weather file | (W/m2)	6. Irradiance GHI from weather file | (W/m2)  7.Subarray 1 Surface azimuth | (deg)	
%8.Subarray 1 Surface tilt | (deg)	9.Sun altitude angle | (deg)	10.Sun azimuth angle | (deg)	11.Array DC power | (kW)	
%12. Inverter MPPT 1 Nominal DC voltage | (V)	13. Subarray 1 Open circuit DC voltage | (V)	14.Subarray 1 Operating DC voltage | (V)	
%15.Subarray 1 Cell temperature | (C)	16.Subarray 1 Module efficiency | (%)	17.Subarray 1 Short circuit DC current | (A)
%% NN architecture 5x10x(4+4)x1
input_layer_size=15;
hidden_layer_size_1=50;
hidden_layer_size_2=15;
output_layer_size=1;
%% Loading Data
fprintf('Loading Data ...\n')
load('dataaug.mat');
%rand_data=rand_data{:,:};
% load('X.mat');
% load('Y.mat');
% load('valid_X.mat')
% load('valid_Y.mat');
m=size(Xaug,1);
%%
% X=table2array(X);
% Y=table2array(Y);
% validX=table2array(validX);
% validY=table2array(validY);
%% PREPROCESSING THE DATA
%Converting power to operating current I=P/V
for i=1:m
    if (Y(i,2)~=0)
        Y(i,1)=Y(i,1)/Y(i,2);
    end
end


nfX=max(Xaug)-min(Xaug);
nfY=max(Y)-min(Y);
% normalizing the inputs
Xaug= Xaug./nfX;
Y=Y./nfY;

%% initialising weight matrices
fprintf('\nInitializing weight Matrices ... \n')
epsilon_init=  1  ;

theta1_1  =rand(hidden_layer_size_1,input_layer_size + 1)*2*epsilon_init - epsilon_init;
theta1_2   =rand(hidden_layer_size_1,input_layer_size + 1)*2*epsilon_init - epsilon_init;
theta2_1=rand(hidden_layer_size_2,hidden_layer_size_1 +1)*2*epsilon_init - epsilon_init;
theta2_2=rand(hidden_layer_size_2,hidden_layer_size_1 +1)*2*epsilon_init - epsilon_init;
theta3_1=rand(output_layer_size,hidden_layer_size_2 + 1)*2*epsilon_init - epsilon_init;
theta3_2=rand(output_layer_size,hidden_layer_size_2 + 1)*2*epsilon_init - epsilon_init;

initial_nn_params_1=[theta1_1(:);theta2_1(:);theta3_1(:)];
initial_nn_params_2=[theta1_2(:);theta2_2(:);theta3_2(:)];

initial_nn_params=[initial_nn_params_1;initial_nn_params_2];
disp(size(initial_nn_params));

%% Check cost function with regularisation
fprintf('\nChecking Cost Function (w/ Regularization) ... \n')
lambda=0.1;
J = nnCostFunction(initial_nn_params,length(initial_nn_params_1),input_layer_size,hidden_layer_size_1,hidden_layer_size_2,output_layer_size, Xaug(1:400000,:),Y(1:400000,:), lambda);
fprintf('\n cost and initial (random) parameters for Current Calculation : %f',J);
%% Training NN
fprintf('\nTraining Neural Network... \n')
options = optimset('MaxIter', 350 );
%options = optimset('GradObj', 'on', 'MaxIter', 300);
lambda=0.1;
costFunction =@(p)nnCostFunction(p,length(initial_nn_params_1),input_layer_size,hidden_layer_size_1,hidden_layer_size_2,output_layer_size, Xaug(1:400000,:),Y(1:400000,:), lambda);
%[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
%[nn_params, cost, exitFlag] = fminunc(costFunction, initial_nn_params, options);

%lambda=0.01
% %[nn_params, cost] = fmincg(costFunction, nn_params, options);
% 
% lambda=0.001
% [nn_params, cost] = fmincg(costFunction, nn_params, options);

nn_params_1=nn_params(1:length(initial_nn_params_1));
nn_params_2=nn_params(length(initial_nn_params_1)+1:end);

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

fprintf("\nTraining complete. Press Enter to continue.\n");
pause();
             %% ======DETERMINING VALUES ON A TEST SET========
   
[predict1, predict2]=forwardpropagation(Xaug(400001:end,:),Y(400001:end,:),nn_params,length(initial_nn_params_1),input_layer_size,hidden_layer_size_1,hidden_layer_size_2,output_layer_size);
predict1=predict1.*nfY(:,1);
predict2=predict2.*nfY(:,2);
validY=Y(400001:end,:).*nfY;
accuracy=[predict1 validY(:,1) predict2 validY(:,2)];
disp(accuracy);

%score=
