% An implementation of backpropagation for a MLP (Multi-Layer Perceptron)
% structure using the Iris dataset.

clear all;
close all;

%% load 
load('IRIS_sample234')
X = irisInputs_Train_234;
label = irisTargets_Train;
x_test = irisInputs_Test234;
label_test = irisTargets_Test;
%% hyper-parameters
lr = 0.005;
N_epoches = 3000;

%% looping over different activation functions & normalization methods
% nm = ["none","np1","p1"];
% act = ["sigmoid", "tanh", "relu", "lrelu"];
% for i = 1:4
%     for a = 1:2
%         for b=1:2
%             error01 = run(X,label,x_test, label_test, lr, N_epoches, nm(k),act(a),act(b), n ); n = n+1;
%         end
%     end
% end

%% single combination
[error01, pred] = run(X,label,x_test, label_test, lr, N_epoches, "np1", "lrelu", "sigmoid", "lrelu-sigmoid");

%% training and testing in a function
function [error01, y_hat] = run(X,label,x_test, y_test, lr, N_epoches, nm, act1, act2, figName)
%% number of neurons
ni = 3; nh = 3; no = 3;

%% weight initialization
W1 = rand(nh,ni)*2-1;
% b1 = rand(nh,1)*2-1;
b1 = zeros(nh,1);
W2 = rand(no,nh)*2-1;
% b2 = rand(no,1)*2-1;
b2 = zeros(no,1);

%% normalization of original dataset
X_np = normalize(X, nm);

%% training
LossList = zeros(1,N_epoches);
for epoch=1:N_epoches
    seq = randperm(90); %shuffled
%     seq = linspace(1,90,90); %naive
    X_np_seq = X_np(:,seq);
    Y_seq = label(:,seq);
    L = 0;
    for i=1:1:90
        %% Stochastic Gradient Descent
        x = X_np_seq(:,i);
        y = Y_seq(:,i);
        
        %% forward propagation:
        [y_hat, z1, z2, a] = forwardprop(x, W1, W2, b1, b2, act1, act2);
    
        %% Loss function
        % L2-loss
        Loss = 0.5*sum((y-y_hat).^2); 

        %% update with backprop
        delta2 = -(y-y_hat).* d_activation(z2, act2);
        delta1 = (W2.'*delta2) .* d_activation(z1, act1);
        
        dLdW2 = delta2 * a.';
        dLdW1 = delta1 * x.';
        
        W2 = W2 - lr * dLdW2;
        b2 = b2 - lr * delta2;
        W1 = W1 - lr * dLdW1;
        b1 = b1 - lr * delta1;
        L = L + Loss;
    end
    %disp(L)
    LossList(epoch) = L;
end

%% testing
x_test = normalize(x_test, nm);
[y_hat, ~, ~, ~] = forwardprop(x_test, W1, W2, b1, b2, act1, act2);
[~, argmax_p] = max(y_hat);
[~, argmax_y] = max(y_test);
error01 = sum(argmax_p ~= argmax_y) / length(y_test);
% disp("error rate:")
disp(error01)

%% plot
figure
plot(LossList)
title("Loss (nm="+nm+", act=["+act1+","+act2+"]), err="+string(error01))
xlabel("epoches")
ylabel("Loss");
%% save figure and then close figure
% saveas(gcf,string(figName)+'.jpg'); close all;
end

%% functions
function [y_hat, z1, z2, a] = forwardprop(x, W1, W2, b1, b2, act1, act2)
% 1st layer
z1 = W1*x + b1; 
a = activation(z1, act1);
% 2nd layer
z2 = W2*a + b2;
y_hat = activation(z2, act2);
end

function X_n = normalize(x, mode)
    X = x-min(x.').';
    X = X ./ max(X.').';
    if mode == "none"
        X_n = x;
    elseif mode == "np1"
        X_n = X*2-1;
    elseif mode == "p1"
        X_n = X;
    end
end

function y = activation(a, mode)
    if mode == "sigmoid"
        y = 1./(1+exp(-a));
    elseif mode == "tanh"
        y = tanh(a);
    elseif mode == "relu"
        a(a<0)=0;
        y = a;
    elseif mode == "lrelu"
        a(a<0)=a(a<0)*0.1;
        y = a;
    else 
        error('please insert a proper activation function name')
    end
end

function y = d_activation(a, mode)
    if mode == 'sigmoid'
        y = ( 1./(1+exp(-a))).*(1- 1./(1+exp(-a)));
    elseif mode == 'tanh'
        y = 1./(cosh(a).^2);
    elseif mode == 'relu'
        a(a>0)=1;
        a(a<=0)=0;
        y = a;
    elseif mode == 'lrelu'
        a(a>0)=1;
        a(a<=0)=0.1;
        y = a;
    else 
        error('please insert a proper activation function name')
    end
end