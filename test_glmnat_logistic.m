%% GLMNAT_LOGISTIC usage examples
%
%% Load test data, choose one
% 
%% 
% TCGA-BRCA data
load data/TCGA_BRCA_expression_estrogen.mat
%% 
% TCGA-COAD data
% load data/TCGA_COAD_expression_stage.mat

%%  GLMNAT_LOGISTIC default
%
%%
% Run natural coordinate descent with default regularization parameter
% sequence and convergence threshold
fitnat = glmnat_logistic(X,Y);
%%
% plot regression coefficients
plot(fitnat.beta');

%% GLMNAT_LOGISTIC with optional parameters
%
%%
% Calculate geometric regularization parameter sequence (this one is same as
% default)
X1 = zscore(X,1); % standardize X
w = (X1'*Y)/length(Y);
nlambda = 100;
lambdaratio = 1e-2;
lambdamax = max(w);
nseq = (nlambda:-1:1)';
lambda = lambdamax * lambdaratio.^((nlambda-nseq)/(nlambda-1));
%%
% run natural coordinate descent with optional regularization parameters
fitnat = glmnat_logistic(X,Y,lambda);
%%
% run natural coordinate descent with optional regularization parameters
% and convergence threshold
epsconv = 1e-3;
fitnat = glmnat_logistic(X,Y,lambda);

%% Compare results to GLMNET
%
%%
% Run glmnet with same settings
X1 = zscore(X,1); % standardize X
opts.alpha = 1;
opts.lambda = fitnat.lambda;
fitnet = glmnet(X1,Y,'binomial',opts);
%%
% Compute logistic regression cost function values and compare
score_nat = glmnat_logisticscore(X,Y,fitnat);
score_net = glmnat_logisticscore(X,Y,fitnet);
max(abs(score_nat-score_net)./score_nat)