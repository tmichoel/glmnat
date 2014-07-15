function score = glmnat_logisticscore(X,Y,fit)
% GLMNAT_LOGISTICSCORE - Compute logistic regression cost function
% GLMNAT_LOGISTICSCORE computes the logistic regression cost function for a
% set of regression results. This function is primarily used to compare the
% output of different logistic regression algorithms.
%
% USAGE: score = glmnat_logisticscore(X,Y,alpha,beta,lambda);
%
% INPUT: X   - NxP matrix of predictor data (N samples, P predictors).
%        Y   - Nx1 vector of 0/1 response data.
%        fit - structure with fields
%              fit.a0     - 1xK vector of intercept parameters.
%              fit.beta   - PxK matrix of regression coefficients
%              fit.lambda - Kx1 vector of regularization parameters.
%
% OUTPUT: score - Kx1 vector of cost function values
%
% AUTHOR : Tom Michoel, The Roslin Institute
%          tom.michoel@roslin.ed.ac.uk
%          http://lab.michoel.info
% 
% REFERENCE: Michoel T. Natural coordinate descent algorithm for 
%  L1-penalised regression in generalised linear models. arXiv:1405.4225
%
% LICENSE: GNU GPL v2


X2 = [ones(size(Y)) zscore(X,1)]; % standardize X and add intercept
w = (X2'*Y)/length(Y); % overlap vector
beta = [fit.a0; fit.beta]; % merge intercept and regression coefficients

score = zeros(size(fit.lambda)); 
for k=1:length(score)
    score(k) = mean(log(1+exp(X2*beta(:,k)))) - w'*beta(:,k) ...
        + fit.lambda(k)*sum(abs(beta(2:end,k)));
end