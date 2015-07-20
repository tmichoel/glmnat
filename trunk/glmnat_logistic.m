function fit = glmnat_logistic(varargin)
% GLMNAT_LOGISTIC - Natural coordinate descent algorithm for logistic regression
% GLMNAT_LOGISTIC runs the natural coordinate descent algorithm for
% logistic regression (see reference below). 
%
% INSTALLATION : This function is a wrapper for the C-MEX function
%   GLMNAT_LOGISTIC_SRC.C which needs to be compiled first. Compilation
%   instructions are provided in the header of this C-file.
% 
% USAGE: fit = glmnat_logistic(X,Y);
%        fit = glmnat_logistic(X,Y,lambda);
%        fit = glmnat_logistic(X,Y,lambda,epsconv);
%
% INPUT: X       - NxP matrix of predictor data (N samples, P predictors).
%        Y       - Nx1 vector of 0/1 response data.
%        lambda  - Sequence of regularization parameters, sort from
%                  high to low for fastest results. Optional parameter, default
%                  is to use a geometric series of length 100.
%        epsconv - Convergence threshold. Optional parameter, default value
%                  1E-3.
%
% OUTPUT: fit         - Structure with results:
%         fit.a0      - Intercept parameter.
%         fit.beta    - Regression coefficients for predictor variables
%         fit.lambda  - Sequence of regularization parameters
%         fit.epsconv - Convergence threshold
%
% AUTHOR : Tom Michoel, The Roslin Institute
%          tom.michoel@roslin.ed.ac.uk
%          http://lab.michoel.info
% 
% REFERENCE: Michoel T. Natural coordinate descent algorithm for 
%  L1-penalised regression in generalised linear models. arXiv:1405.4225
%
% LICENSE: GNU GPL v2

X = varargin{1}; % predictor data
Y = varargin{2}; % response data

% sanity checks on the data
if numel(Y)~=length(Y)
    error('Response data Y is not a vector.')
end
if length(unique(Y))~=2
    error('Response data is not binary.')
end
if sum(unique(Y)==[0;1])~=2
    [~,~,Y] = unique(Y);
    Y = Y-1;
    warning('Response data classes have been converted to 0/1 values.') 
end
if size(X,1)~=size(Y,1)
    error('Number of samples in predictor X and response Y are not identical.')
end

% standardize X and add column of 1s for intercept
X2 = [ones(size(Y)) zscore(X,1)];

% overlap vector
w = (X2'*Y)/length(Y);

% default geometric regularization parameter sequence
nlambda = 100;
lambdaratio = 1e-2;
lambdamax = max(w(2:end));
nseq = (nlambda:-1:1)';
lambda = lambdamax * lambdaratio.^((nlambda-nseq)/(nlambda-1));

% default convergence threshold
epsconv = 1e-3;

% override regularization parameter and convergence threshold if given
switch nargin
    case 3
        lambda = varargin{3}; % sequence of penalty parameters
    case 4
        lambda = varargin{3}; % sequence of penalty parameters
        epsconv = varargin{4}; % convergence parameter
end

% run logistic regression
beta = glmnat_logistic_src(X2,w,lambda,epsconv);

% return output structure
fit.a0      = beta(1,:);     % intercept
fit.beta    = beta(2:end,:); % regression coefficients
fit.lambda  = lambda;        % regularization parameter sequence
fit.epsconv = epsconv;       % convergence threshold
