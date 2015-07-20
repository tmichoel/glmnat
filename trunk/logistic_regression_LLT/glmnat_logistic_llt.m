function fit = glmnat_logistic_llt(varargin)
% GLMNAT_LOGISTIC_LLT - Natural coordinate descent algorithm for logistic regression using the linear-time legendre transform
% GLMNAT_LOGISTIC_LLT runs the exact natural coordinate descent algorithm
% for logistic regression [1] using the linear-time Legendre transform
% function [2].
%
% USAGE: fit = glmnat_logistic(X,Y);
%        fit = glmnat_logistic(X,Y,mu);
%        fit = glmnat_logistic(X,Y,mu,epsconv);
%
% INPUT: X       - NxP matrix of predictor data (N samples, P predictors).
%        Y       - Nx1 vector of 0/1 response data.
%        mu  - Sequence of regularization parameters, sort from
%                  high to low for fastest results. Optional parameter, default
%                  is to use a geometric series of length 100.
%        epsconv - Convergence threshold. Optional parameter, default value
%                  1E-3.
%
% OUTPUT: fit         - Structure with results:
%         fit.a0      - Intercept parameter.
%         fit.beta    - Regression coefficients for predictor variables
%         fit.mu  - Sequence of regularization parameters
%         fit.epsconv - Convergence threshold
%
% AUTHOR : Tom Michoel, The Roslin Institute
%          tom.michoel@roslin.ed.ac.uk
%          http://lab.michoel.info
% 
% REFERENCES: 
%   [1] Michoel T. Natural coordinate descent algorithm for L1-penalised
%       regression in generalised linear models. arXiv:1405.4225 
%   [2] Lucet Y. Faster than the fast Legendre transform, the
%       linear-time Legendre transform. Numerical Algorithms, 16(2), 171-185
%       (1997).  
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
nmu = 100;
muratio = 1e-2;
mumax = max(w(2:end));
nseq = (nmu:-1:1)';
mu = mumax * muratio.^((nmu-nseq)/(nmu-1));

% default convergence threshold
epsconv = 1e-3;

% override regularization parameter and convergence threshold if given
switch nargin
    case 3
        mu = varargin{3}; % sequence of penalty parameters
    case 4
        mu = varargin{3}; % sequence of penalty parameters
        epsconv = varargin{4}; % convergence parameter
end

% dimension variables
N = size(X2,1);
P = size(X2,2);
K = length(mu);

% some parameters
iter_max = 100; % maximum number of coordinate descent iterations
grid_num = 100; % number of grid values for LLT calculations
beta_max = 10;
beta_grid = (-beta_max : 2*beta_max/grid_num : beta_max)';
delta = 1e-3;   % difference value for derivative of legendre transform

% loop variables
betahat_out = zeros(P,K); % where output will be stored
betahat = zeros(P,1);     % current solution
iter = 0;
diff = 1;

% loop over mu values
for k=1:K
    
    while iter<iter_max && diff>epsconv
        % 1d ground states for every coordinate
        w0 = logistic_potential_1d_deriv(X2,betahat);
        % non-zero coordinates
        ix = w-w0 > mu(k);
        % w shifted for non-zero coordinates
        wsh = w(ix) - sign(w(ix)-w0(ix))*mu(k);
        % grid values for 1d potential for all non-zero coordinates
        u = logistic_potential_1d(X2,betahat,beta_grid,j);
        
        % cycle over all coordinates
        for j=1:P
            j
            w0 = logistic_potential_1d_deriv(X2,betahat,0,j);
            if abs(w(j)-w0) > mu(k)
                wsh = w(j) - sign(w(j)-w0)*mu(k);
                % grid of values for Uj
                u = logistic_potential_1d(X2,betahat,beta_grid,j);
                % Legendre transform at w(j)-sig*mu and w(j)-sig*mu+delta
                [~,lw] = LLTd(beta_grid,u,[wsh; wsh+delta]);
                betahat_new = (lw(2)-lw(1))/delta;
            else
                betahat_new = 0;
            end
            diff = abs(betahat_new-betahat(j));
            betahat(j) = betahat_new;
        end
        
    end
    
    betahat_out(:,k) = betahat;
end

% make output
fit.a0      = betahat_out(1,:);
fit.beta    = betahat_out(2:end,:);
fit.lambda  = mu;                   % regularization parameter sequence
fit.epsconv = epsconv;              % convergence threshold