function u = logistic_potential_1d(X,betahat,beta)
% LOGISTIC_POTENTIAL_1D - Coordinate-wise potential energy function for logistic regression
% LOGISTIC_POTENTIAL_1D expects the following input, without checking:
%       - X       : NxP matrix
%       - betahat : Px1 vector
%       - betaj   : Kx1 vector
% and returns a PxK array.

N = size(X,1);
P = size(X,2);
K = size(beta,1);

y1 = X*betahat;                    % y1(i)     = sum_j X(i,j)*betahat(j)
y2 = bsxfun(@times,X,betahat');    % y2(i,j)   = X(i,j)*betahat(j);
y3 = reshape(X(:)*beta',[N,P,K]);  % y3(i,j,k) = X(i,j)*beta(k)

y3 = bsxfun(@minus,y3,y2);
y3 = bsxfun(@plus,y3,y1);          % y3(i,j,k) = y1(i) + y3(i,j,k) - y2(i,j)

u = squeeze(mean(log(1+exp(y3)))); % u = (1/N)*sum_i log(1+e^y3(i,j,k))

% switch nargin
%     case 3
%         Xbj = bsxfun(@times,X,(beta-betahat)');
%     case 4
%         ix = varargin{4};
%         Xbj = bsxfun(@times,X(:,ix),(beta-betahat(ix))');
% end
% Xbmat = bsxfun(@plus,Xbj,X*betahat);
% u = mean(log(1+exp(Xbmat)))';