function u = logistic_potential_1d_deriv(varargin)
% LOGISTIC_POTENTIAL_1D_DERIV - Derivative at 0 of coordinate-wise potential energy function for logistic regression
% LOGISTIC_POTENTIAL_1D_DERIV expects the following input, without
% checking: 
%       - X       : NxP matrix
%       - betahat : Px1 vector
%       - betaj   : double
%       - j       : integer
% and returns a Kx1 vector.

X = varargin{1};
betahat = varargin{2};
switch nargin
    case 2
        Xbj = bsxfun(@times,X,-betahat');
    case 3
        ix = varargin{4};
        Xbj = bsxfun(@times,X(:,ix),-betahat(ix)');
end
Xbmat = bsxfun(@plus,Xbj,X*betahat);
u =  mean(X./(1+exp(-Xbmat)));
