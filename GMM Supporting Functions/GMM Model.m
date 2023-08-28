% input:   
%   X      size: N*p
%   mix    num of mix 
% varargin input:
%   restart_num     the times of restart kmeans
%   iter_num        the maximum of EM iteration
%   cov_type        'full' or 'diag'
%   cov_thresh      the minimum of cov
% output:
%   pi              size: M; dim 1: mix num 
%   mu              size: p*M; dim 1: feature dim, dim 2: mix num
%   Sigma           size: p*p*M; dim 1,2: feature dim, dim 3: mix num
%   loglik          log likelihood of current model
% Reference:Chapter 9 <Pattern Analysis and Machine Learning> and MATLAB voicebox toolbox

function [prior, mu, Sigma, loglik] = Gmm(X, mix_num, varargin)

for i1 = 1:2:length(varargin)
    switch varargin{i1}
        case 'cov_type'
            cov_type = varargin{i1+1};
        case 'cov_thresh'
            cov_thresh = varargin{i1+1};
        case 'restart_num'
            restart_num = varargin{i1+1};
        case 'iter_num'
            iter_num = varargin{i1+1};
    end
end
if (~exist('cov_type'))
    cov_type = 'diag';      % 'full' or 'diag'
end
if (~exist('cov_thresh'))
    cov_thresh = 1e-4;      % the thresh of covariance
end
if (~exist('restart_num'))
    restart_num = 1;       % the times of restart kmeans
end
if (~exist('iter_num'))
    iter_num = 100;         % the maximum of EM iteration
end

% Init gmm by kmeans
[pi0, mu0, Sigma0] = Gmm_init_by_kmeans(X, mix_num, restart_num, cov_thresh);

% EM algorighm for Gmm
[prior, mu, Sigma, loglik] = Gmm_em(X, pi0, mu0, Sigma0, iter_num, cov_type, cov_thresh);

% Initializing GMM using K-means

function [pi0, mu0, Sigma0] = Gmm_init_by_kmeans(X, mix_num, restart_num, cov_thresh)
% restart kmeans for several times
[N,p] = size(X);
err = inf;
for i1 = 1:restart_num
    [indi_curr, mu0_curr, errs] = kmeans( X, mix_num );
    err_curr = sum( errs );
    mu0_curr = mu0_curr';
    if err_curr < err
        err = err_curr;
        mu0 = mu0_curr;
        indi = indi_curr;
    end
end

% calculate pi0
pi0 = zeros(1,mix_num);
for i1 = 1:mix_num
    pi0(i1) = sum(indi==i1);
end
pi0 = pi0 / length(indi);

% calculate Sigma0
for i1 = 1:mix_num
    X_curr = X(indi==i1, :);
    mu0_curr = mu0(:,i1);
    Sigma0_curr = cov(bsxfun(@minus, X_curr, mu0_curr'));

    if min(eig(Sigma0_curr)) < cov_thresh    % prevent cov from being too small
        Sigma0(:,:,i1) = Sigma0_curr + cov_thresh * eye(p);
    else
        Sigma0(:,:,i1) = Sigma0_curr;               % dim 1, dim 2: feature dim, dim 3: mix num
    end
end
end

% EM Algorithm for GMM
function [prior, mu, Sigma, loglik] = Gmm_em(X, prior, mu, Sigma, iter_num, cov_type, cov_thresh)
M = length(prior);  % mix num
[N,p] = size(X);
pre_ll = -inf;

for k = 1:iter_num
    k
    % E step
    p_xn_given_zn = zeros(N, M);
    for i1 = 1:M
        p_xn_given_zn(:, i1) = mvnpdf(X, mu(:,i1)', Sigma(:,:,i1));
    end

    numer = bsxfun(@times, p_xn_given_zn, prior);     % dim 1: N, dim 2: mix num
    denor = sum(numer, 2);                  % dim 1: N
    gamma = bsxfun(@rdivide, numer, denor);  % dim 1: N, dim 2: mix num

    % M step
    Nk = sum(gamma, 1);     % dim 1: mix num
    mu = bsxfun(@rdivide, (X' * gamma), Nk);

    for i1 = 1:M
        x_minus_mu = bsxfun(@minus, X, mu(:,i1)');
        Sigma(:,:,i1) = bsxfun(@times, gamma(:,i1), x_minus_mu)' * x_minus_mu / Nk(i1);
        if (cov_type=='diag')
            Sigma(:,:,i1) = diag(diag(Sigma(:,:,i1)));
        end
        if min(eig(Sigma(:,:,i1))) < cov_thresh    % prevent cov from being too small
            Sigma(:,:,i1) = Sigma(:,:,i1) + cov_thresh * eye(p);
        end
        prior = Nk / N;
    end
    
    % Obtain Probability 
    p_xn = zeros(N,1);
    for i1 = 1:M
        p_xn = p_xn + prior(i1) * mvnpdf(X, mu(:,i1)', Sigma(:,:,i1));
    end
    loglik = sum(log(p_xn));
    
    if (loglik-pre_ll<log(1.0001)) break;
    else pre_ll = loglik; end
end
