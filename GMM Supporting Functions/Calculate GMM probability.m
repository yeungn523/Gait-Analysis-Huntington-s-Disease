function probs = Gmmpdf(X, prior, mu, Sigma)
N = size(X,1);          % num of data
[p,M] = size(mu);       % mix num & feature dim
probs = zeros(N,1);     % init output array
for m = 1:M
    probs = probs + prior(m) * mvnpdf(X, mu(:,m)', Sigma(:,:,m));
end
end
