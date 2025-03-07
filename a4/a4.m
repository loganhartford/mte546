clear; clc;

load('Assignment4_goodness_estimators.mat')

estY = [4.3, 4.0, 6.0];
estX = [3.2, 3.1, 3.3]; 

wOWA = [0.55, 0.30, 0.15];

rmsVals = zeros(1,3);
for i = 1:3
    rowIdx = round(estY(i));
    colIdx = round(estX(i));
    rowIdx = max(1, min(20, rowIdx));
    colIdx = max(1, min(20, colIdx));

    switch i
        case 1
            rmsVals(i) = Est1_RMS_errors(rowIdx, colIdx);
        case 2
            rmsVals(i) = Est2_RMS_errors(rowIdx, colIdx);
        case 3
            rmsVals(i) = Est3_RMS_errors(rowIdx, colIdx);
    end
end

[~, sortIdx] = sort(rmsVals, 'ascend');
ySorted = estY(sortIdx);
xSorted = estX(sortIdx);

fusedY_1 = sum(wOWA .* ySorted);
fusedX_1 = sum(wOWA .* xSorted);

fprintf('Part (1): OWA fused estimate = (y, x) = (%.4f, %.4f)\n', fusedY_1, fusedX_1);

rowFused = round(fusedY_1);
colFused = round(fusedX_1);
rowFused = max(1, min(20, rowFused));
colFused = max(1, min(20, colFused));

rmsVals2 = zeros(1,3);
rmsVals2(1) = Est1_RMS_errors(rowFused, colFused);
rmsVals2(2) = Est2_RMS_errors(rowFused, colFused);
rmsVals2(3) = Est3_RMS_errors(rowFused, colFused);

[~, sortIdx2] = sort(rmsVals2, 'ascend');

ySorted2 = estY(sortIdx2);
xSorted2 = estX(sortIdx2);

fusedY_2 = sum(wOWA .* ySorted2);
fusedX_2 = sum(wOWA .* xSorted2);

fprintf('Part (2): OWA fused estimate (re-evaluated) = (y, x) = (%.4f, %.4f)\n', ...
        fusedY_2, fusedX_2);

weights = [0.15, 0.25, 0.6];
n = length(weights);
orness = sum(((1:n) - 1) ./ (n - 1) .* weights);
fprintf('Orness: %.4f\n', orness);

n = 3; 
alpha = 0.725;
w = meowa_weights(n, alpha);
disp('Max-Entropy OWA weights:');
disp(w);
disp('Check sum and orness:');
disp(['sum(w)= ', num2str(sum(w))]);
orness_val = (1/(n-1)) * sum( (n-(1:n)) .* w );
disp(['Orness(w)= ', num2str(orness_val)]);

function w = meowa_weights(n, alpha)
%MEOWA_WEIGHTS  Compute Max-Entropy OWA weights for n criteria & Orness=alpha.
%
%   w = meowa_weights(n, alpha)
%
%   Returns an n-element row vector w that satisfies:
%     1) w1 >= w2 >= ... >= wn >= 0,
%     2) sum(w) = 1,
%     3) Orness(w) = alpha,
%        where Orness(w) = (1/(n-1)) * sum_{i=1 to n} (n-i) * w_i.
%     4) w maximizes Shannon entropy -sum(w_i*log(w_i)).
%
%   alpha in [0,1].

    if alpha <= 0
        % Entire weight on the last component => "min" aggregator
        w = zeros(1,n); 
        w(n) = 1; 
        return;
    elseif alpha >= 1
        % Entire weight on the first component => "max" aggregator
        w = zeros(1,n); 
        w(1) = 1; 
        return;
    elseif abs(alpha - 0.5) < 1e-14
        % Uniform aggregator => w_i = 1/n
        w = ones(1,n) / n;
        return;
    end

    % We'll solve F(mu) = 0 using fzero
    %  where F(mu) = S1(mu) - alpha*(n-1)*S0(mu).
    %
    %  S0(mu) = sum_{k=0}^{n-1} e^{-mu*k}
    %  S1(mu) = sum_{k=0}^{n-1} k e^{-mu*k}

    F = @(mu) sum1(mu, n) - alpha*(n-1)*sum0(mu, n);

    % A decent initial guess for mu is 0 => alpha=0.5. 
    % If alpha>0.5, we expect mu>0. If alpha<0.5, we expect mu<0.
    % Let's pick a sign based on alpha:
    mu0 = log(alpha/(1-alpha));  % just a guess that moves negative if alpha<0.5, positive if alpha>0.5

    % Solve numerically
    opts = optimset('Display','off');
    [muSol, ~, exitflag] = fzero(@(m) F(m), mu0, opts);

    if exitflag <= 0
        warning('fzero did not converge properly. Using fallback uniform weights.');
        w = ones(1,n)/n;
        return;
    end

    % Now compute final weights
    s0 = sum0(muSol, n);
    A  = 1 / s0;

    w = zeros(1,n);
    for i=1:n
        % i=1 => coefficient = e^{-mu*(n-1)}, i=n => e^{-mu*0}
        w(i) = A * exp(-muSol*(n - i));
    end

    %---- Check descending order: w(1)>=w(2)>=...>=w(n)?
    % For alpha>0.5 => mu>0 => indeed w(1) >= w(2) >= ... >= w(n).
    % For alpha<0.5 => mu<0 => the order flips, so we might want to reverse
    % to keep the "OWA" notion that w_1 is the "largest weight."
    if w(1) < w(n)
        w = fliplr(w);
    end

end

function val = sum0(mu, n)
    % S0(mu) = sum_{k=0}^{n-1} e^{-mu*k}
    k = 0:(n-1);
    val = sum(exp(-mu*k));
end

function val = sum1(mu, n)
    % S1(mu) = sum_{k=0}^{n-1} k e^{-mu*k}
    k = 0:(n-1);
    val = sum(k .* exp(-mu*k));
end
