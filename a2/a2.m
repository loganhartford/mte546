%% Load the data
load('Assignment2_data.mat');

% Use y_contaminated as the target variable
y = y_contaminated; 

% Construct the design matrix for a 4th order polynomial
X = [ones(size(x)) x x.^2 x.^3 x.^4];

%% Perform Ordinary Least Squares (OLS)
beta_ols = (X' * X) \ (X' * y);

% Compute predicted y values
y_pred = X * beta_ols;

% Compute R-squared (R²)
SS_tot = sum((y - mean(y)).^2);
SS_res = sum((y - y_pred).^2);
R_squared = 1 - (SS_res / SS_tot);

% Display results
fprintf('Weight for x^4: %.6f\n', beta_ols(5));
fprintf('R-squared: %.6f\n', R_squared);

%% Ridge Regression (L2 Regularization) with lambda = 2
lambda = 2;
I = eye(size(X,2)); % Identity matrix (same size as X'X)
I(1,1) = 0; % Do not regularize the bias term
beta_ridge = (X' * X + lambda * I) \ (X' * y);

% Compute norm ratio ||β_ridge|| / ||β_ols||
norm_ratio = norm(beta_ridge) / norm(beta_ols);

% Display results
fprintf('Ratio ||β_ridge|| / ||β_ols||: %.6f\n', norm_ratio);

%% Solve Weighted Least Squares (Robust Regression)

% Compute residuals from OLS fit
y_pred_ols = X * beta_ols;
residuals = y - y_pred_ols;

% Compute scaling factor h (4 times the standard deviation of absolute residuals)
h = 4 * median(abs(residuals));

% Compute weights using the Bisquare function
w = 1 ./ (1 + (abs(residuals) / h));

% Form diagonal weight matrix
W = diag(w);

% Solve Weighted Least Squares (Robust Regression)
beta_bisquare = (X' * W * X) \ (X' * W * y);

% Compute predictions with the robust regression model
y_pred_bisquare = X * beta_bisquare;

% Compute weighted mean of y
y_mean_weighted = sum(w .* y) / sum(w);

% Compute weighted SSE (Sum of Squared Errors)
SSE = sum(w .* (y - y_pred_bisquare).^2);

% Compute weighted SST (Total Sum of Squares)
SST = sum(w .* (y - y_mean_weighted).^2);

% Compute weighted R^2 using the correct formula
R_squared_weighted = 1 - (SSE / SST);

% Display results
fprintf('Weight for x^4 (Bisquare Method): %.6f\n', beta_bisquare(5));
fprintf('Weighted R² (Bisquare Method): %.6f\n', R_squared_weighted);

%% Find the appropriate degree polynomial

% Define polynomial degrees to test
degrees = 1:6; % Testing polynomial orders from 1 to 10

% Initialize variables to store R² values
R2_train = zeros(length(degrees),1);
R2_test = zeros(length(degrees),1);

% Split data into training (80%) and testing (20%) sets
n = length(x);
idx = randperm(n);
train_size = round(0.8 * n);
x_train = x(idx(1:train_size));
y_train = y(idx(1:train_size));
x_test = x(idx(train_size+1:end));
y_test = y(idx(train_size+1:end));

for i = 1:length(degrees)
    % Construct Vandermonde matrices for given polynomial order
    X_train = ones(train_size, 1);
    X_test = ones(n - train_size, 1);
    for j = 1:degrees(i)
        X_train = [X_train, x_train.^j];
        X_test = [X_test, x_test.^j];
    end
    
    % Solve OLS for training data
    beta = (X_train' * X_train) \ (X_train' * y_train);
    
    % Compute predictions
    y_pred_train = X_train * beta;
    y_pred_test = X_test * beta;
    
    % Compute R² for training data
    SS_tot_train = sum((y_train - mean(y_train)).^2);
    SS_res_train = sum((y_train - y_pred_train).^2);
    R2_train(i) = 1 - (SS_res_train / SS_tot_train);
    
    % Compute R² for testing data
    SS_tot_test = sum((y_test - mean(y_test)).^2);
    SS_res_test = sum((y_test - y_pred_test).^2);
    R2_test(i) = 1 - (SS_res_test / SS_tot_test);
end

% Plot R² vs Polynomial Order
figure;
plot(degrees, R2_train, '-o', 'LineWidth', 1.5, 'DisplayName', 'Training R²');
hold on;
plot(degrees, R2_test, '-o', 'LineWidth', 1.5, 'DisplayName', 'Testing R²');
xlabel('Polynomial Order');
ylabel('R² Value');
legend;
title('Model Selection Based on R²');
grid on;