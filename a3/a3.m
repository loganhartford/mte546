%% Clear workspace and load data
clear; clc;
load('Assignmet3_data.mat'); % Load data

%% Compute target values (assuming Angle_new is in radians)
theta    = Angle_new;
sinTheta = sin(theta);
cosTheta = cos(theta);

%% Number of measurements and sensors
N = length(theta);
numSensors = 3;

%% Preallocate cell array for weights and errors
weights = cell(numSensors, 1);
errors = zeros(N, numSensors);
predictions = zeros(N, numSensors);
variances = zeros(numSensors, 1);

%% Loop over sensors to compute individual estimators
for i = 1:numSensors
    % Extract sensor data
    sensorX = eval(sprintf('AMR%dx', i));
    sensorY = eval(sprintf('AMR%dy', i));
    
    % Regression for sin(theta)
    D_sin = [ones(N,1), sensorX];
    w_sin = D_sin \ sinTheta;
    
    % Regression for cos(theta)
    D_cos = [ones(N,1), sensorY];
    w_cos = D_cos \ cosTheta;
    
    % Store weights
    w = [w_sin(1); w_sin(2); w_cos(1); w_cos(2)];
    weights{i} = w;
    
    % Compute individual sensor prediction
    prediction = atan2(w(2)*sensorX + w(1), w(4)*sensorY + w(3));
    predictions(:, i) = prediction;

    % Compute error
    E = theta - prediction;
    errors(:, i) = E;
    
    % Compute variance of error
    variances(i) = var(E);
    
    % Display results
    fprintf('Sensor %d:\n', i);
    fprintf('  Weights: [%f, %f, %f, %f]\n', w(1), w(2), w(3), w(4));
    fprintf('  Error variance: %f\n', variances(i));
    fprintf('  Mean error: %f\n\n', mean(E));
    
end

%% Compute Weighted Sum Fusion
% Compute weights as inverse of variance
inverse_variances = 1 ./ variances; % w_i = 1 / variance
normalized_weights = inverse_variances / sum(inverse_variances); % Normalize weights


% Compute fused estimate
fused_prediction = predictions * normalized_weights;

% Compute fused error
fused_error = theta - fused_prediction;

% Compute mean error and variance of fused estimator
mean_fused_error = mean(fused_error);
variance_fused_error = var(fused_error);

%% Display Fused Estimator Results
fprintf('Fused Estimator:\n');

fprintf('  Error variance: %f\n', variance_fused_error);
fprintf('  Mean error: %f\n', mean_fused_error);
