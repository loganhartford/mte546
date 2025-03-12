clear; clc;

load('Assignment_5_data.mat')

fis = readfis('fis.fis');

X1smooth = smooth(X1, 11);
X2smooth = smooth(X2, 11);
X3smooth = smooth(X3, 11);

% Compute noise profiles
NoiseX1 = X1 - X1smooth;
NoiseX2 = X2 - X2smooth;
NoiseX3 = X3 - X3smooth;

% Compute standard deviation of noise for required windows
STD_Noise_X1_201_300 = std(NoiseX1(1,201:300));
STD_Noise_X2_201_300 = std(NoiseX2(1,201:300));
STD_Noise_X3_201_300 = std(NoiseX3(1,201:300));

STD_Noise_X1_801_900 = std(NoiseX1(1,801:900));
STD_Noise_X2_801_900 = std(NoiseX2(1,801:900));
STD_Noise_X3_801_900 = std(NoiseX3(1,801:900));

% Display results for manual input into the FIS Rule Viewer
fprintf('\n--- Standard Deviations (Input to FIS) ---\n');
fprintf('For window 201-300:\n');
fprintf('  STD_Noise_X1: %.6f\n', STD_Noise_X1_201_300);
fprintf('  STD_Noise_X2: %.6f\n', STD_Noise_X2_201_300);
fprintf('  STD_Noise_X3: %.6f\n', STD_Noise_X3_201_300);

fprintf('[%.6f %.6f %.6f]\n', STD_Noise_X1_201_300, STD_Noise_X2_201_300, STD_Noise_X3_201_300)

fprintf('\nFor window 801-900:\n');
fprintf('  STD_Noise_X1: %.6f\n', STD_Noise_X1_801_900);
fprintf('  STD_Noise_X2: %.6f\n', STD_Noise_X2_801_900);
fprintf('  STD_Noise_X3: %.6f\n', STD_Noise_X3_801_900);

fprintf('[%.6f %.6f %.6f]\n', STD_Noise_X1_801_900, STD_Noise_X2_801_900, STD_Noise_X3_801_900)

w1 = [0.86, 0.5 0.5];
w2 = [0.5 0.5 0.83];

X1_201_300 = mean(X1(201:300));
X2_201_300 = mean(X2(201:300));
X3_201_300 = mean(X3(201:300));

X1_801_900 = mean(X1(801:900));
X2_801_900 = mean(X2(801:900));
X3_801_900 = mean(X3(801:900));

Mean_Xfused_201_300 = (X1_201_300 * w1(1) + X2_201_300 * w1(2) + X3_201_300 * w1(3))/(sum(w1));
Mean_Xfused_801_900 = (X1_801_900 * w2(1) + X2_801_900 * w2(2) + X3_801_900 * w2(3))/(sum(w2));

fprintf('Mean Fused Measurement (201-300): %.6f\n', Mean_Xfused_201_300);
fprintf('Mean Fused Measurement (801-900): %.6f\n', Mean_Xfused_801_900);