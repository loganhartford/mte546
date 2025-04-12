import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import mahalanobis
from numpy.linalg import inv

np.random.seed(42)

inhouse = pd.read_csv("regression_data/inhouse_sensor21.csv")
client21 = pd.read_csv("regression_data/client_sensor21.csv")
client22 = pd.read_csv("regression_data/client_sensor22.csv")

def compute_desired(q1, q2):
    return -0.0081 * q1 + 0.0589 * q2 - 0.000625

inhouse['desired'] = compute_desired(inhouse['q1'], inhouse['q2'])
client21['desired'] = compute_desired(client21['q1'], client21['q2'])
client22['desired'] = compute_desired(client22['q1'], client22['q2'])

combined = pd.concat([inhouse, client21], ignore_index=True)

n_iter = 100
n_samples = 1000
coefficients = []

X_full = combined[['v1', 'v2']].values
y_full = combined['desired'].values

for _ in range(n_iter):
    indices = np.random.choice(len(X_full), size=n_samples, replace=True)
    X_sample = X_full[indices]
    y_sample = y_full[indices]
    
    model = LinearRegression().fit(X_sample, y_sample)
    coefficients.append(np.append(model.coef_, model.intercept_))

coefficients = np.array(coefficients)

plt.figure(figsize=(12, 4))
labels = ['v1 coefficient', 'v2 coefficient', 'intercept']
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.hist(coefficients[:, i], bins=20, alpha=0.7)
    plt.title(f"Bootstrap: {labels[i]}")
    plt.grid(True)

plt.tight_layout()
plt.savefig("bootstrap_histograms.png")

mean_coeff = np.mean(coefficients, axis=0)
cov_coeff = np.cov(coefficients.T)

X_22 = client22[['v1', 'v2']].values
y_22 = client22['desired'].values
model_22 = LinearRegression().fit(X_22, y_22)
coeff_22 = np.append(model_22.coef_, model_22.intercept_)

inv_cov = inv(cov_coeff)
distance = mahalanobis(coeff_22, mean_coeff, inv_cov)

print("Mean coefficient vector (from bootstrap):", mean_coeff)
print("Covariance matrix of coefficients:\n", cov_coeff)
print("Coefficients from client_sensor_22:", coeff_22)
print(f"Mahalanobis distance: {distance:.4f}")
